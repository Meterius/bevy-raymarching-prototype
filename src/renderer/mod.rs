use crate::bindings::cuda::MAX_SUN_LIGHT_COUNT;
use crate::bindings::cuda::{
    ConeMarchTextureValue, RenderDataTextureValue, SdSphere, CONE_MARCH_LEVELS, MAX_SPHERE_COUNT,
};
use bevy::core_pipeline::clear_color::ClearColorConfig;
use bevy::render::extract_resource::ExtractResource;
use bevy::window::PrimaryWindow;
use bevy::{prelude::*, render::render_resource::*};
use cudarc::driver::{
    result, sys, CudaDevice, CudaSlice, CudaStream, DevicePtr, LaunchAsync, LaunchConfig,
};
use cudarc::nvrtc::Ptx;
use nvtx::mark;
use std::io::Read;
use std::sync::Arc;

const RENDER_TEXTURE_SIZE: (usize, usize) = (2560, 1440);

const CUDA_GPU_BLOCK_SIZE: usize = 128;

#[derive(Debug, Clone, Default, Resource, Reflect)]
#[reflect(Resource)]
pub struct RenderSettings {
    lockstep_enabled: bool,
}

#[derive(Debug, Clone, Resource, Reflect)]
#[reflect(Resource)]
pub struct RenderConeCompression {
    pub enabled: bool,
    pub levels: [usize; CONE_MARCH_LEVELS as usize],
}

impl Default for RenderConeCompression {
    fn default() -> Self {
        Self {
            enabled: true,
            levels: [16, 8, 4, 2],
        }
    }
}

#[derive(Debug, Clone, Default, Component)]
pub struct RenderCameraTarget {}

#[derive(Clone, Debug, Default, Component)]
pub struct RenderTargetSprite {}

#[derive(Clone, Resource, ExtractResource, Deref)]
struct RenderTargetImage(Handle<Image>);

struct RenderCudaContext {
    device: Arc<CudaDevice>,
}

struct RenderCudaStreams {
    render_stream: CudaStream,
    compression_stream: CudaStream,
}

struct RenderCudaBuffers {
    render_data_texture_buffer: CudaSlice<RenderDataTextureValue>,
    render_texture_buffer: CudaSlice<u8>,

    cm_texture_buffers: [Vec<CudaSlice<ConeMarchTextureValue>>; 2],
    rt_sphere_buffer: [CudaSlice<SdSphere>; 2],
}

// App Systems

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let mut image = Image::new_fill(
        Extent3d {
            width: RENDER_TEXTURE_SIZE.0 as u32,
            height: RENDER_TEXTURE_SIZE.1 as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8Unorm,
    );
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let image = images.add(image);

    commands.spawn((
        SpriteBundle {
            sprite: Sprite {
                color: Color::rgba(1.0, 1.0, 1.0, 1.0),
                custom_size: Some(Vec2::new(
                    RENDER_TEXTURE_SIZE.0 as f32,
                    RENDER_TEXTURE_SIZE.1 as f32,
                )),
                ..default()
            },
            texture: image.clone(),
            ..default()
        },
        RenderTargetSprite::default(),
    ));
    commands.spawn(Camera2dBundle {
        camera: Camera {
            order: 1,
            ..default()
        },
        camera_2d: Camera2d {
            clear_color: ClearColorConfig::None,
            ..default()
        },
        ..default()
    });

    commands.insert_resource(RenderTargetImage(image));
}

// Render Systems

fn setup_cuda(world: &mut World) {
    let compression = RenderConeCompression::default();

    info!("Ray Marcher Cone Marching Compression: {compression:?}");

    let start = std::time::Instant::now();

    let device = CudaDevice::new(0).unwrap();

    info!("CUDA Device Creation took {:.2?} seconds", start.elapsed());

    let start = std::time::Instant::now();

    let ptx = Ptx::from_src(include_str!("../../assets/cuda/compiled/main.ptx"));
    device
        .load_ptx(
            ptx,
            "main",
            &[
                "compute_compressed_depth",
                "compute_render",
                "compute_render_finalize",
            ],
        )
        .unwrap();

    info!("CUDA PTX Loading took {:.2?} seconds", start.elapsed());

    let render_texture_buffer = unsafe {
        device
            .alloc::<u8>(4 * RENDER_TEXTURE_SIZE.0 * RENDER_TEXTURE_SIZE.1)
            .unwrap()
    };

    let render_data_texture_buffer = unsafe {
        device
            .alloc::<RenderDataTextureValue>(RENDER_TEXTURE_SIZE.0 * RENDER_TEXTURE_SIZE.1)
            .unwrap()
    };

    let rt_sphere_buffer = unsafe {
        [
            device.alloc::<SdSphere>(1024).unwrap(),
            device.alloc::<SdSphere>(1024).unwrap(),
        ]
    };

    let mut cm_texture_buffers = [vec![], vec![]];

    for i in 0..CONE_MARCH_LEVELS as usize {
        cm_texture_buffers[0].push(unsafe {
            device
                .alloc::<ConeMarchTextureValue>(
                    (RENDER_TEXTURE_SIZE.0 as f32 / compression.levels[i] as f32).ceil() as usize
                        * (RENDER_TEXTURE_SIZE.1 as f32 / compression.levels[i] as f32).ceil()
                            as usize,
                )
                .unwrap()
        });

        cm_texture_buffers[1].push(unsafe {
            device
                .alloc::<ConeMarchTextureValue>(
                    (RENDER_TEXTURE_SIZE.0 as f32 / compression.levels[i] as f32).ceil() as usize
                        * (RENDER_TEXTURE_SIZE.1 as f32 / compression.levels[i] as f32).ceil()
                            as usize,
                )
                .unwrap()
        });
    }

    let render_stream = device.fork_default_stream().unwrap();
    let compression_stream = device.fork_default_stream().unwrap();

    world.insert_resource(RenderSettings::default());
    world.insert_resource(compression);
    world.insert_non_send_resource(RenderCudaContext { device });
    world.insert_non_send_resource(RenderCudaStreams {
        render_stream,
        compression_stream,
    });
    world.insert_non_send_resource(RenderCudaBuffers {
        render_texture_buffer,
        render_data_texture_buffer,
        cm_texture_buffers,
        rt_sphere_buffer,
    });
}

struct RenderParameters {
    globals: crate::bindings::cuda::GlobalsBuffer,
    camera: crate::bindings::cuda::CameraBuffer,
    sd_runtime_scene: crate::bindings::cuda::SdRuntimeScene,
    cm_textures: crate::bindings::cuda::ConeMarchTextures,
    render_texture: crate::bindings::cuda::Texture,
    render_data_texture: crate::bindings::cuda::RenderDataTexture,
}

#[derive(Default)]
struct PreviousRenderParameter {
    previous: Option<RenderParameters>,
}

fn render(
    settings: Res<RenderSettings>,
    compression: Res<RenderConeCompression>,
    time: Res<Time>,
    camera: Query<(&Camera, &Projection, &GlobalTransform), With<RenderCameraTarget>>,
    render_context: NonSend<RenderCudaContext>,
    mut render_streams: NonSendMut<RenderCudaStreams>,
    mut render_buffers: NonSendMut<RenderCudaBuffers>,
    render_target_image: Res<RenderTargetImage>,
    mut images: ResMut<Assets<Image>>,

    mut tick: Local<u64>,

    mut previous_render_parameters: NonSendMut<PreviousRenderParameter>,
) {
    let parity = tick.rem_euclid(2);

    let image = images.get_mut(&render_target_image.0).unwrap();
    let (cam, cam_projection, cam_transform) = camera.single();

    if *tick == 0 {
        unsafe {
            cudarc::driver::sys::cuMemHostRegister_v2(
                image.data.as_mut_ptr() as *mut _,
                image.data.as_mut_slice().len(),
                0,
            )
        };
    }

    nvtx::mark!("Sync");

    unsafe {
        cudarc::driver::result::stream::synchronize(render_streams.render_stream.stream).unwrap()
    };

    // Render Parameters

    let globals = crate::bindings::cuda::GlobalsBuffer {
        time: time.elapsed_seconds(),
        tick: tick.clone(),
        render_texture_size: [RENDER_TEXTURE_SIZE.0 as u32, RENDER_TEXTURE_SIZE.1 as u32],
        render_screen_size: [
            cam.logical_viewport_size().map(|s| s.x).unwrap_or(1.0) as _,
            cam.logical_viewport_size().map(|s| s.y).unwrap_or(1.0) as _,
        ],
    };

    let camera = crate::bindings::cuda::CameraBuffer {
        position: cam_transform.translation().as_ref().clone(),
        forward: cam_transform.forward().as_ref().clone(),
        up: cam_transform.up().as_ref().clone(),
        right: cam_transform.right().as_ref().clone(),
        fov: match cam_projection {
            Projection::Perspective(perspective) => perspective.fov,
            Projection::Orthographic(_) => 1.0,
        },
    };

    let mut spheres = Vec::with_capacity(MAX_SPHERE_COUNT as usize);

    for i in 0..MAX_SPHERE_COUNT as usize {
        spheres.push(SdSphere {
            translation: [
                i as f32 * 0.7,
                (time.elapsed_seconds() * (i as f32 / 10.0)).sin() * (i as f32 / 25.0),
                (time.elapsed_seconds() * (i as f32 / 10.0)).cos() * (i as f32 / 25.0),
            ],
            radius: 0.3,
        });
    }

    let mut sun_lights = Vec::with_capacity(MAX_SUN_LIGHT_COUNT as usize);

    for i in 0..MAX_SUN_LIGHT_COUNT as usize {
        sun_lights.push(crate::bindings::cuda::SunLight {
            direction: Vec3::new(1.0, -1.0, 0.2).normalize().into(),
        });
    }

    let sd_runtime_scene = crate::bindings::cuda::SdRuntimeScene {
        spheres: spheres.try_into().unwrap(),
        sphere_count: 0, // MAX_SPHERE_COUNT as i32,
        lighting: crate::bindings::cuda::SdRuntimeSceneLighting {
            sun_light_count: 0,
            sun_lights: sun_lights.try_into().unwrap(),
        },
    };

    let render_texture = crate::bindings::cuda::Texture {
        texture: unsafe {
            std::mem::transmute(*(&render_buffers.render_texture_buffer).device_ptr())
        },
        size: [RENDER_TEXTURE_SIZE.0 as _, RENDER_TEXTURE_SIZE.1 as _],
    };

    let render_data_texture = crate::bindings::cuda::RenderDataTexture {
        texture: unsafe {
            std::mem::transmute(*(&render_buffers.render_data_texture_buffer).device_ptr())
        },
        size: [RENDER_TEXTURE_SIZE.0 as _, RENDER_TEXTURE_SIZE.1 as _],
    };

    let mut cm_textures = vec![];

    for i in 0..CONE_MARCH_LEVELS as u32 {
        cm_textures.push(crate::bindings::cuda::ConeMarchTexture {
            texture: unsafe {
                std::mem::transmute(
                    *(&render_buffers.cm_texture_buffers[parity as usize][i as usize]).device_ptr(),
                )
            },
            size: [
                (RENDER_TEXTURE_SIZE.0 as f32 / compression.levels[i as usize] as f32).ceil()
                    as u32,
                (RENDER_TEXTURE_SIZE.1 as f32 / compression.levels[i as usize] as f32).ceil()
                    as u32,
            ],
        });
    }

    let cm_textures = crate::bindings::cuda::ConeMarchTextures {
        textures: cm_textures.try_into().unwrap(),
    };

    let parameters = RenderParameters {
        cm_textures,
        camera,
        globals,
        render_data_texture,
        render_texture,
        sd_runtime_scene,
    };

    //

    nvtx::mark!("Invoke");

    if compression.enabled {
        for i in 0..CONE_MARCH_LEVELS as u32 {
            let grid_size = ((cm_textures.textures[i as usize].size[0]
                * cm_textures.textures[i as usize].size[0]) as f32
                / CUDA_GPU_BLOCK_SIZE as f32)
                .ceil() as u32;

            unsafe {
                render_context
                    .device
                    .get_func("main", "compute_compressed_depth")
                    .unwrap()
                    .launch_on_stream(
                        if settings.lockstep_enabled {
                            &render_streams.compression_stream
                        } else {
                            &render_streams.render_stream
                        },
                        LaunchConfig {
                            block_dim: (CUDA_GPU_BLOCK_SIZE as u32, 1, 1),
                            grid_dim: (grid_size, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            i,
                            parameters.render_data_texture.clone(),
                            parameters.cm_textures.clone(),
                            parameters.globals.clone(),
                            parameters.camera.clone(),
                            parameters.sd_runtime_scene.clone(),
                        ),
                    )
                    .unwrap()
            };
        }
    }

    let render_parameters = if settings.lockstep_enabled {
        previous_render_parameters.previous.as_ref()
    } else {
        Some(&parameters)
    };

    if let Some(render_parameters) = render_parameters {
        unsafe {
            render_context
                .device
                .get_func("main", "compute_render")
                .unwrap()
                .launch_on_stream(
                    &render_streams.render_stream,
                    LaunchConfig {
                        block_dim: (CUDA_GPU_BLOCK_SIZE as u32, 1, 1),
                        grid_dim: (
                            (RENDER_TEXTURE_SIZE.1 as u32 * RENDER_TEXTURE_SIZE.0 as u32)
                                / (CUDA_GPU_BLOCK_SIZE as u32),
                            1,
                            1,
                        ),
                        shared_mem_bytes: 0,
                    },
                    (
                        render_parameters.render_data_texture.clone(),
                        render_parameters.cm_textures.clone(),
                        render_parameters.globals.clone(),
                        render_parameters.camera.clone(),
                        render_parameters.sd_runtime_scene.clone(),
                        compression.enabled,
                    ),
                )
                .unwrap()
        };

        unsafe {
            render_context
                .device
                .get_func("main", "compute_render_finalize")
                .unwrap()
                .launch_on_stream(
                    &render_streams.render_stream,
                    LaunchConfig {
                        block_dim: (CUDA_GPU_BLOCK_SIZE as u32, 1, 1),
                        grid_dim: (
                            (RENDER_TEXTURE_SIZE.1 as u32 * RENDER_TEXTURE_SIZE.0 as u32)
                                / (CUDA_GPU_BLOCK_SIZE as u32),
                            1,
                            1,
                        ),
                        shared_mem_bytes: 0,
                    },
                    (
                        render_parameters.render_texture.clone(),
                        render_parameters.render_data_texture.clone(),
                        render_parameters.globals.clone(),
                        compression.enabled,
                    ),
                )
                .unwrap()
        };

        unsafe {
            cudarc::driver::result::memcpy_dtoh_async(
                image.data.as_mut_slice(),
                *render_buffers.render_texture_buffer.device_ptr(),
                render_streams.render_stream.stream,
            )
            .unwrap()
        };
    }

    previous_render_parameters.previous = if settings.lockstep_enabled {
        Some(parameters)
    } else {
        None
    };
    *tick += 1;

    mark!("Invoke Completed");
}

// Synchronization

fn synchronize_target_sprite(
    mut sprite: Query<&mut Transform, With<RenderTargetSprite>>,
    window: Query<&Window, With<PrimaryWindow>>,
) {
    sprite.single_mut().scale = Vec2::new(
        window.single().width() / (RENDER_TEXTURE_SIZE.0 as f32),
        window.single().height() / (RENDER_TEXTURE_SIZE.1 as f32),
    )
    .extend(1.0);
}

// Render Systems

// Render Pipeline

// Plugin

#[derive(Default)]
pub struct RayMarcherRenderPlugin {}

impl Plugin for RayMarcherRenderPlugin {
    fn build(&self, app: &mut App) {
        // Main App Build

        app.world
            .insert_non_send_resource(PreviousRenderParameter::default());

        app.register_type::<RenderConeCompression>()
            .add_systems(Startup, (setup, setup_cuda))
            .add_systems(Last, render)
            .add_systems(PostUpdate, (synchronize_target_sprite,));
    }
}
