pub mod scene;

use crate::bindings::cuda::{
    RenderDataTextureValue, BLOCK_SIZE,
};
use crate::cudarc_extension::CustomCudaFunction;
use bevy::core_pipeline::clear_color::ClearColorConfig;
use bevy::render::extract_resource::ExtractResource;
use bevy::window::PrimaryWindow;
use bevy::{prelude::*, render::render_resource::*};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

const RENDER_TEXTURE_SIZE: (usize, usize) = (2560, 1440);

#[derive(Debug, Clone, Default, Resource, Reflect)]
#[reflect(Resource)]
pub struct RenderSettings {}

#[derive(Debug, Clone, Default, Component)]
pub struct RenderCameraTarget {}

#[derive(Debug, Clone, Default, Component)]
pub struct RenderRelayCameraTarget {}

#[derive(Clone, Debug, Default, Component)]
pub struct RenderTargetSprite {}

#[derive(Clone, Resource, ExtractResource, Deref)]
struct RenderTargetImage(Handle<Image>);

#[derive(Clone, Resource, ExtractResource, Deref)]
struct EnvironmentImage(Handle<Image>);

struct RenderCudaContext {
    #[allow(dead_code)]
    pub device: Arc<CudaDevice>,

    pub func_compute_render: CustomCudaFunction,
    pub func_compute_render_finalize: CustomCudaFunction,
}

struct RenderCudaStreams {
    render_stream: CudaStream,
}

struct RenderCudaBuffers {
    render_data_texture_buffer: CudaSlice<RenderDataTextureValue>,
    render_data_texture_buffer_size: [usize; 2],
    render_texture_buffer: CudaSlice<u8>,
    render_texture_buffer_size: [usize; 2],
    environment_texture_buffer: CudaSlice<u8>,
    environment_texture_buffer_size: [usize; 2],
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

    let env_image_path = "assets/images/loc00184-22-8k.exr";
    let env_image = exr::image::read::read_first_rgba_layer_from_file(
        env_image_path,
        |size, _| {
            let mut image = Image::new_fill(
                Extent3d {
                    width: size.0 as u32,
                    height: size.1 as u32,
                    depth_or_array_layers: 1,
                },
                TextureDimension::D2,
                &[0, 0, 0, 255],
                TextureFormat::Rgba8Unorm,
            );

            image.texture_descriptor.usage =
                TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;

            return image;
        },
        |image, pos, val: (f32, f32, f32, f32)| {
            let idx = 4 * (image.texture_descriptor.size.width as usize * pos.1 + pos.0);
            image.data[idx] = (val.0 * 255.0f32) as u8;
            image.data[idx + 1] = (val.1 * 255.0f32) as u8;
            image.data[idx + 2] = (val.2 * 255.0f32) as u8;
            image.data[idx + 3] = 255;
        },
    ).unwrap().layer_data.channel_data.pixels;

    info!("Loaded environment image from \"{env_image_path}\" with size {:?}", env_image.texture_descriptor.size);

    let env_image = images.add(env_image);

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
    commands.spawn((
        Camera2dBundle {
            camera: Camera {
                order: 1,
                ..default()
            },
            camera_2d: Camera2d {
                clear_color: ClearColorConfig::None,
                ..default()
            },
            ..default()
        },
        RenderRelayCameraTarget::default(),
    ));

    commands.insert_resource(RenderTargetImage(image));
    commands.insert_resource(EnvironmentImage(env_image));
}

// Render Systems

fn setup_cuda(world: &mut World) {
    let start = std::time::Instant::now();

    let device = CudaDevice::new(0).unwrap();

    info!("CUDA Device Creation took {:.2?} seconds", start.elapsed());

    let start = std::time::Instant::now();

    device
        .load_ptx(
            Ptx::from_src(include_str!(
                "../../assets/cuda/compiled/compute_render.ptx"
            )),
            "compute_render",
            &["compute_render"],
        )
        .unwrap();

    device
        .load_ptx(
            Ptx::from_src(include_str!(
                "../../assets/cuda/compiled/compute_render_finalize.ptx"
            )),
            "compute_render_finalize",
            &["compute_render_finalize"],
        )
        .unwrap();

    let func_compute_render = CustomCudaFunction::from_safe(
        device.get_func("compute_render", "compute_render").unwrap(),
        device.clone(),
    );
    let func_compute_render_finalize = CustomCudaFunction::from_safe(
        device
            .get_func("compute_render_finalize", "compute_render_finalize")
            .unwrap(),
        device.clone(),
    );

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

    let env_image = world.resource::<Assets<Image>>().get(&world.resource::<EnvironmentImage>().0).unwrap();
    let environment_texture_buffer = unsafe {
        device
            .alloc::<u8>(env_image.data.len())
            .unwrap()
    };
    let environment_texture_size = [env_image.texture_descriptor.size.width as usize, env_image.texture_descriptor.size.height as usize];

    let render_stream = device.fork_default_stream().unwrap();

    world.insert_resource(RenderSettings::default());
    world.insert_non_send_resource(RenderCudaContext {
        device,
        func_compute_render,
        func_compute_render_finalize,
    });
    world.insert_non_send_resource(RenderCudaStreams { render_stream });
    world.insert_non_send_resource(RenderCudaBuffers {
        render_texture_buffer,
        render_texture_buffer_size: [RENDER_TEXTURE_SIZE.0 as _, RENDER_TEXTURE_SIZE.1 as _],
        render_data_texture_buffer,
        render_data_texture_buffer_size: [RENDER_TEXTURE_SIZE.0 as _, RENDER_TEXTURE_SIZE.1 as _],
        environment_texture_buffer,
        environment_texture_buffer_size: environment_texture_size,
    });
    world.insert_non_send_resource(PreviousRenderParameter::default());
}

struct RenderParameters {
    globals: crate::bindings::cuda::GlobalsBuffer,
    camera: crate::bindings::cuda::CameraBuffer,
    scene: crate::bindings::cuda::SceneBuffer,
    render_texture: crate::bindings::cuda::Texture,
    render_data_texture: crate::bindings::cuda::RenderDataTexture,
}

#[derive(Default)]
struct PreviousRenderParameter {
    previous: Option<RenderParameters>,
}

fn render(
    time: Res<Time>,
    camera: Query<(&Camera, &Projection, &GlobalTransform), With<RenderCameraTarget>>,
    render_context: NonSend<RenderCudaContext>,
    render_streams: NonSendMut<RenderCudaStreams>,
    render_buffers: NonSendMut<RenderCudaBuffers>,
    environment_image: Res<EnvironmentImage>,
    render_target_image: Res<RenderTargetImage>,
    mut images: ResMut<Assets<Image>>,

    mut tick: Local<u64>,

    mut previous_render_parameters: NonSendMut<PreviousRenderParameter>,
) {
    let range_id = nvtx::range_start!("Render System Wait For Previous Frame");

    if *tick == 0 {
        let env_image = images.get_mut(&environment_image.0).unwrap();

        unsafe {
            cudarc::driver::sys::cuMemHostRegister_v2(
                env_image.data.as_mut_ptr() as *mut _,
                env_image.data.as_mut_slice().len(),
                0,
            )
            .result()
            .unwrap()
        };

        unsafe {
            cudarc::driver::result::memcpy_htod_sync(
                *render_buffers.environment_texture_buffer.device_ptr(),
                env_image.data.as_slice(),
            )
            .unwrap()
        };
    }

    let image = images.get_mut(&render_target_image.0).unwrap();
    let (cam, cam_projection, cam_transform) = camera.single();

    if *tick == 0 {
        unsafe {
            cudarc::driver::sys::cuMemHostRegister_v2(
                image.data.as_mut_ptr() as *mut _,
                image.data.as_mut_slice().len(),
                0,
            )
            .result()
            .unwrap()
        };
    }

    unsafe {
        cudarc::driver::result::stream::synchronize(render_streams.render_stream.stream).unwrap()
    };

    nvtx::range_end!(range_id);

    let range_id = nvtx::range_start!("Render System Invoke");

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

    let scene = crate::bindings::cuda::SceneBuffer {
        sun: crate::bindings::cuda::SunLight {
            direction: Vec3::new(1.0f32, -1.0f32, 1.0f32).normalize().as_ref().clone(),
            color: Vec3::new(1.0f32, 1.0f32, 1.0f32).as_ref().clone(),
            intensity: 20.0f32,
        },
        point_lights: [crate::bindings::cuda::PointLight {
            position: Vec3::ZERO.as_ref().clone(),
            color: Vec3::ZERO.as_ref().clone(),
            intensity: 0.0f32,
        }; 8],
        point_light_count: 0,
        environment_texture: crate::bindings::cuda::Texture {
            texture: unsafe {
                std::mem::transmute(*(&render_buffers.environment_texture_buffer).device_ptr())
            },
            size: [render_buffers.environment_texture_buffer_size[0] as _, render_buffers.environment_texture_buffer_size[1] as _],
        },
    };

    let render_texture = crate::bindings::cuda::Texture {
        texture: unsafe {
            std::mem::transmute(*(&render_buffers.render_texture_buffer).device_ptr())
        },
        size: [render_buffers.render_texture_buffer_size[0] as _, render_buffers.render_texture_buffer_size[1] as _],
    };

    let render_data_texture = crate::bindings::cuda::RenderDataTexture {
        texture: unsafe {
            std::mem::transmute(*(&render_buffers.render_data_texture_buffer).device_ptr())
        },
        size: [render_buffers.render_data_texture_buffer_size[0] as _, render_buffers.render_data_texture_buffer_size[1] as _],
    };

    let parameters = RenderParameters {
        camera,
        globals,
        scene,
        render_data_texture,
        render_texture,
    };

    //

    let render_parameters = Some(&parameters);

    if let Some(render_parameters) = render_parameters {
        unsafe {
            render_context
                .func_compute_render
                .clone()
                .launch_on_stream(
                    &render_streams.render_stream,
                    LaunchConfig {
                        block_dim: (BLOCK_SIZE as usize as u32, 1, 1),
                        grid_dim: (
                            (RENDER_TEXTURE_SIZE.1 as u32 * RENDER_TEXTURE_SIZE.0 as u32)
                                / (BLOCK_SIZE as usize as u32),
                            1,
                            1,
                        ),
                        shared_mem_bytes: 0,
                    },
                    (
                        render_parameters.render_data_texture.clone(),
                        render_parameters.globals.clone(),
                        render_parameters.camera.clone(),
                        render_parameters.scene.clone(),
                    ),
                )
                .unwrap()
        };

        unsafe {
            render_context
                .func_compute_render_finalize
                .clone()
                .launch_on_stream(
                    &render_streams.render_stream,
                    LaunchConfig {
                        block_dim: (BLOCK_SIZE as usize as u32, 1, 1),
                        grid_dim: (
                            (RENDER_TEXTURE_SIZE.1 as u32 * RENDER_TEXTURE_SIZE.0 as u32)
                                / (BLOCK_SIZE as usize as u32),
                            1,
                            1,
                        ),
                        shared_mem_bytes: 0,
                    },
                    (
                        render_parameters.render_texture.clone(),
                        render_parameters.render_data_texture.clone(),
                        render_parameters.globals.clone(),
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

    previous_render_parameters.previous = None;
    *tick += 1;

    nvtx::range_end!(range_id);
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
        app.add_systems(Startup, setup)
            .add_systems(PostStartup, setup_cuda)
            .add_systems(Last, render)
            .add_systems(PostUpdate, (synchronize_target_sprite,));
    }
}
