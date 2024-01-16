use crate::bindings::cuda::{ConeMarchTextureValue, SdRuntimeScene, SdSphere};
use bevy::core_pipeline::clear_color::ClearColorConfig;
use bevy::render::extract_resource::ExtractResource;
use bevy::window::PrimaryWindow;
use bevy::{prelude::*, render::render_resource::*};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

const RENDER_TEXTURE_SIZE: (usize, usize) = (2560, 1440);

const CUDA_GPU_BLOCK_SIZE: usize = 64;
const CUDA_GPU_HARDWARE_MAX_PARALLEL_BLOCK_COUNT: usize = 128;

const CONE_MARCH_LEVELS: usize = 4;

#[derive(Debug, Clone, Resource)]
pub struct RenderConeCompression {
    enabled: bool,
    levels: [usize; CONE_MARCH_LEVELS],
}

impl Default for RenderConeCompression {
    fn default() -> Self {
        Self {
            enabled: true,
            levels: [16, 8, 4, 2], /*[
                                       // to improve resource utilization ensure compression
                                       // is chosen such that every thread can be run in parallel
                                       (RENDER_TEXTURE_SIZE.0 as f32 * RENDER_TEXTURE_SIZE.1 as f32
                                           / (CUDA_GPU_BLOCK_SIZE as f32
                                               * CUDA_GPU_HARDWARE_MAX_PARALLEL_BLOCK_COUNT as f32))
                                           .sqrt()
                                           .floor() as usize,
                                   ],*/
        }
    }
}

#[derive(Debug, Clone, Default, Component)]
pub struct RenderCameraTarget {}

#[derive(Clone, Debug, Default, Component)]
pub struct RenderTargetSprite {}

#[derive(Clone, Resource, ExtractResource, Deref)]
struct RenderTargetImage(Handle<Image>);

#[derive(Deref)]
struct RenderCudaDevice(Arc<CudaDevice>);

struct RenderCudaBuffers {
    render_texture_buffer: CudaSlice<u8>,
    cm_texture_buffers: Vec<CudaSlice<ConeMarchTextureValue>>,
    rt_sphere_buffer: CudaSlice<SdSphere>,
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
        .load_ptx(ptx, "main", &["render", "render_depth"])
        .unwrap();

    info!("CUDA PTX Loading took {:.2?} seconds", start.elapsed());

    let render_texture_buffer = unsafe {
        device
            .alloc::<u8>(4 * RENDER_TEXTURE_SIZE.0 * RENDER_TEXTURE_SIZE.1)
            .unwrap()
    };

    let rt_sphere_buffer = unsafe { device.alloc::<SdSphere>(1024).unwrap() };

    let mut cm_texture_buffers = vec![];

    for i in 0..CONE_MARCH_LEVELS {
        cm_texture_buffers.push(unsafe {
            device
                .alloc::<ConeMarchTextureValue>(
                    (RENDER_TEXTURE_SIZE.0 as f32 / compression.levels[i] as f32).ceil() as usize
                        * (RENDER_TEXTURE_SIZE.1 as f32 / compression.levels[i] as f32).ceil()
                            as usize,
                )
                .unwrap()
        });
    }

    world.insert_resource(compression);
    world.insert_non_send_resource(RenderCudaDevice(device));
    world.insert_non_send_resource(RenderCudaBuffers {
        render_texture_buffer,
        cm_texture_buffers,
        rt_sphere_buffer,
    });
}

fn render(
    compression: Res<RenderConeCompression>,
    time: Res<Time>,
    camera: Query<(&Camera, &Projection, &GlobalTransform), With<RenderCameraTarget>>,
    render_device: NonSend<RenderCudaDevice>,
    mut render_buffers: NonSendMut<RenderCudaBuffers>,
    render_target_image: Res<RenderTargetImage>,
    mut images: ResMut<Assets<Image>>,
    mut tick: Local<u64>,
) {
    *tick += 1;

    let image = images.get_mut(&render_target_image.0).unwrap();
    let (cam, cam_projection, cam_transform) = camera.single();

    let mut spheres = Vec::with_capacity(1024);

    for i in 0..1024 {
        spheres.push(SdSphere {
            translation: [
                i as f32 * 0.7,
                (time.elapsed_seconds() * (i as f32 / 10.0)).sin() * (i as f32 / 25.0),
                (time.elapsed_seconds() * (i as f32 / 10.0)).cos() * (i as f32 / 25.0),
            ],
            radius: 0.3,
        });
    }

    render_device
        .0
        .htod_copy_into(spheres, &mut render_buffers.rt_sphere_buffer)
        .unwrap();

    render_device
        .0
        .dtoh_sync_copy_into(
            &render_buffers.render_texture_buffer,
            image.data.as_mut_slice(),
        )
        .unwrap();

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

    unsafe {
        let render_texture = crate::bindings::cuda::Texture {
            texture: std::mem::transmute(*(&render_buffers.render_texture_buffer).device_ptr()),
            size: [RENDER_TEXTURE_SIZE.0 as _, RENDER_TEXTURE_SIZE.1 as _],
        };

        let sd_runtime_scene = crate::bindings::cuda::SdRuntimeScene {
            spheres: std::mem::transmute(*(&render_buffers.rt_sphere_buffer).device_ptr()),
            sphere_count: 1024,
        };

        let mut cm_textures = vec![];

        for i in 0..CONE_MARCH_LEVELS as u32 {
            cm_textures.push(crate::bindings::cuda::ConeMarchTexture {
                texture: std::mem::transmute(
                    *(&render_buffers.cm_texture_buffers[i as usize]).device_ptr(),
                ),
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

        if compression.enabled {
            for i in 0..CONE_MARCH_LEVELS as u32 {
                let grid_size = ((cm_textures.textures[i as usize].size[0]
                    * cm_textures.textures[i as usize].size[0])
                    as f32
                    / CUDA_GPU_BLOCK_SIZE as f32)
                    .round() as u32;

                render_device
                    .0
                    .get_func("main", "render_depth")
                    .unwrap()
                    .launch(
                        LaunchConfig {
                            block_dim: (CUDA_GPU_BLOCK_SIZE as u32, 1, 1),
                            grid_dim: (grid_size, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            i,
                            render_texture.clone(),
                            cm_textures.clone(),
                            globals.clone(),
                            camera.clone(),
                            sd_runtime_scene.clone(),
                        ),
                    )
                    .unwrap();
            }
        }

        render_device
            .0
            .get_func("main", "render")
            .unwrap()
            .launch(
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
                    render_texture.clone(),
                    cm_textures.clone(),
                    globals.clone(),
                    camera.clone(),
                    sd_runtime_scene.clone(),
                ),
            )
            .unwrap();
    }
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
        app.add_systems(Startup, (setup, setup_cuda));
        app.add_systems(Last, render);
        app.add_systems(PostUpdate, (synchronize_target_sprite,));
    }
}
