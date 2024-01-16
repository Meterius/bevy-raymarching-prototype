use bevy::core_pipeline::clear_color::ClearColorConfig;
use bevy::render::extract_resource::ExtractResource;
use bevy::window::PrimaryWindow;
use bevy::{prelude::*, render::render_resource::*};
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

const RENDER_TEXTURE_SIZE: (usize, usize) = (1920, 1080);

#[derive(Debug, Clone, Default, Component)]
pub struct RenderCameraTarget {}

#[derive(Clone, Debug, Default, Component)]
pub struct RenderTargetSprite {}

#[derive(Clone, Resource, ExtractResource, Deref)]
struct RenderTargetImage(Handle<Image>);

struct RenderCuda {
    device: Arc<CudaDevice>,
    render_texture_buffer: CudaSlice<u8>,
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

const BLOCK_DIM: (u32, u32, u32) = (64, 1, 1);

fn setup_cuda(world: &mut World) {
    let start = std::time::Instant::now();

    let device = CudaDevice::new(0).unwrap();

    info!("CUDA Device Creation took {:.2?} seconds", start.elapsed());

    let start = std::time::Instant::now();

    let ptx = Ptx::from_src(include_str!("../../assets/cuda/compiled/renderer.ptx"));
    device.load_ptx(ptx, "renderer", &["render"]).unwrap();

    info!("CUDA PTX Loading took {:.2?} seconds", start.elapsed());

    let render_texture_buffer = unsafe {
        device
            .alloc::<u8>(4 * RENDER_TEXTURE_SIZE.0 * RENDER_TEXTURE_SIZE.1)
            .unwrap()
    };

    world.insert_non_send_resource(RenderCuda {
        device,
        render_texture_buffer,
    });
}

fn render(
    time: Res<Time>,
    camera: Query<(&Camera, &Projection, &GlobalTransform), With<RenderCameraTarget>>,
    render_cuda: NonSend<RenderCuda>,
    render_target_image: Res<RenderTargetImage>,
    mut images: ResMut<Assets<Image>>,
    mut tick: Local<u64>,
) {
    *tick += 1;

    let image = images.get_mut(&render_target_image.0).unwrap();
    let (cam, cam_projection, cam_transform) = camera.single();

    render_cuda
        .device
        .dtoh_sync_copy_into(
            &render_cuda.render_texture_buffer,
            image.data.as_mut_slice(),
        )
        .unwrap();

    unsafe {
        render_cuda
            .device
            .get_func("renderer", "render")
            .unwrap()
            .launch(
                LaunchConfig {
                    block_dim: BLOCK_DIM,
                    grid_dim: (
                        (RENDER_TEXTURE_SIZE.1 as u32 * RENDER_TEXTURE_SIZE.0 as u32) / BLOCK_DIM.0,
                        1,
                        1,
                    ),
                    shared_mem_bytes: 0,
                },
                (
                    &render_cuda.render_texture_buffer,
                    crate::bindings::cuda::GlobalsBuffer {
                        time: time.elapsed_seconds(),
                        tick: tick.clone(),
                        render_texture_size: [
                            RENDER_TEXTURE_SIZE.0 as u32,
                            RENDER_TEXTURE_SIZE.1 as u32,
                        ],
                        render_screen_size: [
                            cam.logical_viewport_size().map(|s| s.x).unwrap_or(1.0) as _,
                            cam.logical_viewport_size().map(|s| s.y).unwrap_or(1.0) as _,
                        ],
                    },
                    crate::bindings::cuda::CameraBuffer {
                        position: cam_transform.translation().as_ref().clone(),
                        forward: cam_transform.forward().as_ref().clone(),
                        up: cam_transform.up().as_ref().clone(),
                        right: cam_transform.right().as_ref().clone(),
                        fov: match cam_projection {
                            Projection::Perspective(perspective) => perspective.fov,
                            Projection::Orthographic(_) => 1.0,
                        },
                    },
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
