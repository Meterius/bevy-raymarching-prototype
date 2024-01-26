pub mod scene;

use crate::bindings::cuda::{
    ConeMarchTextureValue, Mesh, MeshBuffer, MeshTriangle, MeshVertex, RenderDataTextureValue,
    SdComposition, SdRuntimeSceneGeometry, BLOCK_SIZE, CONE_MARCH_LEVELS, MAX_SUN_LIGHT_COUNT,
};
use crate::cudarc_extension::CustomCudaFunction;
use crate::renderer::scene::RenderSceneSettings;
use bevy::core_pipeline::clear_color::ClearColorConfig;
use bevy::render::extract_resource::ExtractResource;
use bevy::window::PrimaryWindow;
use bevy::{prelude::*, render::render_resource::*};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::ffi::CString;
use std::sync::Arc;

const MAX_COMPOSITION_NODE_COUNT: usize = 160000;

const MAX_VERTEX_COUNT: usize = 1024;
const MAX_TRIANGLE_COUNT: usize = 1024;
const MAX_MESH_COUNT: usize = 1024;

const RENDER_TEXTURE_SIZE: (usize, usize) = (2560, 1440);

#[derive(Debug, Clone, Default, Resource, Reflect)]
#[reflect(Resource)]
pub struct RenderSettings {}

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
            levels: [16, 8, 2],
        }
    }
}

#[derive(Debug, Clone, Default, Component)]
pub struct RenderCameraTarget {}

#[derive(Debug, Clone, Default, Component)]
pub struct RenderRelayCameraTarget {}

#[derive(Clone, Debug, Default, Component)]
pub struct RenderTargetSprite {}

#[derive(Clone, Resource, ExtractResource, Deref)]
struct RenderTargetImage(Handle<Image>);

#[derive(Clone, Debug)]
struct RenderSceneGeometry {
    compositions: Box<[SdComposition; MAX_COMPOSITION_NODE_COUNT]>,
    triangles: Box<[MeshTriangle; MAX_TRIANGLE_COUNT]>,
    vertices: Box<[MeshVertex; MAX_VERTEX_COUNT]>,
    meshes: Box<[Mesh; MAX_MESH_COUNT]>,
}

impl Default for RenderSceneGeometry {
    fn default() -> Self {
        Self {
            compositions: Box::try_from(
                vec![SdComposition::default(); MAX_COMPOSITION_NODE_COUNT].into_boxed_slice(),
            )
            .unwrap(),
            triangles: Box::try_from(
                vec![MeshTriangle::default(); MAX_TRIANGLE_COUNT].into_boxed_slice(),
            )
            .unwrap(),
            vertices: Box::try_from(
                vec![MeshVertex::default(); MAX_VERTEX_COUNT].into_boxed_slice(),
            )
            .unwrap(),
            meshes: Box::try_from(vec![Mesh::default(); MAX_MESH_COUNT].into_boxed_slice())
                .unwrap(),
        }
    }
}

struct RenderCudaContext {
    #[allow(dead_code)]
    pub device: Arc<CudaDevice>,
    pub geometry_transferred_event: cudarc::driver::sys::CUevent,

    pub func_compute_compressed_depth: CustomCudaFunction,
    pub func_compute_render: CustomCudaFunction,
    pub func_compute_render_finalize: CustomCudaFunction,
}

struct RenderCudaStreams {
    render_stream: CudaStream,
}

struct RenderCudaBuffers {
    render_data_texture_buffer: CudaSlice<RenderDataTextureValue>,
    render_texture_buffer: CudaSlice<u8>,

    cm_texture_buffers: Vec<CudaSlice<ConeMarchTextureValue>>,
    compositions_buffer: CudaSlice<SdComposition>,

    mesh_vertex_buffer: CudaSlice<MeshVertex>,
    mesh_triangle_buffer: CudaSlice<MeshTriangle>,
    mesh_buffer: CudaSlice<Mesh>,
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
}

// Render Systems

const USE_PTX: bool = true;

fn setup_cuda(world: &mut World) {
    let compression = RenderConeCompression::default();

    info!("Ray Marcher Cone Marching Compression: {compression:?}");

    let start = std::time::Instant::now();

    let device = CudaDevice::new(0).unwrap();

    info!("CUDA Device Creation took {:.2?} seconds", start.elapsed());

    let start = std::time::Instant::now();

    let func_compute_compressed_depth: CustomCudaFunction;
    let func_compute_render: CustomCudaFunction;
    let func_compute_render_finalize: CustomCudaFunction;
    if USE_PTX {
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

        func_compute_compressed_depth = CustomCudaFunction::from_safe(
            device.get_func("main", "compute_compressed_depth").unwrap(),
            device.clone(),
        );
        func_compute_render = CustomCudaFunction::from_safe(
            device.get_func("main", "compute_render").unwrap(),
            device.clone(),
        );
        func_compute_render_finalize = CustomCudaFunction::from_safe(
            device.get_func("main", "compute_render_finalize").unwrap(),
            device.clone(),
        );
    } else {
        let module = unsafe {
            let src = include_bytes!("../../assets/cuda/compiled/main.cubin");
            cudarc::driver::result::module::load_data(src.as_ptr() as *const _).unwrap()
        };

        let regl_module = unsafe {
            let src = include_bytes!("../../assets/cuda/compiled/main_regl.cubin");
            cudarc::driver::result::module::load_data(src.as_ptr() as *const _).unwrap()
        };

        func_compute_compressed_depth = unsafe {
            let name = CString::new("compute_compressed_depth").unwrap();
            CustomCudaFunction::from_sys(
                cudarc::driver::result::module::get_function(module.clone(), name).unwrap(),
                device.clone(),
            )
        };

        func_compute_render = unsafe {
            let name = CString::new("compute_render").unwrap();
            CustomCudaFunction::from_sys(
                cudarc::driver::result::module::get_function(regl_module.clone(), name).unwrap(),
                device.clone(),
            )
        };

        func_compute_render_finalize = unsafe {
            let name = CString::new("compute_render_finalize").unwrap();
            CustomCudaFunction::from_sys(
                cudarc::driver::result::module::get_function(module.clone(), name).unwrap(),
                device.clone(),
            )
        };
    }

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

    let vertex_buffer = unsafe { device.alloc::<MeshVertex>(MAX_VERTEX_COUNT).unwrap() };
    let triangle_buffer = unsafe { device.alloc::<MeshTriangle>(MAX_TRIANGLE_COUNT).unwrap() };
    let mesh_buffer = unsafe { device.alloc::<Mesh>(MAX_MESH_COUNT).unwrap() };

    let compositions_buffer = unsafe {
        device
            .alloc::<SdComposition>(MAX_COMPOSITION_NODE_COUNT)
            .unwrap()
    };

    let mut cm_texture_buffers = vec![];

    for i in 0..CONE_MARCH_LEVELS as usize {
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

    let render_stream = device.fork_default_stream().unwrap();

    world.insert_resource(RenderSettings::default());
    world.insert_resource(compression);
    world.insert_non_send_resource(RenderCudaContext {
        device,
        geometry_transferred_event: cudarc::driver::result::event::create(
            cudarc::driver::sys::CUevent_flags_enum::CU_EVENT_DEFAULT,
        )
        .unwrap(),
        func_compute_render,
        func_compute_render_finalize,
        func_compute_compressed_depth,
    });
    world.insert_non_send_resource(RenderCudaStreams { render_stream });
    world.insert_non_send_resource(RenderCudaBuffers {
        render_texture_buffer,
        render_data_texture_buffer,
        cm_texture_buffers,
        compositions_buffer,
        mesh_vertex_buffer: vertex_buffer,
        mesh_triangle_buffer: triangle_buffer,
        mesh_buffer,
    });
    world.insert_non_send_resource(PreviousRenderParameter::default());
    world.insert_non_send_resource(RenderSceneGeometry::default());

    let mut geometry = world.non_send_resource_mut::<RenderSceneGeometry>();

    unsafe {
        cudarc::driver::sys::cuMemHostRegister_v2(
            geometry.compositions.as_mut_ptr() as *mut _,
            geometry.compositions.len() * std::mem::size_of::<SdComposition>(),
            0,
        )
        .result()
        .unwrap()
    };
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
    settings: Res<RenderSceneSettings>,
    compression: Res<RenderConeCompression>,
    geometry: NonSend<RenderSceneGeometry>,
    time: Res<Time>,
    camera: Query<(&Camera, &Projection, &GlobalTransform), With<RenderCameraTarget>>,
    render_context: NonSend<RenderCudaContext>,
    render_streams: NonSendMut<RenderCudaStreams>,
    render_buffers: NonSendMut<RenderCudaBuffers>,
    render_target_image: Res<RenderTargetImage>,
    mut images: ResMut<Assets<Image>>,

    mut tick: Local<u64>,

    mut previous_render_parameters: NonSendMut<PreviousRenderParameter>,
) {
    let range_id = nvtx::range_start!("Render System Wait For Previous Frame");

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

    // Geometry Transfer

    unsafe {
        cudarc::driver::result::memcpy_htod_async(
            render_buffers.compositions_buffer.device_ptr().clone(),
            &geometry.compositions.as_slice(),
            render_streams.render_stream.stream,
        )
        .unwrap()
    };

    unsafe {
        cudarc::driver::result::memcpy_htod_async(
            render_buffers.mesh_buffer.device_ptr().clone(),
            &geometry.meshes.as_slice(),
            render_streams.render_stream.stream,
        )
        .unwrap()
    };

    unsafe {
        cudarc::driver::result::memcpy_htod_async(
            render_buffers.mesh_vertex_buffer.device_ptr().clone(),
            &geometry.vertices.as_slice(),
            render_streams.render_stream.stream,
        )
        .unwrap()
    };

    unsafe {
        cudarc::driver::result::memcpy_htod_async(
            render_buffers.mesh_triangle_buffer.device_ptr().clone(),
            &geometry.triangles.as_slice(),
            render_streams.render_stream.stream,
        )
        .unwrap()
    };

    unsafe {
        cudarc::driver::result::event::record(
            render_context.geometry_transferred_event.clone(),
            render_streams.render_stream.stream.clone(),
        )
        .unwrap()
    };

    // Render Parameters

    let globals = crate::bindings::cuda::GlobalsBuffer {
        use_step_glow_on_foreground: settings.enable_step_glow_on_foreground,
        use_step_glow_on_background: settings.enable_step_glow_on_background,
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

    let mut sun_lights = Vec::with_capacity(MAX_SUN_LIGHT_COUNT as usize);

    for _ in 0..MAX_SUN_LIGHT_COUNT as usize {
        sun_lights.push(crate::bindings::cuda::SunLight {
            direction: Vec3::new(1.0, -1.0, 0.2).normalize().into(),
        });
    }

    let sd_runtime_scene = crate::bindings::cuda::SdRuntimeScene {
        geometry: SdRuntimeSceneGeometry {
            compositions: unsafe {
                std::mem::transmute(*(&render_buffers.compositions_buffer).device_ptr())
            },
            meshes: MeshBuffer {
                meshes: unsafe { std::mem::transmute(*(&render_buffers.mesh_buffer).device_ptr()) },
                triangles: unsafe {
                    std::mem::transmute(*(&render_buffers.mesh_triangle_buffer).device_ptr())
                },
                vertices: unsafe {
                    std::mem::transmute(*(&render_buffers.mesh_vertex_buffer).device_ptr())
                },
            },
        },
        lighting: crate::bindings::cuda::SdRuntimeSceneLighting {
            sun_light_count: 1,
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
                std::mem::transmute(*(&render_buffers.cm_texture_buffers[i as usize]).device_ptr())
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

    if compression.enabled {
        for i in 0..CONE_MARCH_LEVELS as u32 {
            let grid_size = ((cm_textures.textures[i as usize].size[0]
                * cm_textures.textures[i as usize].size[0]) as f32
                / BLOCK_SIZE as usize as f32)
                .ceil() as u32;

            unsafe {
                render_context
                    .func_compute_compressed_depth
                    .clone()
                    .launch_on_stream(
                        &render_streams.render_stream,
                        LaunchConfig {
                            block_dim: (BLOCK_SIZE as usize as u32, 1, 1),
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
        app.register_type::<RenderConeCompression>()
            .add_systems(Startup, (setup, setup_cuda))
            .add_systems(Last, render)
            .add_systems(PostUpdate, (synchronize_target_sprite,));
    }
}
