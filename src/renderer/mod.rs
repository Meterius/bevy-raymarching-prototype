use crate::renderer::types::{RenderCamera, RenderGlobals, RenderScene};
use bevy::core_pipeline::clear_color::ClearColorConfig;
use bevy::render::renderer::RenderQueue;
use bevy::window::PrimaryWindow;
use bevy::{
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph},
        render_resource::*,
        renderer::{RenderContext, RenderDevice},
        Render, RenderApp, RenderSet,
    },
};
use std::borrow::Cow;

pub mod types;

pub const RENDER_TEXTURE_SIZE: (u32, u32) = (2560, 1440);
const WORKGROUP_SIZE: u32 = 8;

#[derive(Debug, Clone, Default, Component)]
pub struct RayMarcherCamera {}

#[derive(Resource)]
pub struct RayMarcherBuffers {
    globals_buffer: UniformBuffer<RenderGlobals>,
    camera_buffer: UniformBuffer<RenderCamera>,
    scene_buffer: UniformBuffer<RenderScene>,
}

#[derive(Debug, Clone, Default, Component)]
pub struct RayMarcherTargetSprite {}

#[derive(Clone, Resource, ExtractResource, Deref)]
struct RayMarcherTargetImage(Handle<Image>);

#[derive(Resource)]
struct RayMarcherBindGroup(BindGroup);

// App Systems

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let mut image = Image::new_fill(
        Extent3d {
            width: RENDER_TEXTURE_SIZE.0,
            height: RENDER_TEXTURE_SIZE.1,
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
        RayMarcherTargetSprite::default(),
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

    commands.insert_resource(RayMarcherTargetImage(image));
}

fn scale_target_to_screen(
    mut sprite: Query<&mut Transform, With<RayMarcherTargetSprite>>,
    window: Query<&Window, With<PrimaryWindow>>,
) {
    sprite.single_mut().scale = Vec2::new(
        window.single().width() / (RENDER_TEXTURE_SIZE.0 as f32),
        window.single().height() / (RENDER_TEXTURE_SIZE.1 as f32),
    )
    .extend(1.0);
}

// Synchronization

fn synchronize_globals(
    mut render_globals: ResMut<RenderGlobals>,
    time: Res<Time>,
) {
    render_globals.time = time.elapsed_seconds();
}

fn synchronize_camera(
    camera: Query<(&Camera, &GlobalTransform), With<RayMarcherCamera>>,
    mut render_camera: ResMut<RenderCamera>,
) {
    let (cam, cam_transform) = camera.single();
    render_camera.aspect_ratio = cam.logical_viewport_size().unwrap_or_default().x
        / cam.logical_viewport_size().unwrap_or_default().y;
    render_camera.position = cam_transform.translation();
    render_camera.forward = cam_transform.forward();
    render_camera.right = cam_transform.right();
    render_camera.up = cam_transform.up();
    render_camera.unit_plane_distance = 1.25;
}
// Render Systems

fn setup_buffers(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    buffers: Option<ResMut<RayMarcherBuffers>>,
    render_globals: Res<RenderGlobals>,
    render_camera: Res<RenderCamera>,
    render_scene: Res<RenderScene>,
    render_queue: Res<RenderQueue>,
) {
    if let Some(mut buffers) = buffers {
        buffers.globals_buffer.set(render_globals.clone());
        buffers
            .globals_buffer
            .write_buffer(&render_device, &render_queue);

        buffers.camera_buffer.set(render_camera.clone());
        buffers
            .camera_buffer
            .write_buffer(&render_device, &render_queue);

        buffers.scene_buffer.set(render_scene.clone());
        buffers
            .scene_buffer
            .write_buffer(&render_device, &render_queue);
    } else {
        let mut camera_buffer: UniformBuffer<RenderCamera> = UniformBuffer::from(render_camera.clone());
        camera_buffer.write_buffer(&render_device, &render_queue);

        let mut scene_buffer: UniformBuffer<RenderScene> = UniformBuffer::from(render_scene.clone());
        scene_buffer.write_buffer(&render_device, &render_queue);

        let mut globals_buffer: UniformBuffer<RenderGlobals> = UniformBuffer::from(render_globals.clone());
        globals_buffer.write_buffer(&render_device, &render_queue);

        commands.insert_resource(RayMarcherBuffers {
            camera_buffer,
            scene_buffer,
            globals_buffer,
        });
    }
}

fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<RayMarcherPipeline>,
    gpu_images: Res<RenderAssets<Image>>,
    image: Res<RayMarcherTargetImage>,
    buffers: Res<RayMarcherBuffers>,
    render_device: Res<RenderDevice>,
) {
    let view = gpu_images.get(&image.0).unwrap();

    let bind_group = render_device.create_bind_group(
        None,
        &pipeline.texture_bind_group_layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&view.texture_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: buffers.globals_buffer.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 2,
                resource: buffers.camera_buffer.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 3,
                resource: buffers.scene_buffer.binding().unwrap(),
            },
        ],
    );

    commands.insert_resource(RayMarcherBindGroup(bind_group));
}

// Render Pipeline

#[derive(Resource)]
pub struct RayMarcherPipeline {
    texture_bind_group_layout: BindGroupLayout,
    init_pipeline: CachedComputePipelineId,
    update_pipeline: CachedComputePipelineId,
}

impl FromWorld for RayMarcherPipeline {
    fn from_world(world: &mut World) -> Self {
        let texture_bind_group_layout =
            world
                .resource::<RenderDevice>()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::StorageTexture {
                                access: StorageTextureAccess::ReadWrite,
                                format: TextureFormat::Rgba8Unorm,
                                view_dimension: TextureViewDimension::D2,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 2,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 3,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let _ = world
            .resource::<AssetServer>()
            .load::<Shader>("shaders/utils.wgsl");

        let shader = world
            .resource::<AssetServer>()
            .load("shaders/renderer.wgsl");

        let pipeline_cache = world.resource::<PipelineCache>();
        let init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![texture_bind_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Cow::from("init"),
        });
        let update_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![texture_bind_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: vec![],
            entry_point: Cow::from("update"),
        });

        RayMarcherPipeline {
            texture_bind_group_layout,
            init_pipeline,
            update_pipeline,
        }
    }
}

enum RayMarcherState {
    Loading,
    Init,
    Update,
}

struct RayMarcherNode {
    state: RayMarcherState,
}

impl Default for RayMarcherNode {
    fn default() -> Self {
        Self {
            state: RayMarcherState::Loading,
        }
    }
}

impl render_graph::Node for RayMarcherNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<RayMarcherPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            RayMarcherState::Loading => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.init_pipeline)
                {
                    self.state = RayMarcherState::Init;
                }
            }
            RayMarcherState::Init => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.update_pipeline)
                {
                    self.state = RayMarcherState::Update;
                }
            }
            RayMarcherState::Update => {}
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        let bind_group = &world.resource::<RayMarcherBindGroup>().0;
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<RayMarcherPipeline>();

        pass.set_bind_group(0, bind_group, &[]);

        // select the pipeline based on the current state
        match self.state {
            RayMarcherState::Loading => {}
            RayMarcherState::Init => {
                let init_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.init_pipeline)
                    .unwrap();
                pass.set_pipeline(init_pipeline);
                pass.dispatch_workgroups(
                    RENDER_TEXTURE_SIZE.0 / WORKGROUP_SIZE,
                    RENDER_TEXTURE_SIZE.1 / WORKGROUP_SIZE,
                    1,
                );
            }
            RayMarcherState::Update => {
                let update_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.update_pipeline)
                    .unwrap();
                pass.set_pipeline(update_pipeline);
                pass.dispatch_workgroups(
                    RENDER_TEXTURE_SIZE.0 / WORKGROUP_SIZE,
                    RENDER_TEXTURE_SIZE.1 / WORKGROUP_SIZE,
                    1,
                );
            }
        }

        Ok(())
    }
}

// Plugin

#[derive(Default)]
pub struct RayMarcherRenderPlugin {}

impl Plugin for RayMarcherRenderPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(RenderCamera::default());
        app.insert_resource(RenderScene {
            sun_direction: Vec3::new(0.5, 1.0, 3.0).normalize(),
            ..default()
        });
        app.insert_resource(RenderGlobals {
            render_texture_size: Vec2::new(RENDER_TEXTURE_SIZE.0 as _, RENDER_TEXTURE_SIZE.1 as _),
            ..default()
        });

        app.add_plugins((
            ExtractResourcePlugin::<RayMarcherTargetImage>::default(),
            ExtractResourcePlugin::<RenderCamera>::default(),
            ExtractResourcePlugin::<RenderScene>::default(),
            ExtractResourcePlugin::<RenderGlobals>::default(),
        ));

        app.add_systems(Startup, setup);
        app.add_systems(Update, scale_target_to_screen);
        app.add_systems(PostUpdate, (
            synchronize_camera, synchronize_globals,
        ));

        let render_app = app.sub_app_mut(RenderApp);

        render_app.add_systems(
            Render,
            (
                prepare_bind_group.in_set(RenderSet::PrepareBindGroups),
                setup_buffers.in_set(RenderSet::PrepareResources),
            ),
        );

        let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
        render_graph.add_node("ray_marcher", RayMarcherNode::default());
        render_graph.add_node_edge("ray_marcher", bevy::render::main_graph::node::CAMERA_DRIVER);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<RayMarcherPipeline>();
    }
}
