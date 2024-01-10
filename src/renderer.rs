use crate::data::RayMarcherFrameData;
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

const SIZE: (u32, u32) = (2560, 1440);
const WORKGROUP_SIZE: u32 = 8;

#[derive(Resource)]
pub struct RayMarcherBuffers {
    frame_data_buffer: UniformBuffer<RayMarcherFrameData>,
}

#[derive(Debug, Clone, Default, Component)]
pub struct RayMarcherTargetSprite {}

#[derive(Clone, Resource, ExtractResource, Deref)]
struct RayMarcherTargetImage(Handle<Image>);

#[derive(Resource)]
struct RayMarcherBindGroup(BindGroup);

// App Systems

pub fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let mut image = Image::new_fill(
        Extent3d {
            width: SIZE.0,
            height: SIZE.1,
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
                custom_size: Some(Vec2::new(SIZE.0 as f32, SIZE.1 as f32)),
                ..default()
            },
            texture: image.clone(),
            ..default()
        },
        RayMarcherTargetSprite::default(),
    ));
    commands.spawn(Camera2dBundle::default());

    commands.insert_resource(RayMarcherTargetImage(image));
}

pub fn scale_target_to_screen(
    mut sprite: Query<&mut Transform, With<RayMarcherTargetSprite>>,
    window: Query<&Window, With<PrimaryWindow>>,
) {
    sprite.single_mut().scale = Vec2::new(
        window.single().width() / (SIZE.0 as f32),
        window.single().width() / (SIZE.0 as f32),
    )
    .extend(1.0);
}

// Render Systems

fn setup_buffers(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    buffers: Option<ResMut<RayMarcherBuffers>>,
    frame_data: Res<RayMarcherFrameData>,
    render_queue: Res<RenderQueue>,
) {
    if let Some(mut buffers) = buffers {
        buffers.frame_data_buffer.set(frame_data.clone());
        buffers
            .frame_data_buffer
            .write_buffer(&render_device, &render_queue);
    } else {
        let mut frame_data_buffer = UniformBuffer::from(frame_data.clone());
        frame_data_buffer.write_buffer(&render_device, &render_queue);
        commands.insert_resource(RayMarcherBuffers { frame_data_buffer });
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
                resource: buffers.frame_data_buffer.binding().unwrap(),
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
                    ],
                });
        let shader = world
            .resource::<AssetServer>()
            .load("shaders/ray_marcher.wgsl");
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
        /*render_context.command_encoder().copy_buffer_to_buffer(
            &buffers.staging_buffer,
            0,
            &buffers.buffer,
            0,
            std::mem::size_of::<RayMarcherFrameData>() as u64,
        );*/

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
                pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);
            }
            RayMarcherState::Update => {
                let update_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.update_pipeline)
                    .unwrap();
                pass.set_pipeline(update_pipeline);
                pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);
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
        app.add_plugins(ExtractResourcePlugin::<RayMarcherTargetImage>::default());
        app.add_plugins(ExtractResourcePlugin::<RayMarcherFrameData>::default());

        app.add_systems(Startup, setup);

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
