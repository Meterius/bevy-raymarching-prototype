use bevy::{prelude::*, render::{extract_resource::ExtractResource, render_resource::ShaderType}};

#[derive(Clone, Debug, Default, Resource, Reflect, ExtractResource, ShaderType)]
#[reflect(Resource)]
pub struct RenderGlobals {
    pub time: f32,

    pub render_texture_size: Vec2,
}

#[derive(Clone, Debug, Default, Resource, Reflect, ExtractResource, ShaderType)]
#[reflect(Resource)]
pub struct RenderScene {
    pub sun_direction: Vec3,
}

#[derive(Clone, Debug, Default, Resource, Reflect, ExtractResource, ShaderType)]
#[reflect(Resource)]
pub struct RenderCamera {
    pub unit_plane_distance: f32,
    pub aspect_ratio: f32,
    pub position: Vec3,
    pub forward: Vec3,
    pub up: Vec3,
    pub right: Vec3,
}

/* Signed-Distance Types */

// Command-Encoding

#[derive(Clone, Debug, Default, ShaderType)]
pub struct RenderSDScene {
    pub compound_count: i32,
    pub compounds: [RenderSDCompoundNode; 32],

    pub primitive_count: i32,
    pub primitives: [RenderSDPrimitiveNode; 32],
}

#[derive(Clone, Copy, Debug, ShaderType)]
pub struct RenderSDCompoundNode {
    // pre-transforming
    pub pre_translation: Vec3,
    pub pre_scale: Vec3,

    // post-transforming
    pub post_scale: f32,

    // relations
    pub parent: i32,
    pub children: [i32; 2],
}

impl Default for RenderSDCompoundNode {
    fn default() -> Self {
        Self {
            pre_translation: Vec3::ZERO,
            pre_scale: Vec3::ONE,
            post_scale: 1.0,
            parent: -1,
            children: [-1, -1],
        }
    }
}

#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct RenderSDPrimitiveNode {
    pub use_sphere: i32,
    pub sphere: f32,

    pub use_block: i32,
    pub block: Vec3,

    // output
    pub container: i32,
}
