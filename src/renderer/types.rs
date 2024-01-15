use bevy::{
    prelude::*,
    render::{extract_resource::ExtractResource, render_resource::ShaderType},
};

#[derive(Clone, Debug, Default, Resource, Reflect, ExtractResource, ShaderType)]
#[reflect(Resource)]
pub struct RenderGlobals {
    pub seed: u32,
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
    pub pre_count: i32,
    pub pre: [RenderSDPreNode; 32],

    pub primitive_count: i32,
    pub primitive: [RenderSDPrimitiveNode; 32],
}

#[derive(Clone, Copy, Debug, ShaderType)]
pub struct RenderSDPreNode {
    pub translation: Vec3,
    pub scale: Vec3,
    pub min_scale: f32,
}

impl Default for RenderSDPreNode {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            scale: Vec3::ONE,
            min_scale: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct RenderSDPrimitiveNode {
    pub is_sphere: i32,
}
