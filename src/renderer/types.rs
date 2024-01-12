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
