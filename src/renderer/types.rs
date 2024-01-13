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

// Properties

#[derive(Clone, Debug, ShaderType)]
pub struct RenderSDTransform {
    pub translation: Vec3,
    pub scale: Vec3,
}

impl Default for RenderSDTransform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            scale: Vec3::ONE,
        }
    }
}

// Elements Primitives

#[derive(Clone, Debug, Default, ShaderType)]
pub struct RenderSDSphere {
    pub radius: f32,
}


#[derive(Clone, Debug, Default, ShaderType)]
pub struct RenderSDBox {
    pub size: Vec3,
}

// Element Compounds

#[derive(Clone, Debug, Default, ShaderType)]
pub struct RenderSDUnion {
    pub first: RenderSDObject,
    pub second: RenderSDObject,
}

// Element Object

pub enum RenderSDReferenceType {
    Object = 0,
    Sphere = 1,
    Box = 2,
    Union = 3,
}

#[derive(Clone, Debug, Default, ShaderType)]
pub struct RenderSDReference {
    pub variant: i32,
    pub index: i32,
}

#[derive(Clone, Debug, Default, ShaderType)]
pub struct RenderSDObject {
    pub transform: RenderSDTransform,
    pub content: RenderSDReference,
}
