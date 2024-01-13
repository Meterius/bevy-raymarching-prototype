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

// Elements Primitives

#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct RenderSDSphere {
    pub radius: f32,
}


#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct RenderSDBox {
    pub size: Vec3,
}

// Element Compounds

#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct RenderSDTransform {
    pub translation: Vec3,
    pub scale: Vec3,

    pub content: RenderSDReference,
}

#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct RenderSDUnion {
    pub first: RenderSDReference,
    pub second: RenderSDReference,
}

// Element Object

pub enum RenderSDReferenceType {
    Sphere,
    Box,
    Transform,
    Union,
}

impl RenderSDReferenceType {
    pub fn as_i32(&self) -> i32 {
        match self {
            RenderSDReferenceType::Sphere => 1,
            RenderSDReferenceType::Box => 2,
            RenderSDReferenceType::Transform => 3,
            RenderSDReferenceType::Union => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct RenderSDReference {
    pub variant: i32,
    pub index: i32,
}
