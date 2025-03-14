use crate::bindings::cuda;
use crate::renderer::{RenderCudaContext};
use bevy::prelude::*;
use bevy::render::mesh::{MeshVertexAttributeId, PrimitiveTopology, VertexAttributeValues};
use itertools::Itertools;
use std::collections::{HashMap, VecDeque};

//

#[derive(Default, Debug, Clone, Resource, Reflect)]
#[reflect(Resource)]
pub struct RenderSceneSettings {
}

#[derive(Default)]
pub struct RenderScenePlugin {}

impl Plugin for RenderScenePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<RenderSceneSettings>()
            .insert_resource(RenderSceneSettings {
            });
    }
}

//

fn reflect_point(p: Vec3, mirr_p: Vec3, mirr_d: Vec3) -> Vec3 {
    let diff = (p - mirr_p).dot(mirr_d);
    return p - 2.0 * diff * mirr_d;
}
