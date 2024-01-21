use crate::bindings::cuda;
use crate::renderer::{render, RenderSceneGeometry};
use bevy::prelude::*;

#[derive(Debug, Clone, Component, Reflect)]
#[reflect(Component)]
pub enum SdPrimitive {
    Sphere(f32),
    Box(Vec3),
}

impl Default for SdPrimitive {
    fn default() -> Self {
        Self::Sphere(1.0)
    }
}

#[derive(Debug, Clone, Component, Reflect)]
#[reflect(Component)]
pub enum SdComposition {
    Union(Vec<Entity>),
    UnionRelocation(Vec<Entity>),
    Difference(Vec<Entity>),
    DifferenceRelocation(Vec<Entity>),
    Mirror(Vec3),
}

impl Default for SdComposition {
    fn default() -> Self {
        Self::Union(Vec::new())
    }
}

#[derive(Default, Debug, Clone, Component, Reflect)]
#[reflect(Component)]
pub struct SdVisual {
    pub enabled: bool,
}

//

#[derive(Default, Debug, Clone, Resource, Reflect)]
#[reflect(Resource)]
pub struct RenderSceneSettings {
    pub enable_debug_gizmos: bool,
}

#[derive(Default)]
pub struct RenderScenePlugin {}

impl Plugin for RenderScenePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<SdPrimitive>()
            .register_type::<SdComposition>()
            .register_type::<SdVisual>()
            .register_type::<RenderSceneSettings>()
            .insert_resource(RenderSceneSettings {
                enable_debug_gizmos: true,
            })
            .add_systems(PostUpdate, (compile_scene_geometry, render_scene_gizmos));
    }
}

fn compile_scene_geometry(
    mut geometry: ResMut<RenderSceneGeometry>,
    nodes: Query<(
        &GlobalTransform,
        Option<&SdPrimitive>,
        Option<&SdComposition>,
        Option<&SdVisual>,
    )>,
) {
    let mut cube_index = 0;
    let mut sphere_index = 0;
    let mut composition_index = 1;

    for (trn, primitive, composition, visual) in nodes.iter() {
        if let Some(primitive) = primitive {
            match primitive {
                SdPrimitive::Sphere(radius) => {
                    geometry.spheres[sphere_index] = cuda::SdSpherePrimitive {
                        translation: trn.translation().to_array(),
                        scale: [*radius; 3],
                    };

                    geometry.compositions[composition_index] = cuda::SdComposition {
                        variant: cuda::SdCompositionVariant_Union,
                        primitive: sphere_index as i32,
                        primitive_variant: cuda::SdPrimitiveVariant_Sphere,
                        parent: 0,
                        child_leftmost: -1,
                        child_rightmost: -1,
                    };

                    composition_index += 1;
                    sphere_index += 1;
                }
                _ => {}
            };
        }
    }

    geometry.compositions[0] = cuda::SdComposition {
        variant: cuda::SdCompositionVariant_Union,
        primitive: 0,
        primitive_variant: cuda::SdPrimitiveVariant_None,
        parent: 0,
        child_leftmost: 1,
        child_rightmost: composition_index as i32 - 1,
    };
}

fn render_scene_gizmos(
    mut gizmos: Gizmos,
    settings: Res<RenderSceneSettings>,
    nodes: Query<(
        &GlobalTransform,
        Option<&SdPrimitive>,
        Option<&SdComposition>,
        Option<&SdVisual>,
    )>,
) {
}
