use crate::bindings::cuda;
use crate::renderer::{render, RenderSceneGeometry};
use bevy::prelude::*;
use std::collections::VecDeque;

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
    Difference(Vec<Entity>),
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
            .add_systems(PostUpdate, (compile_scene_geometry,));
    }
}

//

#[derive(Default, Debug)]
pub enum SdPrimitiveNodeVariant {
    #[default]
    Sphere,
    Cube,
}

#[derive(Default, Debug)]
pub struct SdPrimitiveNode {
    variant: SdPrimitiveNodeVariant,
    translation: Vec3,
    scale: Vec3,
}

impl SdPrimitiveNode {
    fn bounding_box(&self) -> (Vec3, Vec3) {
        return match self.variant {
            SdPrimitiveNodeVariant::Sphere => {
                (self.translation - self.scale, self.translation + self.scale)
            }
            SdPrimitiveNodeVariant::Cube => (
                self.translation - self.scale * 0.5,
                self.translation + self.scale * 0.5,
            ),
        };
    }
}

#[derive(Default, Debug)]
pub enum SdCompositionNodeVariant {
    Primitive,
    #[default]
    Union,
    Difference,
}

#[derive(Debug)]
pub struct SdCompositionNode {
    variant: SdCompositionNodeVariant,
    primitive: Option<SdPrimitiveNode>,
    children: Vec<SdCompositionNode>,
    bounding_box: (Vec3, Vec3),
}

impl Default for SdCompositionNode {
    fn default() -> Self {
        Self {
            variant: SdCompositionNodeVariant::Union,
            primitive: None,
            children: Vec::new(),
            bounding_box: (Vec3::INFINITY, Vec3::NEG_INFINITY),
        }
    }
}

impl SdCompositionNode {
    fn depth(&self) -> usize {
        return 1 + self
            .children
            .iter()
            .fold(0, |x, child| x.max(child.depth()));
    }

    fn total(&self) -> usize {
        return self.children.iter().fold(1, |x, child| x + child.total());
    }
}

#[allow(dead_code)]
fn simple_bvh_reordering(node: &mut SdCompositionNode) {
    match node.variant {
        SdCompositionNodeVariant::Union => {}
        _ => {
            return;
        }
    };

    while node.children.len() > 2 {
        let mut best_pair = (0, 0, f32::INFINITY);

        for i in 0..node.children.len() {
            for j in 0..i {
                let combined = node.children[i]
                    .bounding_box
                    .0
                    .max(node.children[j].bounding_box.0)
                    - node.children[i]
                        .bounding_box
                        .1
                        .min(node.children[j].bounding_box.1);

                let vol = combined.x * combined.y * combined.z;

                if vol < best_pair.2 {
                    best_pair = (i, j, vol);
                }
            }
        }

        if best_pair.2.is_infinite() {
            return;
        }

        let child1 = node.children.remove(best_pair.0);
        let child2 = node.children.remove(best_pair.1);

        node.children.push(SdCompositionNode {
            variant: SdCompositionNodeVariant::Union,
            primitive: None,
            bounding_box: (
                child1.bounding_box.0.min(child2.bounding_box.0),
                child1.bounding_box.1.max(child2.bounding_box.1),
            ),
            children: vec![child1, child2],
        });
    }
}

fn simple_center_split_reordering(node: &mut SdCompositionNode) {
    if node.children.len() <= 2 {
        return;
    }

    let mut axis = 0;

    if node.bounding_box.1[axis] - node.bounding_box.0[axis]
        < node.bounding_box.1[1] - node.bounding_box.0[1]
    {
        axis = 1;
    }

    if node.bounding_box.1[axis] - node.bounding_box.0[axis]
        < node.bounding_box.1[2] - node.bounding_box.0[2]
    {
        axis = 2;
    }

    let mut child_left = SdCompositionNode::default();
    let mut child_right = SdCompositionNode::default();

    let center = (node.bounding_box.0[axis] + node.bounding_box.1[axis]) / 2.0;

    let mut prev_children = Vec::new();

    std::mem::swap(&mut prev_children, &mut node.children);

    for child in prev_children.into_iter() {
        if (child.bounding_box.1[axis] + child.bounding_box.0[axis]) / 2.0 > center {
            child_left.children.push(child);
        } else {
            child_right.children.push(child);
        }
    }

    // only insert children if both are non-empty, if one of them is empty bounding boxes were all aligned and cannot be split
    if child_left.children.len() == 0 {
        std::mem::swap(&mut node.children, &mut child_right.children);
    } else if child_right.children.len() == 0 {
        std::mem::swap(&mut node.children, &mut child_left.children);
    } else {
        node.children.push(child_left);
        node.children.push(child_right);
    }

    for child in node.children.iter_mut() {
        fill_scene_geometry_node_bounding_boxes(child);
        simple_center_split_reordering(child);
    }
}

#[allow(dead_code)]
fn simple_paired_reordering(node: &mut SdCompositionNode) {
    match node.variant {
        SdCompositionNodeVariant::Union => {}
        _ => {
            return;
        }
    };

    while node.children.len() > 2 {
        for i in 0..node.children.len() / 2 {
            let child1 = node.children.remove(node.children.len() - 1 - i);
            let child2 = node.children.remove(node.children.len() - 1 - i);

            node.children.push(SdCompositionNode {
                variant: SdCompositionNodeVariant::Union,
                primitive: None,
                bounding_box: (
                    child1.bounding_box.0.min(child2.bounding_box.0),
                    child1.bounding_box.1.max(child2.bounding_box.1),
                ),
                children: vec![child1, child2],
            });
        }
    }
}

fn fill_scene_geometry_node_bounding_boxes(node: &mut SdCompositionNode) {
    if let Some(primitive) = node.primitive.as_ref() {
        node.bounding_box = primitive.bounding_box();
    } else {
        let mut bounding_box = (Vec3::INFINITY, Vec3::NEG_INFINITY);

        node.children.iter_mut().for_each(|child| {
            bounding_box.0 = child.bounding_box.0.min(bounding_box.0);
            bounding_box.1 = child.bounding_box.1.max(bounding_box.1);
        });

        node.bounding_box = bounding_box;
    }
}

fn fill_scene_geometry_non_leaf_bounding_boxes(node: &mut SdCompositionNode) {
    node.children
        .iter_mut()
        .for_each(|child| fill_scene_geometry_non_leaf_bounding_boxes(child));

    fill_scene_geometry_node_bounding_boxes(node);
}

fn collect_scene_geometry(
    mut nodes: &mut Query<(
        &GlobalTransform,
        Option<&SdPrimitive>,
        Option<&SdComposition>,
        Option<&SdVisual>,
    )>,
) -> SdCompositionNode {
    let mut primitives = Vec::new();

    for (trn, primitive, _composition, _visual) in nodes.iter() {
        if let Some(primitive) = primitive {
            let primitive_node = SdPrimitiveNode {
                variant: match primitive {
                    SdPrimitive::Sphere(_) => SdPrimitiveNodeVariant::Sphere,
                    SdPrimitive::Box(_) => SdPrimitiveNodeVariant::Cube,
                },
                translation: trn.translation(),
                scale: match primitive {
                    SdPrimitive::Sphere(radius) => Vec3::ONE * *radius,
                    SdPrimitive::Box(size) => size.clone(),
                },
            };

            primitives.push(SdCompositionNode {
                variant: SdCompositionNodeVariant::Primitive,
                children: Vec::new(),
                bounding_box: primitive_node.bounding_box(),
                primitive: Some(primitive_node),
            })
        }
    }

    return SdCompositionNode {
        variant: SdCompositionNodeVariant::Union,
        primitive: None,
        children: primitives,
        bounding_box: (Vec3::NEG_INFINITY, Vec3::INFINITY),
    };
}

fn draw_bb_gizmos(gizmos: &mut Gizmos, node: &SdCompositionNode) {
    gizmos.cuboid(
        Transform {
            translation: (node.bounding_box.1 + node.bounding_box.0) / 2.0,
            scale: node.bounding_box.1 - node.bounding_box.0,
            ..default()
        },
        Color::BLUE,
    );

    node.children.iter().for_each(|child| {
        draw_bb_gizmos(gizmos, child);
    })
}

const PRINT_COMPILE_SCENE_GEOMETRY_INFO: bool = false;

fn compile_scene_geometry(
    settings: Res<RenderSceneSettings>,
    mut gizmos: Gizmos,
    mut geometry: ResMut<RenderSceneGeometry>,
    mut nodes: Query<(
        &GlobalTransform,
        Option<&SdPrimitive>,
        Option<&SdComposition>,
        Option<&SdVisual>,
    )>,
) {
    let start = std::time::Instant::now();

    let mut cube_index: usize = 0;
    let mut sphere_index: usize = 0;

    let mut composition_index: usize = 0;
    let mut composition_children_index: usize = 1;

    let mut root = collect_scene_geometry(&mut nodes);
    fill_scene_geometry_non_leaf_bounding_boxes(&mut root);
    simple_center_split_reordering(&mut root);
    let root_depth = if PRINT_COMPILE_SCENE_GEOMETRY_INFO {
        root.depth()
    } else {
        0
    };
    let root_count = if PRINT_COMPILE_SCENE_GEOMETRY_INFO {
        root.total()
    } else {
        0
    };

    if settings.enable_debug_gizmos {
        draw_bb_gizmos(&mut gizmos, &root);
    }

    let mut queue = VecDeque::new();
    queue.push_back((-1, root));

    while let Some((parent, item)) = queue.pop_front() {
        geometry.compositions[composition_index] = cuda::SdComposition {
            variant: match item.variant {
                SdCompositionNodeVariant::Primitive => cuda::SdCompositionVariant_Union,
                SdCompositionNodeVariant::Union => cuda::SdCompositionVariant_Union,
                SdCompositionNodeVariant::Difference => cuda::SdCompositionVariant_Difference,
            },
            primitive: match item.primitive.as_ref() {
                Some(primitive) => match primitive.variant {
                    SdPrimitiveNodeVariant::Cube => cube_index as i32,
                    SdPrimitiveNodeVariant::Sphere => sphere_index as i32,
                },
                None => 0,
            },
            primitive_variant: match item.primitive {
                Some(primitive) => match primitive.variant {
                    SdPrimitiveNodeVariant::Cube => {
                        geometry.cubes[cube_index] = cuda::SdCubePrimitive {
                            translation: primitive.translation.to_array(),
                            scale: primitive.scale.to_array(),
                        };
                        cube_index += 1;

                        cuda::SdPrimitiveVariant_Cube
                    }
                    SdPrimitiveNodeVariant::Sphere => {
                        geometry.spheres[sphere_index] = cuda::SdSpherePrimitive {
                            translation: primitive.translation.to_array(),
                            scale: primitive.scale.to_array(),
                        };
                        sphere_index += 1;

                        cuda::SdPrimitiveVariant_Sphere
                    }
                },
                None => cuda::SdPrimitiveVariant_None,
            },
            parent: parent as i32,
            child_leftmost: composition_children_index as i32,
            child_rightmost: (composition_children_index + item.children.len() - 1) as i32,
            bound_min: item.bounding_box.0.to_array(),
            bound_max: item.bounding_box.1.to_array(),
        };

        composition_children_index += item.children.len();
        queue.extend(
            item.children
                .into_iter()
                .map(|x| (composition_index as i32, x)),
        );
        composition_index += 1;
    }

    if PRINT_COMPILE_SCENE_GEOMETRY_INFO {
        info!(
            "Scene Compilation Took {:.2}ms",
            1000.0 * start.elapsed().as_secs_f32()
        );
        info!(
            "Scene Composition Depth {} And Count {}",
            root_depth, root_count
        );
    }
}
