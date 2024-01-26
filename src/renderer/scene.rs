use crate::bindings::cuda;
use crate::renderer::{RenderCudaContext, RenderSceneGeometry};
use bevy::prelude::*;
use bevy::render::mesh::{MeshVertexAttributeId, PrimitiveTopology, VertexAttributeValues};
use itertools::Itertools;
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Component, Reflect)]
#[reflect(Component)]
pub enum SdPrimitive {
    Sphere(f32),
    Box(Vec3),
    Mandelbulb(f32),
    Mesh(Handle<Mesh>),
    Triangle([Vec3; 3]),
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
    Intersect(Vec<Entity>),
    Difference(Vec<Entity>), // first element minus the remaining
    Mirror(Vec<Entity>),
}

impl Default for SdComposition {
    fn default() -> Self {
        Self::Union(Vec::new())
    }
}

#[derive(Debug, Clone, Component, Reflect)]
#[reflect(Component)]
pub struct SdVisual {
    pub enabled: bool,
}

impl Default for SdVisual {
    fn default() -> Self {
        Self { enabled: true }
    }
}

//

#[derive(Default, Debug, Clone, Resource)]
pub struct RenderSceneStoredGeometry {
    root: Option<SdCompositionNode>,
}

#[derive(Default, Debug, Clone, Resource, Reflect)]
#[reflect(Resource)]
pub struct RenderSceneSettings {
    pub enable_debug_gizmos: bool,
    pub enable_step_glow_on_foreground: bool,
    pub enable_step_glow_on_background: bool,
}

#[derive(Default)]
pub struct RenderScenePlugin {}

impl Plugin for RenderScenePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<SdPrimitive>()
            .register_type::<SdComposition>()
            .register_type::<SdVisual>()
            .register_type::<RenderSceneSettings>()
            .insert_resource(RenderSceneStoredGeometry::default())
            .insert_resource(RenderSceneSettings {
                enable_debug_gizmos: false,
                enable_step_glow_on_background: false,
                enable_step_glow_on_foreground: false,
            })
            .add_systems(PostUpdate, (compile_scene_geometry,));
    }
}

//

#[derive(Default, Clone, Debug)]
pub enum SdPrimitiveNodeVariant {
    #[default]
    Empty,
    Sphere,
    Cube,
    Mandelbulb,
    Mesh,
    Triangle,
}

#[derive(Default, Clone, Debug)]
pub struct SdPrimitiveNode {
    variant: SdPrimitiveNodeVariant,
    translation: Vec3,
    scale: Vec3,
    rotation: Quat,
    mesh_id: u32,
    triangle_v0: Vec3,
    triangle_v1: Vec3,
    triangle_v2: Vec3,
}

impl SdPrimitiveNode {
    fn bounding_box(&self) -> (Vec3, Vec3) {
        return match self.variant {
            SdPrimitiveNodeVariant::Triangle => (
                self.triangle_v0.min(self.triangle_v1.min(self.triangle_v2)),
                self.triangle_v0.max(self.triangle_v1.max(self.triangle_v2)),
            ),
            _ => {
                let mut bb_min = Vec3::INFINITY;
                let mut bb_max = Vec3::NEG_INFINITY;

                for xi in 0..=1 {
                    for yi in 0..=1 {
                        for zi in 0..=1 {
                            let p = self.rotation
                                * (Vec3::new(-0.5 + xi as f32, -0.5 + yi as f32, -0.5 + zi as f32)
                                    * self.scale);
                            bb_min = bb_min.min(p);
                            bb_max = bb_max.max(p);
                        }
                    }
                }

                (bb_min + self.translation, bb_max + self.translation)
            }
        };
    }
}

#[derive(Default, Clone, Debug)]
pub enum SdCompositionNodeVariant {
    Primitive(SdPrimitiveNode),
    #[default]
    Union,
    Difference,
    Intersect,
    Mirror(Vec3, Vec3), // translation, direction
}

#[derive(Debug, Clone)]
pub struct SdCompositionNode {
    variant: SdCompositionNodeVariant,
    children: Vec<SdCompositionNode>,
    bounding_box: (Vec3, Vec3),
}

impl Default for SdCompositionNode {
    fn default() -> Self {
        Self {
            variant: SdCompositionNodeVariant::Union,
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

    fn flatten(&mut self) {
        self.children.iter_mut().for_each(|child| child.flatten());

        match self.variant {
            SdCompositionNodeVariant::Union => {
                let mut prev_children = Vec::new();
                std::mem::swap(&mut self.children, &mut prev_children);

                prev_children.into_iter().for_each(|mut child| {
                    match child.variant {
                        SdCompositionNodeVariant::Union => {
                            self.children.append(&mut child.children);
                        }
                        _ => {
                            self.children.push(child);
                        }
                    };
                });
            }
            _ => {}
        };
    }

    fn normalize(mut self) -> Self {
        self.children = self
            .children
            .into_iter()
            .map(|item| item.normalize())
            .collect();

        match &self.variant {
            SdCompositionNodeVariant::Mirror(_, _)
            | SdCompositionNodeVariant::Primitive(_)
            | SdCompositionNodeVariant::Difference => self,
            SdCompositionNodeVariant::Union | SdCompositionNodeVariant::Intersect => {
                if self.children.len() == 1 {
                    self.children.remove(0)
                } else {
                    self
                }
            }
        }
    }

    fn pad_with_empties(&mut self) {
        for child in self.children.iter_mut() {
            child.pad_with_empties();
        }

        if self.children.len() == 1 {
            self.children.push(SdCompositionNode {
                variant: SdCompositionNodeVariant::Primitive(SdPrimitiveNode {
                    translation: Vec3::ZERO,
                    scale: Vec3::ONE,
                    rotation: Quat::IDENTITY,
                    variant: SdPrimitiveNodeVariant::Empty,
                    ..default()
                }),
                children: Vec::new(),
                bounding_box: (Vec3::NEG_INFINITY, Vec3::INFINITY),
            });
        }
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
            bounding_box: (
                child1.bounding_box.0.min(child2.bounding_box.0),
                child1.bounding_box.1.max(child2.bounding_box.1),
            ),
            children: vec![child1, child2],
        });
    }
}

fn simple_center_split_reordering(node: &mut SdCompositionNode) {
    if node.children.len() > 2 {
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
            child_left.children = child_right
                .children
                .split_off(child_right.children.len() / 2);
        } else if child_right.children.len() == 0 {
            child_right.children = child_left.children.split_off(child_left.children.len() / 2);
        }

        // unwrap singular child nodes
        let child_left = if child_left.children.len() == 1 {
            child_left.children.remove(0)
        } else {
            child_left
        };
        let child_right = if child_right.children.len() == 1 {
            child_right.children.remove(0)
        } else {
            child_right
        };

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
                bounding_box: (
                    child1.bounding_box.0.min(child2.bounding_box.0),
                    child1.bounding_box.1.max(child2.bounding_box.1),
                ),
                children: vec![child1, child2],
            });
        }
    }
}

fn reflect_point(p: Vec3, mirr_p: Vec3, mirr_d: Vec3) -> Vec3 {
    let diff = (p - mirr_p).dot(mirr_d);
    return p - 2.0 * diff * mirr_d;
}

fn fill_scene_geometry_node_bounding_boxes(node: &mut SdCompositionNode) {
    let mut bounding_box;
    match &node.variant {
        SdCompositionNodeVariant::Primitive(primitive) => {
            bounding_box = primitive.bounding_box();
        }
        SdCompositionNodeVariant::Union => {
            bounding_box = (Vec3::INFINITY, Vec3::NEG_INFINITY);

            node.children.iter_mut().for_each(|child| {
                bounding_box.0 = child.bounding_box.0.min(bounding_box.0);
                bounding_box.1 = child.bounding_box.1.max(bounding_box.1);
            });
        }
        SdCompositionNodeVariant::Intersect => {
            bounding_box = (Vec3::NEG_INFINITY, Vec3::INFINITY);

            node.children.iter_mut().for_each(|child| {
                bounding_box.0 = child.bounding_box.0.max(bounding_box.0);
                bounding_box.1 = child.bounding_box.1.min(bounding_box.1);
            });
        }
        SdCompositionNodeVariant::Difference => {
            bounding_box = node.children[0].bounding_box.clone();
        }
        SdCompositionNodeVariant::Mirror(pos, dir) => {
            bounding_box = (Vec3::INFINITY, Vec3::NEG_INFINITY);

            node.children.iter_mut().for_each(|child| {
                bounding_box.0 = child.bounding_box.0.min(bounding_box.0);
                bounding_box.1 = child.bounding_box.1.max(bounding_box.1);

                let mirrored_min = reflect_point(child.bounding_box.0, pos.clone(), dir.clone());
                let mirrored_max = reflect_point(child.bounding_box.1, pos.clone(), dir.clone());

                bounding_box.0 = mirrored_min.min(mirrored_max).min(bounding_box.0);
                bounding_box.1 = mirrored_min.max(mirrored_max).max(bounding_box.1);
            });
        }
    };

    node.bounding_box = bounding_box;
}

fn fill_scene_geometry_non_leaf_bounding_boxes(node: &mut SdCompositionNode) {
    node.children
        .iter_mut()
        .for_each(|child| fill_scene_geometry_non_leaf_bounding_boxes(child));

    fill_scene_geometry_node_bounding_boxes(node);
}

fn collect_scene_geometry(
    nodes: &mut Query<
        (
            Entity,
            &GlobalTransform,
            Option<&SdPrimitive>,
            Option<&SdComposition>,
            Option<&SdVisual>,
        ),
        Or<(With<SdPrimitive>, With<SdComposition>)>,
    >,
    mesh_collection: &mut HashMap<Handle<Mesh>, u32>,
    meshes: &mut Res<Assets<Mesh>>,
) -> SdCompositionNode {
    let mut roots = Vec::new();

    fn convert(
        id: Entity,
        nodes: &Query<
            (
                Entity,
                &GlobalTransform,
                Option<&SdPrimitive>,
                Option<&SdComposition>,
                Option<&SdVisual>,
            ),
            Or<(With<SdPrimitive>, With<SdComposition>)>,
        >,
        mut mesh_collection: &mut HashMap<Handle<Mesh>, u32>,
        mut meshes: &mut Res<Assets<Mesh>>,
    ) -> Option<SdCompositionNode> {
        let (_id, trn, primitive, composition, _visual) = nodes.get(id).ok()?;

        if let Some(composition) = composition {
            match composition {
                SdComposition::Intersect(children) => {
                    return Some(SdCompositionNode {
                        variant: SdCompositionNodeVariant::Intersect,
                        children: children
                            .iter()
                            .filter_map(|child| {
                                convert(child.clone(), nodes, mesh_collection, meshes)
                            })
                            .collect(),
                        ..default()
                    });
                }
                SdComposition::Union(children) => {
                    return Some(SdCompositionNode {
                        variant: SdCompositionNodeVariant::Union,
                        children: children
                            .iter()
                            .filter_map(|child| {
                                convert(child.clone(), nodes, mesh_collection, meshes)
                            })
                            .collect(),
                        ..default()
                    });
                }
                SdComposition::Difference(children) => {
                    return if children.len() == 0 {
                        None
                    } else if children.len() == 1 {
                        convert(children[0].clone(), nodes, mesh_collection, meshes)
                    } else {
                        convert(children[0].clone(), nodes, mesh_collection, meshes).map(
                            move |node| SdCompositionNode {
                                variant: SdCompositionNodeVariant::Difference,
                                children: vec![
                                    node,
                                    SdCompositionNode {
                                        variant: SdCompositionNodeVariant::Union,
                                        children: children
                                            .iter()
                                            .skip(1)
                                            .filter_map(|child| {
                                                convert(
                                                    child.clone(),
                                                    nodes,
                                                    mesh_collection,
                                                    meshes,
                                                )
                                            })
                                            .collect(),
                                        ..default()
                                    },
                                ],
                                ..default()
                            },
                        )
                    }
                }
                SdComposition::Mirror(children) => {
                    return Some(SdCompositionNode {
                        variant: SdCompositionNodeVariant::Mirror(trn.translation(), trn.forward()),
                        children: children
                            .iter()
                            .filter_map(|child| {
                                convert(child.clone(), nodes, mesh_collection, meshes)
                            })
                            .collect(),
                        ..default()
                    });
                }
            }
        } else if let Some(primitive) = primitive {
            let trn_c = trn.compute_transform();

            let primitive_node = match primitive {
                SdPrimitive::Mesh(mesh_handle) => {
                    let next_mesh_id = mesh_collection.len() as u32;
                    let mesh_id = match mesh_collection.entry(mesh_handle.clone()) {
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            entry.insert(next_mesh_id);
                            next_mesh_id
                        }
                        std::collections::hash_map::Entry::Occupied(entry) => entry.get().clone(),
                    };

                    let mesh = meshes.get(mesh_handle);

                    if let Some(mesh) = mesh {
                        let bb = mesh.compute_aabb();

                        if let Some(bb) = bb {
                            Some(SdPrimitiveNode {
                                variant: SdPrimitiveNodeVariant::Mesh,
                                translation: trn_c.translation + Vec3::from(bb.center),
                                scale: trn_c.scale * bb.half_extents.max_element() * 2.0,
                                rotation: trn_c.rotation,
                                mesh_id: mesh_id,
                                ..default()
                            })
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                SdPrimitive::Triangle(vertices) => Some(SdPrimitiveNode {
                    variant: SdPrimitiveNodeVariant::Triangle,
                    triangle_v0: trn.transform_point(vertices[0]),
                    triangle_v1: trn.transform_point(vertices[1]),
                    triangle_v2: trn.transform_point(vertices[2]),
                    ..default()
                }),
                _ => Some(SdPrimitiveNode {
                    variant: match primitive {
                        SdPrimitive::Sphere(_) => SdPrimitiveNodeVariant::Sphere,
                        SdPrimitive::Box(_) => SdPrimitiveNodeVariant::Cube,
                        SdPrimitive::Mandelbulb(_) => SdPrimitiveNodeVariant::Mandelbulb,
                        SdPrimitive::Mesh(_) => SdPrimitiveNodeVariant::Mesh,
                        SdPrimitive::Triangle(_) => SdPrimitiveNodeVariant::Triangle,
                    },
                    translation: trn_c.translation,
                    scale: match primitive {
                        SdPrimitive::Sphere(radius) => Vec3::ONE * *radius,
                        SdPrimitive::Box(size) => size.clone(),
                        SdPrimitive::Mandelbulb(radius) => Vec3::ONE * *radius,
                        SdPrimitive::Mesh(_) => Vec3::ONE,
                        SdPrimitive::Triangle(_) => Vec3::ONE,
                    } * trn_c.scale,
                    rotation: trn_c.rotation,
                    ..default()
                }),
            };

            if let Some(primitive_node) = primitive_node {
                return Some(SdCompositionNode {
                    bounding_box: primitive_node.bounding_box(),
                    variant: SdCompositionNodeVariant::Primitive(primitive_node),
                    children: Vec::new(),
                });
            } else {
                return None;
            }
        }

        return None;
    }

    for (id, _trn, _primitive, _composition, visual) in nodes.iter() {
        if let Some(SdVisual { enabled: true }) = visual {
            if let Some(item) = convert(id, nodes, mesh_collection, meshes) {
                roots.push(item);
            }
        }
    }

    let mut node = SdCompositionNode {
        variant: SdCompositionNodeVariant::Union,
        children: roots,
        bounding_box: (Vec3::NEG_INFINITY, Vec3::INFINITY),
    };

    node.flatten();

    return node.normalize();
}

fn compile_scene_mesh(
    geometry: &mut NonSendMut<RenderSceneGeometry>,
    mesh: &Mesh,
    mesh_id: u32,
    vertex_offset: &mut usize,
    triangle_offset: &mut usize,
) {
    if let Some(bb) = mesh.compute_aabb() {
        let center = Vec3::from(bb.center);
        let scale = bb.half_extents.max_element() * 2.0;

        if mesh.primitive_topology() != PrimitiveTopology::TriangleList {
            panic!("Ray Marcher can only handle triangle list topologies");
        }

        let positions = match mesh.attribute(MeshVertexAttributeId::from(Mesh::ATTRIBUTE_POSITION))
        {
            Some(VertexAttributeValues::Float32x3(positions)) => positions,
            _ => panic!("Expected Float32x3 position attributes to be present in mesh"),
        };

        let triangle_start_id = *triangle_offset as u32;

        mesh.indices().iter().for_each(|island| {
            println!("Island {island:?}");

            island
                .iter()
                .chunks(3)
                .into_iter()
                .enumerate()
                .for_each(|(face_idx, face)| {
                    println!("Triangle {triangle_offset}");

                    let mut face_vertices = Vec::new();

                    face.for_each(|idx| {
                        println!("Vertex {vertex_offset}; Position {:?}", positions[idx]);

                        face_vertices.push(*vertex_offset as u32);
                        geometry.vertices[*vertex_offset] = cuda::MeshVertex {
                            pos: positions[idx].clone(),
                            ..default()
                        };
                        *vertex_offset += 1;
                    });

                    geometry.triangles[*triangle_offset] = cuda::MeshTriangle {
                        vertex_ids: face_vertices
                            .try_into()
                            .expect("Expected 3 vertices per face"),
                        ..default()
                    };
                    *triangle_offset += 1;
                });
        });

        let triangle_end_id = *triangle_offset as u32 - 1;

        geometry.meshes[mesh_id as usize] = cuda::Mesh {
            triangle_start_id,
            triangle_end_id,
            ..default()
        };
    } else {
        geometry.meshes[mesh_id as usize] = cuda::Mesh {
            triangle_start_id: 1,
            triangle_end_id: 0,
        };
    }
}

enum SdCompositionAppendix {
    None,
    Mirror(cuda::SdCompositionMirrorAppendix),
    Primitive(cuda::SdCompositionPrimitiveAppendix),
    PrimitiveTriangle(cuda::SdCompositionPrimitiveTriangleAppendix),
}

fn convert_node_to_native(
    node: &SdCompositionNode,
) -> (cuda::SdComposition, SdCompositionAppendix) {
    let mut native_entry = cuda::SdComposition {
        child: 0 as _,
        bound_min: node.bounding_box.0.to_array(),
        bound_max: node.bounding_box.1.to_array(),
        ..default()
    };

    native_entry.set_parent(0 as _);

    native_entry.set_variant(match &node.variant {
        SdCompositionNodeVariant::Primitive(_) => cuda::SdCompositionVariant_Union,
        SdCompositionNodeVariant::Union => cuda::SdCompositionVariant_Union,
        SdCompositionNodeVariant::Difference => cuda::SdCompositionVariant_Difference,
        SdCompositionNodeVariant::Intersect => cuda::SdCompositionVariant_Intersect,
        SdCompositionNodeVariant::Mirror(_, _) => cuda::SdCompositionVariant_Mirror,
    });

    native_entry.set_primitive_variant(match &node.variant {
        SdCompositionNodeVariant::Primitive(primitive) => match primitive.variant {
            SdPrimitiveNodeVariant::Empty => cuda::SdPrimitiveVariant_Empty,
            SdPrimitiveNodeVariant::Cube => cuda::SdPrimitiveVariant_Cube,
            SdPrimitiveNodeVariant::Sphere => cuda::SdPrimitiveVariant_Sphere,
            SdPrimitiveNodeVariant::Mandelbulb => cuda::SdPrimitiveVariant_Mandelbulb,
            SdPrimitiveNodeVariant::Mesh => cuda::SdPrimitiveVariant_Mesh,
            SdPrimitiveNodeVariant::Triangle => cuda::SdPrimitiveVariant_Triangle,
        },
        _ => cuda::SdPrimitiveVariant_None,
    });

    let appendix = match &node.variant {
        SdCompositionNodeVariant::Mirror(pos, dir) => {
            SdCompositionAppendix::Mirror(cuda::SdCompositionMirrorAppendix {
                translation: pos.to_array(),
                direction: dir.to_array(),
                ..default()
            })
        }
        SdCompositionNodeVariant::Primitive(primitive) => match primitive.variant {
            SdPrimitiveNodeVariant::Triangle => {
                let mut v1: Vec3 = Vec3::ZERO;
                let mut v2: Vec3 = Vec3::ZERO;
                let mut bb_v0: [bool; 3] = [false; 3];
                let vertices = [
                    primitive.triangle_v0,
                    primitive.triangle_v1,
                    primitive.triangle_v2,
                ];

                for i in 0..3 {
                    let v0 = vertices[i];

                    if (0..3).all(|axis| {
                        native_entry.bound_min[axis] == v0[axis]
                            || native_entry.bound_max[axis] == v0[axis]
                    }) {
                        v1 = vertices[(i + 1) % 3];
                        v2 = vertices[(i + 2) % 3];
                        bb_v0[0] = native_entry.bound_max[0] == v0[0];
                        bb_v0[1] = native_entry.bound_max[1] == v0[1];
                        bb_v0[2] = native_entry.bound_max[2] == v0[2];
                        break;
                    }
                }

                SdCompositionAppendix::PrimitiveTriangle(
                    cuda::SdCompositionPrimitiveTriangleAppendix {
                        bb_v0: bb_v0[0] as u32 * 1 + bb_v0[1] as u32 * 2 + bb_v0[2] as u32 * 4,
                        v1: v1.to_array(),
                        v2: v2.to_array(),
                        ..default()
                    },
                )
            }
            _ => SdCompositionAppendix::Primitive(cuda::SdCompositionPrimitiveAppendix {
                scale: primitive.scale.to_array(),
                rotation: [
                    primitive.rotation.w,
                    primitive.rotation.x,
                    primitive.rotation.y,
                    primitive.rotation.z,
                ],
                mesh_id: primitive.mesh_id,
                ..default()
            }),
        },
        _ => SdCompositionAppendix::None,
    };

    return (native_entry, appendix);
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
    mut render_scene_stored: ResMut<RenderSceneStoredGeometry>,
    settings: Res<RenderSceneSettings>,
    render_context: NonSend<RenderCudaContext>,
    mut gizmos: Gizmos,
    mut geometry: NonSendMut<RenderSceneGeometry>,
    mut nodes: Query<
        (
            Entity,
            &GlobalTransform,
            Option<&SdPrimitive>,
            Option<&SdComposition>,
            Option<&SdVisual>,
        ),
        Or<(With<SdPrimitive>, With<SdComposition>)>,
    >,
    changed_nodes: Query<
        Entity,
        (
            Or<(
                Changed<GlobalTransform>,
                Changed<SdPrimitive>,
                Changed<SdComposition>,
                Changed<SdVisual>,
            )>,
            Or<(With<SdPrimitive>, With<SdComposition>)>,
        ),
    >,
    mut meshes: Res<Assets<Mesh>>,
) {
    if !changed_nodes.is_empty() {
        let range_id = nvtx::range_start!("Compile Geometry");

        let start = std::time::Instant::now();

        let mut composition_index: usize = 0;
        let mut mesh_collection = HashMap::<Handle<Mesh>, u32>::default();

        let sub_range_id = nvtx::range_start!("Compile Geometry - Collect");
        let mut root = collect_scene_geometry(&mut nodes, &mut mesh_collection, &mut meshes);

        nvtx::range_end!(sub_range_id);

        fill_scene_geometry_non_leaf_bounding_boxes(&mut root);

        let sub_range_id = nvtx::range_start!("Compile Geometry - BVH");
        simple_center_split_reordering(&mut root);
        nvtx::range_end!(sub_range_id);

        render_scene_stored.root = Some(root.clone());

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

        // store compiled geometry

        unsafe {
            cudarc::driver::sys::cuEventSynchronize(
                render_context.geometry_transferred_event.clone(),
            )
            .result()
            .unwrap()
        };

        root.pad_with_empties();

        let mut vertex_offset = 0;
        let mut triangle_offset = 0;

        for (mesh_handle, mesh_id) in mesh_collection.iter() {
            if let Some(mesh) = meshes.get(mesh_handle) {
                compile_scene_mesh(
                    &mut geometry,
                    mesh,
                    mesh_id.clone(),
                    &mut vertex_offset,
                    &mut triangle_offset,
                );
            }
        }

        let mut queue = VecDeque::<(i32, usize, SdCompositionNode)>::new();
        queue.push_back((0, 0, root));

        while let Some((parent, child_idx, item)) = queue.pop_front() {
            assert!(
                match &item.variant {
                    SdCompositionNodeVariant::Primitive(_) => true,
                    _ => item.children.len() == 2,
                },
                "Must be leaf node or is binary, got {item:?} with {} children",
                item.children.len()
            );

            let (mut node, appendix) = convert_node_to_native(&item);
            node.set_parent(parent as _);

            if child_idx == 0 && parent >= 0 {
                geometry.compositions[parent as usize].child = composition_index as _;
            }

            geometry.compositions[composition_index] = node;

            queue.extend(
                item.children
                    .into_iter()
                    .enumerate()
                    .map(|(i, x)| (composition_index as i32, i, x)),
            );

            let (offset, appendix) = match appendix {
                SdCompositionAppendix::Mirror(item) => (
                    2,
                    Some(unsafe { std::mem::transmute::<_, cuda::SdComposition>(item) }),
                ),
                SdCompositionAppendix::Primitive(item) => (
                    2,
                    Some(unsafe { std::mem::transmute::<_, cuda::SdComposition>(item) }),
                ),
                SdCompositionAppendix::PrimitiveTriangle(item) => (
                    2,
                    Some(unsafe { std::mem::transmute::<_, cuda::SdComposition>(item) }),
                ),
                SdCompositionAppendix::None => (1, None),
            };

            if let Some(value) = appendix {
                geometry.compositions[composition_index + 1] = value;
            }
            composition_index += offset;
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

        nvtx::range_end!(range_id);
    }

    if settings.enable_debug_gizmos {
        if let Some(root) = render_scene_stored.root.as_ref() {
            draw_bb_gizmos(&mut gizmos, root);
        }
    }
}
