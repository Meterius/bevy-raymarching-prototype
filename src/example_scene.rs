use crate::renderer::scene::{SdComposition, SdCompositionNodeVariant, SdPrimitive, SdVisual};
use crate::renderer::RenderCameraTarget;
use bevy::prelude::*;
use bevy_flycam::FlyCam;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const SAMPLE_SPHERE_DISTANCE: f32 = 60.0;
const SAMPLE_SPHERE_CHUNK_DISTANCE: f32 = 1000.0;

pub fn setup_scene(
    mut commands: Commands,
    _meshes: ResMut<Assets<Mesh>>,
    _materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut ss_random = StdRng::seed_from_u64(0);

    let mut example_offset = 0.0;
    let mut spawn_example = |variant: SdCompositionNodeVariant| {
        let child1 = commands
            .spawn((
                SdPrimitive::Box(Vec3::new(5.0, 2.0, 1.0)),
                SpatialBundle {
                    transform: Transform::from_xyz(0.0, 0.0, -0.5),
                    ..default()
                },
            ))
            .id();

        let child2 = commands
            .spawn((
                SdPrimitive::Sphere(1.0),
                SpatialBundle {
                    transform: Transform::from_xyz(0.0, 0.0, 0.0),
                    ..default()
                },
            ))
            .id();

        commands
            .spawn((
                match variant {
                    SdCompositionNodeVariant::Union => SdComposition::Union(vec![child1, child2]),
                    SdCompositionNodeVariant::Intersect => {
                        SdComposition::Intersect(vec![child1, child2])
                    }
                    SdCompositionNodeVariant::Difference => {
                        SdComposition::Difference(vec![child1, child2])
                    }
                    _ => unimplemented!(),
                },
                SdVisual::default(),
                SpatialBundle {
                    transform: Transform::from_xyz(0.0, 2.0, example_offset),
                    ..default()
                },
            ))
            .push_children(&[child1, child2]);

        *&mut example_offset += 4.0;
    };

    spawn_example(SdCompositionNodeVariant::Union);
    spawn_example(SdCompositionNodeVariant::Intersect);

    if false {
        for _ in 0..16 {
            let base = Vec3::new(
                ss_random.gen::<f32>() * SAMPLE_SPHERE_CHUNK_DISTANCE
                    - SAMPLE_SPHERE_CHUNK_DISTANCE * 0.5,
                ss_random.gen::<f32>() * SAMPLE_SPHERE_CHUNK_DISTANCE
                    - SAMPLE_SPHERE_CHUNK_DISTANCE * 0.5,
                ss_random.gen::<f32>() * SAMPLE_SPHERE_CHUNK_DISTANCE
                    - SAMPLE_SPHERE_CHUNK_DISTANCE * 0.5,
            );

            for _ in 0..4 {
                commands.spawn((
                    /*PbrBundle {
                        mesh: meshes.add(Mesh::from(shape::Cube { size: 0.5 })),
                        material: materials.add(Color::rgb_u8(124, 144, 255).into()),
                        ..default()
                    },*/
                    SpatialBundle {
                        transform: Transform::from_translation(
                            base + Vec3::new(
                                ss_random.gen::<f32>() * SAMPLE_SPHERE_DISTANCE
                                    - SAMPLE_SPHERE_DISTANCE * 0.5,
                                ss_random.gen::<f32>() * SAMPLE_SPHERE_DISTANCE
                                    - SAMPLE_SPHERE_DISTANCE * 0.5,
                                ss_random.gen::<f32>() * SAMPLE_SPHERE_DISTANCE
                                    - SAMPLE_SPHERE_DISTANCE * 0.5,
                            ),
                        ),
                        ..default()
                    },
                    SdPrimitive::Sphere(0.25 + ss_random.gen::<f32>() * 1.75),
                    SdVisual::default(),
                ));
            }
        }
    }

    /*    // light
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });*/

    // camera
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(5.0, 5.0, -5.0)
                .looking_at(Vec3::new(0.0, 2.0, 0.0), Vec3::Y),
            ..default()
        },
        FlyCam,
        RenderCameraTarget::default(),
    ));
}
