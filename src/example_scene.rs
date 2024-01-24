use crate::renderer::scene::{SdComposition, SdCompositionNodeVariant, SdPrimitive, SdVisual};
use crate::renderer::RenderCameraTarget;
use bevy::prelude::*;
use bevy_flycam::FlyCam;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const SAMPLE_SPHERE_DISTANCE: f32 = 120.0;
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
                    transform: Transform::from_xyz(0.0, 0.0, 0.0),
                    ..default()
                },
            ))
            .id();

        let child2 = commands
            .spawn((
                SdPrimitive::Sphere(2.0),
                SpatialBundle {
                    transform: Transform::from_xyz(0.0, 0.0, 0.0),
                    ..default()
                },
                AxisCyclicMotion {
                    direction: Vec3::new(0.0, 0.0, 1.75),
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
    spawn_example(SdCompositionNodeVariant::Difference);

    commands.spawn((
        SpatialBundle {
            transform: Transform::from_xyz(0.0, -0.5, 0.0),
            ..default()
        },
        SdPrimitive::Box(Vec3::new(30.0, 1.0, 30.0)),
        SdVisual::default(),
    ));

    commands.spawn((
        SpatialBundle {
            transform: Transform::from_xyz(-425.0, 0.0, 0.0),
            ..default()
        },
        SdPrimitive::Mandelbulb(400.0),
        SdVisual::default(),
    ));

    if true {
        for _ in 0..64 {
            let base = Vec3::new(
                ss_random.gen::<f32>() * SAMPLE_SPHERE_CHUNK_DISTANCE
                    - SAMPLE_SPHERE_CHUNK_DISTANCE * 0.5,
                ss_random.gen::<f32>() * SAMPLE_SPHERE_CHUNK_DISTANCE
                    - SAMPLE_SPHERE_CHUNK_DISTANCE * 0.5,
                ss_random.gen::<f32>() * SAMPLE_SPHERE_CHUNK_DISTANCE
                    - SAMPLE_SPHERE_CHUNK_DISTANCE * 0.5,
            );

            for _ in 0..64 {
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
                    SphericCyclicMotion {
                        distances: 50.0
                            * Vec3::new(
                                ss_random.gen::<f32>(),
                                ss_random.gen::<f32>(),
                                ss_random.gen::<f32>(),
                            ),
                        cycle_durations: 10.0 * Vec3::ONE
                            + 30.0
                                * Vec3::new(
                                    ss_random.gen::<f32>(),
                                    ss_random.gen::<f32>(),
                                    ss_random.gen::<f32>(),
                                ),
                        ..default()
                    },
                    TogglableVisual::default(),
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
            camera: Camera {
                is_active: false,
                ..default()
            },
            transform: Transform::from_xyz(5.0, 5.0, -5.0)
                .looking_at(Vec3::new(0.0, 2.0, 0.0), Vec3::Y),
            ..default()
        },
        FlyCam,
        RenderCameraTarget::default(),
    ));
}

#[derive(Debug, Clone, Component)]
pub struct SphericCyclicMotion {
    center: Option<Vec3>,
    distances: Vec3,
    cycle_durations: Vec3,
}

impl Default for SphericCyclicMotion {
    fn default() -> Self {
        Self {
            center: None,
            distances: Vec3::ONE,
            cycle_durations: Vec3::ONE * 5.0,
        }
    }
}

#[derive(Debug, Default, Clone, Component)]
pub struct TogglableVisual {}

#[derive(Debug, Clone, Component)]
pub struct AxisCyclicMotion {
    center: Option<Vec3>,
    direction: Vec3,
    cycle_duration: f32,
}

impl Default for AxisCyclicMotion {
    fn default() -> Self {
        Self {
            center: None,
            direction: Vec3::Y,
            cycle_duration: 5.0,
        }
    }
}

fn set_center(
    mut motions: Query<(&Transform, &mut AxisCyclicMotion), Added<AxisCyclicMotion>>,
    mut sphere_motions: Query<(&Transform, &mut SphericCyclicMotion), Added<SphericCyclicMotion>>,
) {
    for (trn, mut mot) in motions.iter_mut() {
        if mot.center.is_none() {
            mot.center = Some(trn.translation);
        }
    }

    for (trn, mut mot) in sphere_motions.iter_mut() {
        if mot.center.is_none() {
            mot.center = Some(trn.translation);
        }
    }
}

fn apply_motion(
    time: Res<Time>,
    mut motions: Query<(&mut Transform, &AxisCyclicMotion), Without<SphericCyclicMotion>>,
    mut sphere_motions: Query<(&mut Transform, &SphericCyclicMotion), Without<AxisCyclicMotion>>,
) {
    for (mut trn, mot) in motions.iter_mut() {
        trn.translation = mot.center.unwrap_or_default()
            + mot.direction
                * (2.0 * std::f32::consts::PI * time.elapsed_seconds() / mot.cycle_duration).sin();
    }

    for (mut trn, mot) in sphere_motions.iter_mut() {
        let d =
            Vec3::ONE * 2.0 * std::f32::consts::PI * time.elapsed_seconds() / mot.cycle_durations;

        trn.translation = mot.center.unwrap_or_default()
            + mot.distances * Vec3::new(d.x.sin(), d.y.sin(), d.z.sin());
    }
}

#[derive(Default)]
pub struct ExampleScenePlugin {}

impl Plugin for ExampleScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (set_center, apply_motion.after(set_center)));
    }
}
