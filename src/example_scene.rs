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
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut assets: Res<AssetServer>,
) {
    let mut ss_random = StdRng::seed_from_u64(0);

    let mut example_offset = 0.0;
    let mut spawn_example = |variant: SdCompositionNodeVariant| {
        let child1 = commands
            .spawn((
                SdPrimitive::Box(Vec3::new(5.0, 2.0, 1.0)),
                SpatialBundle::default(),
                RotateAxisMotion {
                    axis: Vec3::X,
                    cycle_duration: 10.0,
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

        let id = commands
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
                SdVisual { enabled: true },
                SpatialBundle {
                    transform: Transform::from_xyz(0.0, 2.0, example_offset),
                    ..default()
                },
            ))
            .push_children(&[child1, child2])
            .id();

        *&mut example_offset += 4.0;

        return id;
    };

    let mut mirrored = vec![
        // spawn_example(SdCompositionNodeVariant::Union),
        // spawn_example(SdCompositionNodeVariant::Intersect),
        // spawn_example(SdCompositionNodeVariant::Difference),
    ];

    commands
        .spawn((
            SpatialBundle {
                transform: Transform::from_xyz(0.0, -0.5, 0.0),
                ..default()
            },
            bevy::core::Name::new("Box"),
            SdVisual { enabled: false },
            SdPrimitive::Box(Vec3::new(30.0, 1.0, 30.0)),
        ))
        .id();

    commands.spawn((
        bevy::core::Name::new("Mirror"),
        SpatialBundle {
            transform: Transform::from_xyz(-10.0, 0.0, 0.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        SdComposition::Mirror(mirrored),
        SdVisual { enabled: false },
        AxisCyclicMotion {
            direction: Vec3::X * 15.0,
            cycle_duration: 30.0,
            ..default()
        },
    ));

    let mb_id = commands
        .spawn((
            SpatialBundle {
                transform: Transform::from_xyz(-425.0, 0.0, 0.0),
                ..default()
            },
            SdPrimitive::Mandelbulb(400.0),
            RotateAxisMotion {
                axis: Vec3::Y,
                cycle_duration: 60.0,
            },
            SdVisual { enabled: false },
        ))
        .id();

    commands.spawn((
        SpatialBundle {
            transform: Transform::from_xyz(-425.0 / 2.0, 0.0, 0.0).looking_to(Vec3::Z, Vec3::Y),
            ..default()
        },
        SdComposition::Mirror(vec![mb_id]),
        SdVisual { enabled: false },
    ));

    commands.spawn((
        SpatialBundle {
            transform: Transform::from_xyz(0.0, 5.0, 10.0),
            ..default()
        },
        SdPrimitive::Triangle([
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0, 2.0, 1.0),
            Vec3::new(-1.0, 1.0, 2.0),
        ]),
        SdVisual { enabled: false },
    ));

    for i in 0..5 {
        for j in 0..5 {
            commands.spawn((
                PbrBundle {
                    transform: Transform::from_xyz(
                        10.0 * i as f32 - 5.0,
                        1.0,
                        10.0 * j as f32 - 5.0,
                    ),
                    mesh: assets.load("models/monkey.obj"),
                    material: materials.add(Color::rgb_u8(124, 144, 255).into()),
                    ..default()
                },
                SdPrimitive::Mesh(assets.load("models/monkey.obj")),
                SdVisual { enabled: true },
                TogglableVisual::default(),
            ));
        }
    }

    if false {
        let mut sphere_clouds = Vec::new();

        commands
            .spawn((
                SpatialBundle::default(),
                bevy::core::Name::new("Sphere Clouds"),
            ))
            .with_children(|commands| {
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
                        sphere_clouds.push(
                            commands
                                .spawn((
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
                                    SdVisual { enabled: true },
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
                                ))
                                .id(),
                        );
                    }
                }
            });

        commands.spawn((
            bevy::core::Name::new("Cloud Sphere Mirror"),
            SpatialBundle {
                transform: Transform::from_translation(Vec3::ZERO)
                    .looking_at(Vec3::new(1.0, 1.0, 1.0), Vec3::Y),
                ..default()
            },
            SdComposition::Mirror(sphere_clouds),
            SdVisual { enabled: false },
        ));
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
            transform: Transform::from_xyz(5.0, 7.0, -5.0)
                .looking_at(Vec3::new(25.0, 2.0, 25.0), Vec3::Y),
            ..default()
        },
        FlyCam,
        RenderCameraTarget::default(),
    ));
}

#[derive(Debug, Clone, Component)]
pub struct RotateAxisMotion {
    axis: Vec3,
    cycle_duration: f32,
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
    settings: Res<ExampleSceneSettings>,
    time: Res<Time>,
    mut motions: Query<(
        &mut Transform,
        Option<&AxisCyclicMotion>,
        Option<&SphericCyclicMotion>,
        Option<&RotateAxisMotion>,
    )>,
) {
    if settings.enable_movement {
        for (mut trn, ax_mot, sp_mot, rot_mot) in motions.iter_mut() {
            if let Some(ax_mot) = ax_mot {
                trn.translation = ax_mot.center.unwrap_or_default()
                    + ax_mot.direction
                        * (2.0 * std::f32::consts::PI * time.elapsed_seconds()
                            / ax_mot.cycle_duration)
                            .sin();
            } else if let Some(sp_mot) = sp_mot {
                let d = Vec3::ONE * 2.0 * std::f32::consts::PI * time.elapsed_seconds()
                    / sp_mot.cycle_durations;

                trn.translation = sp_mot.center.unwrap_or_default()
                    + sp_mot.distances * Vec3::new(d.x.sin(), d.y.sin(), d.z.sin());
            }

            if let Some(rot_mot) = rot_mot {
                trn.rotation = Quat::from_axis_angle(
                    rot_mot.axis,
                    2.0 * std::f32::consts::PI * (time.elapsed_seconds() / rot_mot.cycle_duration),
                );
            }
        }
    }
}

#[derive(Debug, Default, Resource, Reflect)]
#[reflect(Resource)]
pub struct ExampleSceneSettings {
    pub enable_movement: bool,
}

#[derive(Default)]
pub struct ExampleScenePlugin {}

impl Plugin for ExampleScenePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ExampleSceneSettings::default())
            .add_systems(Update, (set_center, apply_motion.after(set_center)));
    }
}
