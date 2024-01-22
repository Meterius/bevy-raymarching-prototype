use crate::renderer::scene::{SdPrimitive, SdVisual};
use crate::renderer::RenderCameraTarget;
use bevy::prelude::*;
use bevy_flycam::FlyCam;

const SAMPLE_SPHERE_DISTANCE: f32 = 20.0;
const SAMPLE_SPHERE_CHUNK_DISTANCE: f32 = 200.0;

pub fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for _ in 0..32 {
        let base = Vec3::new(
            rand::random::<f32>() * SAMPLE_SPHERE_CHUNK_DISTANCE
                - SAMPLE_SPHERE_CHUNK_DISTANCE * 0.5,
            rand::random::<f32>() * SAMPLE_SPHERE_CHUNK_DISTANCE
                - SAMPLE_SPHERE_CHUNK_DISTANCE * 0.5,
            rand::random::<f32>() * SAMPLE_SPHERE_CHUNK_DISTANCE
                - SAMPLE_SPHERE_CHUNK_DISTANCE * 0.5,
        );

        for _ in 0..64 {
            commands.spawn((
                PbrBundle {
                    mesh: meshes.add(Mesh::from(shape::Cube { size: 0.5 })),
                    material: materials.add(Color::rgb_u8(124, 144, 255).into()),
                    transform: Transform::from_translation(
                        base + Vec3::new(
                            rand::random::<f32>() * SAMPLE_SPHERE_DISTANCE
                                - SAMPLE_SPHERE_DISTANCE * 0.5,
                            rand::random::<f32>() * SAMPLE_SPHERE_DISTANCE
                                - SAMPLE_SPHERE_DISTANCE * 0.5,
                            rand::random::<f32>() * SAMPLE_SPHERE_DISTANCE
                                - SAMPLE_SPHERE_DISTANCE * 0.5,
                        ),
                    ),
                    ..default()
                },
                SdPrimitive::Sphere(0.25),
                SdVisual::default(),
            ));
        }
    }

    // light
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });
    // camera
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(-50.0, 2.0, 5.0)
                .looking_at(Vec3::new(0.0, 2.0, -40.0), Vec3::Y),
            ..default()
        },
        FlyCam,
        RenderCameraTarget::default(),
    ));
}
