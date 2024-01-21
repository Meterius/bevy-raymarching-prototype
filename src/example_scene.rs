use crate::renderer::scene::{SdPrimitive, SdVisual};
use crate::renderer::RenderCameraTarget;
use bevy::prelude::*;
use bevy_flycam::FlyCam;

pub fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // cube
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
            material: materials.add(Color::rgb_u8(124, 144, 255).into()),
            transform: Transform::from_xyz(0.0, 0.5, 0.0),
            ..default()
        },
        SdPrimitive::Box(Vec3::new(1.0, 1.0, 1.0)),
        SdVisual::default(),
    ));

    for i in 0..100 {
        commands.spawn((
            PbrBundle {
                mesh: meshes.add(Mesh::from(shape::Cube { size: 0.5 })),
                material: materials.add(Color::rgb_u8(124, 144, 255).into()),
                transform: Transform::from_xyz(-4.0 + (i as f32) * -3.0, 0.5, 0.0),
                ..default()
            },
            SdPrimitive::Sphere(0.5),
            SdVisual::default(),
        ));
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
