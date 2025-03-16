use crate::renderer::RenderCameraTarget;
use bevy::prelude::*;
use bevy_flycam::FlyCam;

pub fn setup_scene(
    mut commands: Commands,
) {
    // camera
    commands.spawn((
        Camera3dBundle {
            camera: Camera {
                is_active: false,
                ..default()
            },
            transform: Transform::from_xyz(0.0, 2.0, 0.0)
                .looking_at(Vec3::new(20.0, 2.0, 0.0), Vec3::Y),
            ..default()
        },
        FlyCam,
        RenderCameraTarget::default(),
    ));
}


#[derive(Debug, Default, Resource, Reflect)]
#[reflect(Resource)]
pub struct ExampleSceneSettings {
}

#[derive(Default)]
pub struct ExampleScenePlugin {}

impl Plugin for ExampleScenePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ExampleSceneSettings::default());
    }
}
