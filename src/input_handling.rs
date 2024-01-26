use crate::example_scene::{ExampleSceneSettings, TogglableVisual};
use crate::renderer::scene::{RenderSceneSettings, SdVisual};
use crate::renderer::{RenderCameraTarget, RenderConeCompression, RenderRelayCameraTarget};
use bevy::{app::AppExit, prelude::*};
use bevy_flycam::MovementSettings;

pub fn receive_input(
    mut togglable_visuals: Query<&mut SdVisual, With<TogglableVisual>>,
    mut relay_camera: Query<
        &mut Camera,
        (With<RenderRelayCameraTarget>, Without<RenderCameraTarget>),
    >,
    mut render_camera: Query<
        &mut Camera,
        (With<RenderCameraTarget>, Without<RenderRelayCameraTarget>),
    >,
    mut movement_settings: ResMut<MovementSettings>,
    mut compression_settings: ResMut<RenderConeCompression>,
    mut render_settings: ResMut<RenderSceneSettings>,
    mut e_scene_settings: ResMut<ExampleSceneSettings>,
    keyboard_input: Res<Input<KeyCode>>,
    mut exit: EventWriter<AppExit>,
) {
    if keyboard_input.just_pressed(KeyCode::Escape) {
        exit.send(AppExit);
    }

    if keyboard_input.just_pressed(KeyCode::ControlLeft) {
        movement_settings.speed = 200.0;
    } else if keyboard_input.just_released(KeyCode::ControlLeft) {
        movement_settings.speed = 12.0;
    }

    if keyboard_input.just_pressed(KeyCode::C) {
        compression_settings.enabled = !compression_settings.enabled;
    }

    if keyboard_input.just_pressed(KeyCode::Y) {
        let toggle = !render_settings.enable_debug_gizmos;
        render_settings.enable_debug_gizmos = toggle;
        relay_camera.single_mut().is_active = !toggle;
        render_camera.single_mut().is_active = toggle;
    }

    if keyboard_input.just_pressed(KeyCode::X) {
        let (foreground, background) = match (
            render_settings.enable_step_glow_on_foreground,
            render_settings.enable_step_glow_on_background,
        ) {
            (false, false) => (false, true),
            (false, true) => (true, false),
            (true, false) => (true, true),
            (true, true) => (false, false),
        };

        render_settings.enable_step_glow_on_foreground = foreground;
        render_settings.enable_step_glow_on_background = background;
    }

    if keyboard_input.just_pressed(KeyCode::V) {
        for mut vis in togglable_visuals.iter_mut() {
            vis.enabled = !vis.enabled;
        }
    }

    if keyboard_input.just_pressed(KeyCode::B) {
        e_scene_settings.enable_movement = !e_scene_settings.enable_movement;
    }
}
