use crate::renderer::scene::RenderSceneSettings;
use crate::renderer::{RenderConeCompression, RenderTargetSprite};
use bevy::{app::AppExit, prelude::*};
use bevy_flycam::MovementSettings;

pub fn receive_input(
    mut render_sprite: Query<&mut Sprite, With<RenderTargetSprite>>,
    mut movement_settings: ResMut<MovementSettings>,
    mut compression_settings: ResMut<RenderConeCompression>,
    mut render_settings: ResMut<RenderSceneSettings>,
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
        render_sprite
            .single_mut()
            .color
            .set_a(if toggle { 0.5 } else { 1.0 });
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
}
