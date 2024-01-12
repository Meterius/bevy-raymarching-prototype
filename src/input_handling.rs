use bevy::prelude::*;

use crate::renderer::types::RenderScene;

pub fn receive_input(
    mut render_scene: ResMut<RenderScene>,
    keyboard_input: Res<Input<KeyCode>>,
) {
    // Sun Rotation

    let drx = keyboard_input.pressed(KeyCode::Numpad6) as i32
        - keyboard_input.pressed(KeyCode::Numpad4) as i32;
    let dry = keyboard_input.pressed(KeyCode::Numpad8) as i32
        - keyboard_input.pressed(KeyCode::Numpad2) as i32;
    let dr = Vec2::new(drx as _, dry as _).normalize_or_zero() * 0.01;

    render_scene.sun_direction = (
        Quat::from_axis_angle(Vec3::Y, -dr.x)
        * Quat::from_axis_angle(Vec3::Y.cross(render_scene.sun_direction), -dr.y)
        * render_scene.sun_direction
    ).normalize();
}
