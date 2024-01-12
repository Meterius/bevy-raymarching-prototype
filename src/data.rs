use crate::renderer::{RayMarcherCamera, RENDER_TEXTURE_SIZE};
use bevy::input::keyboard::KeyboardInput;
use bevy::input::mouse::MouseWheel;
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_resource::ShaderType;
use bevy::window::PrimaryWindow;

const DRAW_GIZMOS: bool = false;

#[derive(Copy, Clone, Debug, Default, Resource, Reflect, ExtractResource, ShaderType)]
#[reflect(Resource)]
pub struct RayMarcherFrameData {
    time: f32,
    texture_size: Vec2,
    screen_size: Vec2,
    aspect_ratio: f32,
    cam_unit_plane_dist: f32,
    cam_pos: Vec3,
    cam_forward: Vec3,
    cam_up: Vec3,
    cam_right: Vec3,
    sun_dir: Vec3,
    world_scale: f32,
}

#[derive(Default)]
pub struct RayMarcherDataPlugin {}

impl Plugin for RayMarcherDataPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<RayMarcherFrameData>()
            .insert_resource(RayMarcherFrameData {
                texture_size: Vec2::new(RENDER_TEXTURE_SIZE.0 as _, RENDER_TEXTURE_SIZE.1 as _),
                world_scale: 1.0,
                sun_dir: Vec3::new(0.5, 1.0, 3.0).normalize(),
                ..default()
            })
            .add_systems(PostUpdate, update_frame_data);
    }
}

fn update_frame_data(
    mut gizmos: Gizmos,
    window: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform), With<RayMarcherCamera>>,
    time: Res<Time>,
    mut frame_data: ResMut<RayMarcherFrameData>,
    mut keyboard_input: Res<Input<KeyCode>>,
) {
    // Scroll

    static MINIMAL_SCALE: f32 = 0.25;
    static MAXIMAL_SCALE: f32 = 20.0;

    let rx = keyboard_input.pressed(KeyCode::Numpad6) as i32
        - keyboard_input.pressed(KeyCode::Numpad4) as i32;
    let ry = keyboard_input.pressed(KeyCode::Numpad8) as i32
        - keyboard_input.pressed(KeyCode::Numpad2) as i32;
    let r = Vec2::new(rx as _, ry as _).normalize_or_zero();

    frame_data.sun_dir =
        Quat::from_rotation_y(-r.x * 0.2 * time.delta_seconds()).mul_vec3(frame_data.sun_dir);

    //

    frame_data.time = time.elapsed_seconds();

    let (cam, cam_transform) = camera.single();
    frame_data.aspect_ratio = cam.logical_viewport_size().unwrap_or_default().x
        / cam.logical_viewport_size().unwrap_or_default().y;
    frame_data.cam_pos = cam_transform.translation();
    frame_data.cam_forward = cam_transform.forward();
    frame_data.cam_right = cam_transform.right();
    frame_data.cam_up = cam_transform.up();
    frame_data.cam_unit_plane_dist = 1.25;
    frame_data.screen_size = Vec2::new(window.single().width(), window.single().height());

    if DRAW_GIZMOS {
        gizmos.sphere(frame_data.cam_pos, Quat::IDENTITY, 0.01, Color::BLUE);

        let ray = |v: Vec2| -> Vec3 {
            return frame_data.cam_forward * frame_data.cam_unit_plane_dist
                + frame_data.cam_right * v.x * 0.5 * frame_data.aspect_ratio
                + frame_data.cam_up * v.y * 0.5;
        };

        let viewport_pos = |v: Vec2| -> Vec3 {
            return frame_data.cam_pos + ray(v);
        };

        gizmos.ray(frame_data.cam_pos, ray(Vec2::new(-1.0, -1.0)), Color::BLUE);
        gizmos.ray(frame_data.cam_pos, ray(Vec2::new(1.0, -1.0)), Color::BLUE);
        gizmos.ray(frame_data.cam_pos, ray(Vec2::new(-1.0, 1.0)), Color::BLUE);
        gizmos.ray(frame_data.cam_pos, ray(Vec2::new(1.0, 1.0)), Color::BLUE);
        gizmos.ray(frame_data.cam_pos, ray(Vec2::new(0.0, 0.0)), Color::BLUE);

        gizmos.sphere(
            viewport_pos(Vec2::new(0.0, 0.0)),
            Quat::IDENTITY,
            0.01,
            Color::GREEN,
        );
        gizmos.sphere(
            viewport_pos(Vec2::new(-1.0, -1.0)),
            Quat::IDENTITY,
            0.05,
            Color::LIME_GREEN,
        );
        gizmos.sphere(
            viewport_pos(Vec2::new(1.0, -1.0)),
            Quat::IDENTITY,
            0.05,
            Color::LIME_GREEN,
        );
        gizmos.sphere(
            viewport_pos(Vec2::new(-1.0, 1.0)),
            Quat::IDENTITY,
            0.05,
            Color::LIME_GREEN,
        );
        gizmos.sphere(
            viewport_pos(Vec2::new(1.0, 1.0)),
            Quat::IDENTITY,
            0.05,
            Color::LIME_GREEN,
        );
    }
}
