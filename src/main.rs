use bevy::diagnostic::{EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::window::{PresentMode, WindowResolution, CursorGrabMode, PrimaryWindow};
use bevy_editor_pls::EditorPlugin;
use bevy_flycam::NoCameraPlayerPlugin;

pub mod input_handling;
pub mod example_scene;
pub mod renderer;

fn main() {
    let mut app = App::new();

    app.insert_resource(Msaa::Sample8).add_plugins((
        DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                position: WindowPosition::Centered(MonitorSelection::Index(1)),
                present_mode: PresentMode::AutoVsync,
                resolution: WindowResolution::new(1280., 720.),
                ..default()
            }),
            ..default()
        }),
        FrameTimeDiagnosticsPlugin::default(),
        EntityCountDiagnosticsPlugin::default(),
        EditorPlugin::default(),
        renderer::RayMarcherRenderPlugin::default(),
        NoCameraPlayerPlugin,
    ));

    app.add_systems(PostStartup, |
        mut primary_window: Query<&mut Window, With<PrimaryWindow>>,
        mut key_binds: ResMut<bevy_flycam::KeyBindings>,
    | {
        let mut window = primary_window.single_mut();
        window.cursor.grab_mode = CursorGrabMode::None;
        window.cursor.visible = true;
        key_binds.toggle_grab_cursor = KeyCode::F;
    });

    app.add_systems(Startup, example_scene::setup_scene);
    app.add_systems(Update, input_handling::receive_input);

    app.run();
}
