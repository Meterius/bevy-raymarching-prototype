use bevy::diagnostic::{EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::window::WindowResolution;
use bevy_editor_pls::EditorPlugin;
use bevy_flycam::NoCameraPlayerPlugin;

pub mod data;
pub mod example_scene;
pub mod renderer;

fn main() {
    let mut app = App::new();

    app.insert_resource(Msaa::Sample8).add_plugins((
        DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                resolution: WindowResolution::new(1920., 1080.),
                ..default()
            }),
            ..default()
        }),
        FrameTimeDiagnosticsPlugin::default(),
        EntityCountDiagnosticsPlugin::default(),
        EditorPlugin::default(),
        data::RayMarcherDataPlugin::default(),
        renderer::RayMarcherRenderPlugin::default(),
        NoCameraPlayerPlugin,
    ));

    app.add_systems(Startup, example_scene::setup_scene);

    app.run();
}
