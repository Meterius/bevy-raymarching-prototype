use bevy::prelude::*;
use bevy_editor_pls::EditorPlugin;
use bevy_flycam::NoCameraPlayerPlugin;

pub mod example_scene;

fn main() {
    let mut app = App::new();

    app.add_plugins((
        DefaultPlugins,
        EditorPlugin::default(),
        NoCameraPlayerPlugin,
    ));

    app.add_systems(Startup, example_scene::setup_scene);

    app.run();
}
