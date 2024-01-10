use bevy::prelude::*;
use bevy_editor_pls::EditorPlugin;
use bevy_flycam::NoCameraPlayerPlugin;

pub mod data;
pub mod example_scene;
pub mod renderer;

fn main() {
    let mut app = App::new();

    app.add_plugins((
        DefaultPlugins,
        EditorPlugin::default(),
        data::RayMarcherDataPlugin::default(),
        renderer::RayMarcherRenderPlugin::default(),
        // NoCameraPlayerPlugin,
    ));

    // app.add_systems(Startup, example_scene::setup_scene);

    app.run();
}
