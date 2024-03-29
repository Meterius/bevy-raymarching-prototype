use crate::example_scene::ExampleScenePlugin;
use bevy::diagnostic::{EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::render::settings::{Backends, RenderCreation, WgpuSettings};
use bevy::render::RenderPlugin;
use bevy::window::{CursorGrabMode, PresentMode, PrimaryWindow, WindowResolution};
use bevy_editor_pls::EditorPlugin;
use bevy_flycam::NoCameraPlayerPlugin;
use bevy_obj::ObjPlugin;

pub mod bindings;
pub mod cudarc_extension;
pub mod example_scene;
pub mod input_handling;
pub mod renderer;

fn main() {
    unsafe { cudarc::driver::sys::cuProfilerStart() };

    let mut app = App::new();

    app.insert_resource(Msaa::Sample8).add_plugins((
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    position: WindowPosition::Centered(MonitorSelection::Index(0)),
                    present_mode: PresentMode::AutoVsync,
                    resolution: WindowResolution::new(1920., 1080.),
                    ..default()
                }),
                ..default()
            })
            .set(RenderPlugin {
                render_creation: RenderCreation::Automatic(WgpuSettings {
                    backends: Some(Backends::VULKAN),
                    ..default()
                }),
            }),
        ObjPlugin,
        FrameTimeDiagnosticsPlugin::default(),
        EntityCountDiagnosticsPlugin::default(),
        EditorPlugin::default(),
        renderer::RayMarcherRenderPlugin::default(),
        renderer::scene::RenderScenePlugin::default(),
        NoCameraPlayerPlugin,
        ExampleScenePlugin::default(),
    ));

    app.add_systems(
        PostStartup,
        |mut primary_window: Query<&mut Window, With<PrimaryWindow>>,
         mut key_binds: ResMut<bevy_flycam::KeyBindings>| {
            let mut window = primary_window.single_mut();
            window.cursor.grab_mode = CursorGrabMode::None;
            window.cursor.visible = true;
            key_binds.toggle_grab_cursor = KeyCode::F;
        },
    );

    app.add_systems(Startup, example_scene::setup_scene);
    app.add_systems(Update, input_handling::receive_input);

    app.run();
}
