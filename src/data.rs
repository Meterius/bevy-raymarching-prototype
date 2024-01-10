use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_resource::ShaderType;

#[derive(Copy, Clone, Debug, Default, Resource, ExtractResource, Reflect, ShaderType)]
#[reflect(Resource)]
pub struct RayMarcherFrameData {
    time: f32,
}

#[derive(Default)]
pub struct RayMarcherDataPlugin {}

impl Plugin for RayMarcherDataPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(RayMarcherFrameData::default())
            .add_systems(Last, update_frame_data);
    }
}

fn update_frame_data(time: Res<Time>, mut frame_data: ResMut<RayMarcherFrameData>) {
    frame_data.time = time.elapsed_seconds();
}
