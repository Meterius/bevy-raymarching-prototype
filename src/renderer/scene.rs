use bevy::prelude::*;

//

#[derive(Default, Debug, Clone, Resource, Reflect)]
#[reflect(Resource)]
pub struct RenderSceneSettings {
}

#[derive(Default)]
pub struct RenderScenePlugin {}

impl Plugin for RenderScenePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<RenderSceneSettings>()
            .insert_resource(RenderSceneSettings {
            });
    }
}
