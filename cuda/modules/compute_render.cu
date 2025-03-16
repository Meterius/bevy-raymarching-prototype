#include "./common.cu"

extern "C" __global__ void compute_render(
    const RenderDataTexture render_data_texture,
    const GlobalsBuffer globals,
    const CameraBuffer camera,
    const SceneBuffer scene,
    const Texture environment
) {
    // calculate ray

    u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    uvec2 texture_coord = render_texture_coord({ render_data_texture.size[0], render_data_texture.size[1] });

    if (texture_coord.y >= render_data_texture.size[1] || texture_coord.x >= render_data_texture.size[0]) {
        return;
    }

    u32 texture_index = id;

    vec2 ndc_coord = texture_to_ndc(
        texture_coord,
        { render_data_texture.size[0], render_data_texture.size[1] }
    );
    vec2 cam_coord = ndc_to_camera(
        ndc_coord, { render_data_texture.size[0], render_data_texture.size[1] }
    );
    Ray ray {
        { camera.position[0], camera.position[1], camera.position[2] },
        camera_to_ray(
            cam_coord,
            camera,
            from_array(globals.render_screen_size),
            vec2(globals.render_texture_size[0], globals.render_texture_size[1])
        )
    };

    // ray march and fill preliminary values in render data texture

    DefaultSignedDistanceScene sd_scene;
    RayRender ray_render = render_ray(ray, scene, sd_scene, environment);

    render_data_texture.texture[texture_index] = {
        ray_render.hit.depth,
        (float) ray_render.hit.steps,
        ray_render.hit.outcome,
        { ray_render.color.x, ray_render.color.y, ray_render.color.z },
    };
}
