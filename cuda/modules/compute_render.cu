#include "./common.cu"

extern "C" __global__ void compute_render(
    const RenderDataTexture render_data_texture,
    const ConeMarchTextures cm_textures,
    const GlobalsBuffer globals,
    const CameraBuffer camera,
    const SdRuntimeScene runtime_scene_param,
    const bool compression_enabled
) {
    composition_traversal_count[threadIdx.x] = 0;

    if (!threadIdx.x) {
        runtime_scene = runtime_scene_param;
    }

    __syncthreads();

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

    // if enabled, fetch cone march compression starting point

#ifndef DISABLE_CONE_MARCH
    ConeMarchTextureValue entry = { 0.0f, 0, Collision };

    if (compression_enabled) {
        uvec2 cm_texture_coord = ndc_to_texture(
            ndc_coord,
            {
                (float) cm_textures.textures[CONE_MARCH_LEVELS - 1].size[0],
                (float) cm_textures.textures[CONE_MARCH_LEVELS - 1].size[1]
            }
        );

        entry = cm_textures.textures[CONE_MARCH_LEVELS - 1].texture
        [cm_texture_coord.x +
         cm_textures.textures[CONE_MARCH_LEVELS - 1].size[0] *
         cm_texture_coord.y];

        if (COMPRESSION_STEP_INTERPOLATION) {
            entry.steps = ndc_to_interpolated_value(
                ndc_coord,
                cm_textures.textures[CONE_MARCH_LEVELS - 1],
                [](ConeMarchTextureValue entry) { return entry.steps; }
            );
        }
    }
#else
    float interpolated_cm_steps = 0.0f;
    ConeMarchTextureValue entry = {0.0f, 0, Collision};
#endif

    float cone_radius_at_unit = get_pixel_cone_radius(
        texture_coord, camera, render_data_texture,
        globals
    );

    // ray march and fill preliminary values in render data texture

    RayRender ray_render = render_ray(
        ray,
        cone_radius_at_unit,
        make_sdi_scene<SdInvocationType::ConeType>(globals, camera),
        make_sdi_scene<SdInvocationType::SurfaceType>(globals, camera),
        make_sdi_scene<SdInvocationType::PointType>(globals, camera),
        make_sdi_scene<SdInvocationType::RayType>(globals, camera),
        runtime_scene.lighting,
        entry
    );

    // ray_render.color = vec3(clamp((float) ray_render.hit.cycles * 0.000001f, 0.0f, 1.0f));

    render_data_texture.texture[texture_index] = {
        ray_render.hit.depth,
        (float) ray_render.hit.steps + entry.steps,
        ray_render.hit.outcome,
        { ray_render.color.x, ray_render.color.y, ray_render.color.z },
        ray_render.light
    };
}
