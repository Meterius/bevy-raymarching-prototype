#include "./common.cu"

extern "C" __global__ void compute_render_finalize(
    const Texture render_texture,
    const RenderDataTexture render_data_texture,
    const GlobalsBuffer globals,
    const bool compression_enabled
) {
    u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    uvec2 texture_coord = render_texture_coord({ render_data_texture.size[0], render_data_texture.size[1] });
    u32 texture_index = id;

    RenderDataTextureValue texture_value = render_data_texture.texture[texture_index];

    float blended_steps = 0.0f;

    if (compression_enabled && false) {
        float total = 0.0f;

        for (int i = -STEP_GAUSSIAN_SIZE; i <= STEP_GAUSSIAN_SIZE; i++) {
            for (int j = -STEP_GAUSSIAN_SIZE; j <= STEP_GAUSSIAN_SIZE; j++) {
                int px = texture_coord.x + i;
                int py = texture_coord.y + j;

                if (0 <= px && px <= render_texture.size[0] && 0 <= py &&
                    py <= render_texture.size[1]) {
                    float fac;

                    if (render_data_texture.texture[id].outcome == Collision &&
                        render_data_texture
                            .texture[render_data_texture.size[0] * py + px]
                            .outcome == Collision) {
                        fac = clamp(
                            1.0f -
                            4.0f *
                            abs(
                                render_data_texture.texture[id].depth -
                                render_data_texture
                                    .texture
                                [render_data_texture.size[0] *
                                 py +
                                 px]
                                    .depth
                            ),
                            0.0f,
                            1.0f
                        );
                    } else if (render_data_texture.texture[id].outcome ==
                               DepthLimit &&
                               render_data_texture
                                   .texture[render_data_texture.size[0] *
                                            py +
                                            px]
                                   .outcome == DepthLimit) {
                        fac = 1.0f;
                    } else {
                        fac = 0.0f;
                    }

                    float val = step_gaussian_value(i, j) * fac;
                    total += val;
                    blended_steps +=
                        val *
                        render_data_texture
                            .texture[render_data_texture.size[0] * py + px]
                            .steps;
                }
            }
        }

        blended_steps /= total;
    } else {
        blended_steps = render_data_texture.texture[id].steps;
    }

    vec3 color;

    float geo_step_fac = texture_value.steps / (RAY_MARCH_STEP_LIMIT * 0.1f);
    float step_fac = texture_value.steps / (RAY_MARCH_STEP_LIMIT * 0.5f);
    float geo_powered_step_fac = pow(geo_step_fac, 3.0f);
    float powered_step_fac = pow(step_fac, 3.0f);

    if (texture_value.outcome == Collision) {
        color = from_array(texture_value.color) * texture_value.light *
                (1.0f + 5.0f * geo_powered_step_fac * float(globals.use_step_glow_on_foreground)) +
                vec3(texture_value.depth * 0.00001f);
        // color = from_array(texture_value.color);
        // color = vec3(texture_value.depth * 0.00005f);
    } else if (texture_value.outcome == DepthLimit) {
        color = vec3(vec3(0.2f, 0.4f, 1.0f) * 3.0f * 0.0f + texture_value.steps * 0.01f);
        color = vec3(0.0f) + 1.0f * powered_step_fac * float(globals.use_step_glow_on_background);
    } else {
        color = vec3(0.0, 1.0, 0.0) + from_array(texture_value.color);
    }

    color = hdr_map_aces_tone(max(color, 0.0f));

    unsigned int rgba = ((unsigned int) (255.0f * color.x) & 0xff) |
                        (((unsigned int) (255.0f * color.y) & 0xff) << 8) |
                        (((unsigned int) (255.0f * color.z) & 0xff) << 16) |
                        ((unsigned int) 255 << 24);

    render_texture.texture[index_2d(texture_coord, render_texture)] = rgba;
}
