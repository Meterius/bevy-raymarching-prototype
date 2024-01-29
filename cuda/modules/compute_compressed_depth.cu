#include "./common.cu"

extern "C" __global__ void compute_compressed_depth(
    const unsigned int level,
    const RenderDataTexture render_data_texture,
    const ConeMarchTextures cm_textures,
    const GlobalsBuffer globals,
    const CameraBuffer camera,
    const SdRuntimeScene runtime_scene_param
) {
    if (!threadIdx.x) {
        runtime_scene = runtime_scene_param;
    }

    __syncthreads();

    u32 id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id > cm_textures.textures[level].size[0] *
             cm_textures.textures[level].size[1]) {
        return;
    }

    uvec2 cm_texture_coord = uvec2(
        id % cm_textures.textures[level].size[0],
        id / cm_textures.textures[level].size[0]
    );
    vec2 ndc_coord = texture_to_ndc(
        cm_texture_coord,
        {
            cm_textures.textures[level].size[0],
            cm_textures.textures[level].size[1]
        }
    );
    vec2 cam_coord = ndc_to_camera(
        ndc_coord,
        {
            cm_textures.textures[level].size[0],
            cm_textures.textures[level].size[1]
        }
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

    float cone_radius_at_unit = get_pixel_cone_radius(
        cm_texture_coord, camera, cm_textures.textures[level],
        globals
    );

    ConeMarchTextureValue entry { 0.0f, 0, Collision };
    if (level > 0) {
        uvec2 lower_cm_texture_coord = ndc_to_texture(
            ndc_coord,
            {
                (float) cm_textures.textures[level - 1].size[0],
                (float) cm_textures.textures[level - 1].size[1]
            }
        );

        entry = cm_textures.textures[level - 1].texture
        [lower_cm_texture_coord.x +
         cm_textures.textures[level - 1].size[0] *
         lower_cm_texture_coord.y];

        if (COMPRESSION_STEP_INTERPOLATION) {
            entry.steps = ndc_to_interpolated_value(
                ndc_coord,
                cm_textures.textures[level - 1],
                [](ConeMarchTextureValue entry) { return entry.steps; }
            );
        }
    }

    RayMarchHit hit = ray_march(
        make_sdi_scene<SdInvocationType::ConeType>(globals, camera),
        ray,
        entry,
        cone_radius_at_unit
    );

    if (RELATIVIZE_STEP_COUNT) {
        float compression_factor =
            (float) (render_data_texture.size[0] * render_data_texture.size[1]) /
            (float) (cm_textures.textures[level].size[0] *
                     cm_textures.textures[level].size[1]);
        hit.steps = (int) ceil((float) hit.steps / compression_factor);
    }

    cm_textures.textures[level].texture[id] = ConeMarchTextureValue {
        hit.depth, (float) hit.steps + entry.steps, hit.outcome
    };

    /*
    __syncthreads();

   if (hit.outcome == Collision) {
       float total = (float) hit.steps + entry.steps;
       float value = 1.0f;

       for (int i = -1; i <= 1; i++) {
           for (int j = -1; j <= 1; j++) {
               ConeMarchTextureValue item = fetch_2d(
                   ivec2(cm_texture_coord.x + i, cm_texture_coord.y + j),
                   cm_textures.textures[level],
                   [](auto item) { return item; }
               );

               if (item.outcome != Collision) {
                   value += item.steps;
                   total += 1.0f;
               }
           }
       }

       value /= total;
       cm_textures.textures[level].texture[id].steps = value;
   */
}
