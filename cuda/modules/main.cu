#include "../includes/libraries/glm/glm.hpp"
#include "../includes/bindings.h"
#include "../includes/color.cu"
#include "../includes/utils.h"
#include "../includes/signed_distance.cu"
#include "../includes/rendering.cu"
#include "../includes/ray_marching.cu"

using namespace glm;

#define RELATIVIZE_STEP_COUNT false
#define COMPRESSION_STEP_INTERPOLATION false

// coordinate system conversion

__device__ vec2 texture_to_ndc(vec2 p, vec2 texture_size) {
    return (p + vec2(0.5f, 0.5f)) / texture_size;
}

__device__ uvec2 ndc_to_texture(vec2 p, vec2 texture_size) {
    return uvec2(round((p * texture_size) - vec2(0.5f, 0.5f)));
}

template<typename Func, typename Texture>
__device__ auto fetch_2d(ivec2 p, Texture &texture, Func map) {
    return map(
        texture.texture
        [min(max(p.x, 0), texture.size[0] - 1) +
         min(max(p.y, 0), texture.size[1] - 1) * texture.size[0]]
    );
}

template<typename Texture>
__device__ auto index_2d(ivec2 p, Texture &texture) {
    return min(max(p.x, 0), texture.size[0] - 1) + min(max(p.y, 0), texture.size[1] - 1) * texture.size[0];
}


__device__ float
cubic_interpolate(float y0, float y1, float y2, float y3, float rx1) {
    return y1 + 0.5f * rx1 *
                (y2 - y0 +
                 rx1 * (2.0f * y0 - 5.0f * y1 + 4.0f * y2 - y3 +
                        rx1 * (3.0f * (y1 - y2) + y3 - y0)));
}

template<typename Func, typename Texture>
__device__ auto ndc_to_interpolated_value(vec2 p, Texture &texture, Func map) {
    vec2 t = (p * vec2((float) texture.size[0], (float) texture.size[1])) -
             vec2(0.5f, 0.5f);
    ivec2 tc = ivec2(floor(t));

    float interps[4];
    for (int i = 0; i < 4; i++) {
        interps[i] = cubic_interpolate(
            fetch_2d(ivec2(tc.x - 1, tc.y + i - 1), texture, map),
            fetch_2d(ivec2(tc.x, tc.y + i - 1), texture, map),
            fetch_2d(ivec2(tc.x + 1, tc.y + i - 1), texture, map),
            fetch_2d(ivec2(tc.x + 2, tc.y + i - 1), texture, map),
            t.x - (float) tc.x
        );
    }

    return cubic_interpolate(
        interps[0], interps[1], interps[2], interps[3], t.y - (float) tc.y
    );
}

__device__ vec2 ndc_to_camera(vec2 p, vec2 render_screen_size) {
    return {
        (2 * p.x - 1) * (render_screen_size.x / render_screen_size.y),
        1 - 2 * p.y
    };
}

__device__ vec3 camera_to_ray(
    vec2 p, CameraBuffer CAMERA, vec2 screen_size, vec2 texture_size
) {
    float width_factor =
        (screen_size.x / texture_size.x) * (texture_size.y / screen_size.y);

    float fov_fac = tan(CAMERA.fov / 2);
    return normalize(
        vec3(CAMERA.forward[0], CAMERA.forward[1], CAMERA.forward[2]) +
        p.y * fov_fac * vec3(CAMERA.up[0], CAMERA.up[1], CAMERA.up[2]) +
        p.x * fov_fac * width_factor *
        vec3(CAMERA.right[0], CAMERA.right[1], CAMERA.right[2])
    );
}

// scene

__device__ SdRuntimeScene runtime_scene;

__device__ auto make_sds_scene(GlobalsBuffer &globals, CameraBuffer &camera) {
    auto box_sds = make_generic_sds(
        [](vec3 p, float cd) {
            return sd_box(p, vec3(-15.0f, 1.0f, 0.0f), vec3(1.0f, 2.0f, 10.0f));
        },
        RenderSurfaceData { vec3(1.0, 0.0, 0.0) }
    );

    auto runtime_scene_sds = make_generic_sds(
        [](vec3 p, float cd) {
            return sd_composition(p, cd, runtime_scene.geometry, 0);
        },
        RenderSurfaceData { vec3(0.4, 0.5, 0.2) }
    );

    auto plane_scene_sds = make_generic_sds(
        [](vec3 p, float cd) {
            return sd_box(p, vec3(0.0f, -0.5f, 0.0f), vec3(50.0f, 1.0f, 50.0f));
        },
        RenderSurfaceData { vec3(0.3f, 0.4f, 0.2f) }
    );

    auto axes_scene_sds = make_generic_location_dependent_sds(
        [](vec3 p, float cd) { return sd_axes(p); },
        [](vec3 p, float cd) {
            vec3 color;
            color.x = (p.x > 0.25 ? 1.0f : (p.x < 0.25 ? 0.5f : 0.0f));
            color.y = (p.y > 0.25 ? 1.0f : (p.y < 0.25 ? 0.5f : 0.0f));
            color.z = (p.z > 0.25 ? 1.0f : (p.z < 0.25 ? 0.5f : 0.0f));
            return RenderSurfaceData { color };
        }
    );

    auto mandelbulb_scene_sds = make_generic_sds(
        [&](vec3 p, float cd) {
            const float scale = 100.0f;
            p /= scale;
            p.z = wrap(p.z, -1.0f, 1.0f);
            p.x -= 1.0f;
            return sd_mandelbulb(p, globals.time) * scale;
        },
        RenderSurfaceData { vec3(1.5f, 0.1f, 1.9f) }
    );

    return [=](vec3 p, float cd, RenderSurfaceData &surface_output) {
        return runtime_scene_sds(p, cd, surface_output);
    };
}

// ray-marching

#include <cuda_runtime.h>

template<typename Texture>
__device__ float get_pixel_cone_radius(
    uvec2 texture_coord,
    CameraBuffer &camera,
    Texture &texture,
    GlobalsBuffer &globals
) {
    const auto texture_to_dir = [&camera, &texture, &globals](
        vec2 p
    ) {
        vec2 ndc_coord = texture_to_ndc(
            p,
            {
                texture.size[0],
                texture.size[1]
            }
        );
        vec2 cam_coord = ndc_to_camera(
            ndc_coord,
            {
                texture.size[0],
                texture.size[1]
            }
        );
        return camera_to_ray(
            cam_coord,
            camera,
            from_array(globals.render_screen_size),
            vec2(globals.render_texture_size[0], globals.render_texture_size[1])
        );
    };

    vec2 ndc_coord = texture_to_ndc(
        texture_coord,
        {
            texture.size[0],
            texture.size[1]
        }
    );
    vec2 cam_coord = ndc_to_camera(
        ndc_coord,
        {
            texture.size[0],
            texture.size[1]
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

    vec3 border_dirs[4] = {
        texture_to_dir(
            {
                (float) texture_coord.x - SQRT_INV,
                (float) texture_coord.y - SQRT_INV
            }
        ),
        texture_to_dir(
            {
                (float) texture_coord.x - SQRT_INV,
                (float) texture_coord.y + SQRT_INV
            }
        ),
        texture_to_dir(
            {
                (float) texture_coord.x + SQRT_INV,
                (float) texture_coord.y - SQRT_INV
            }
        ),
        texture_to_dir(
            {
                (float) texture_coord.x + SQRT_INV,
                (float) texture_coord.y + SQRT_INV
            }
        ),
    };

    return max(
        max(
            length(ray.direction - border_dirs[0]),
            length(ray.direction - border_dirs[1])),
        max(
            length(ray.direction - border_dirs[2]),
            length(ray.direction - border_dirs[3])));
}

#ifndef DISABLE_CONE_MARCH
extern "C" __global__ void compute_compressed_depth(
    unsigned int level,
    RenderDataTexture render_data_texture,
    ConeMarchTextures cm_textures,
    GlobalsBuffer globals,
    CameraBuffer camera,
    SdRuntimeScene runtime_scene_param
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

    auto sds = make_sds_scene(globals, camera);
    RenderSurfaceData surface {};
    RayMarchHit hit = ray_march<true>(
        [&surface, &sds](vec3 p, float cd) { return sds(p, cd, surface); },
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
#endif

__device__ ivec2 render_texture_coord(ivec2 render_texture_size) {
    const int WARP_H = 8;
    const int WARP_W = 4;

    int warp_local_id = threadIdx.x % 32;
    ivec2 warp_local_coord = { warp_local_id % WARP_W, warp_local_id / WARP_W };

    const int BLOCK_WARP_H = 2;
    const int BLOCK_WARP_W = (BLOCK_SIZE / BLOCK_WARP_H) / 32;

    int warp_id = threadIdx.x / 32;
    ivec2 warp_coord = { warp_id % BLOCK_WARP_W, warp_id / BLOCK_WARP_W };

    ivec2 block_local_texture_coord = {
        warp_local_coord.x + WARP_W * warp_coord.x,
        warp_local_coord.y + WARP_H * warp_coord.y
    };

    ivec2 block_count = {
        render_texture_size.x / (WARP_W * BLOCK_WARP_W),
        render_texture_size.x / (WARP_H * BLOCK_WARP_H),
    };

    ivec2 block_texture_coord = {
        (WARP_W * BLOCK_WARP_W) * (blockIdx.x % block_count.x),
        (WARP_H * BLOCK_WARP_H) * (blockIdx.x / block_count.x)
    };

    return block_texture_coord + block_local_texture_coord;
}

extern "C" __global__ void compute_render(
    RenderDataTexture render_data_texture,
    ConeMarchTextures cm_textures,
    GlobalsBuffer globals,
    CameraBuffer camera,
    SdRuntimeScene runtime_scene_param,
    bool compression_enabled
) {
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

    auto sds = make_sds_scene(globals, camera);
    RayRender ray_render = render_ray(
        ray,
        cone_radius_at_unit,
        sds,
        runtime_scene.lighting,
        entry
    );

    render_data_texture.texture[texture_index] = {
        ray_render.hit.depth,
        (float) ray_render.hit.steps + entry.steps,
        ray_render.hit.outcome,
        { ray_render.color.x, ray_render.color.y, ray_render.color.z },
        ray_render.light
    };
}

#define STEP_GAUSSIAN_SIZE 24
#define STEP_GAUSSIAN_DEV 12.0f

__device__ float step_gaussian_value(int i, int j) {
    return exp(
        -((float) (i * i + j * j) /
          (2.0f * STEP_GAUSSIAN_DEV * STEP_GAUSSIAN_DEV))
    ) /
           (2.0f * PI * STEP_GAUSSIAN_DEV * STEP_GAUSSIAN_DEV);
}

extern "C" __global__ void compute_render_finalize(
    Texture render_texture,
    RenderDataTexture render_data_texture,
    GlobalsBuffer globals,
    bool compression_enabled
) {
    u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    uvec2 texture_coord = render_texture_coord({ render_data_texture.size[0], render_data_texture.size[1] });
    u32 texture_index = id;

    RenderDataTextureValue texture_value = render_data_texture.texture[texture_index];

    float blended_steps = 0.0f;

    if (compression_enabled && true) {
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
    float geo_powered_step_fac = pow(geo_step_fac, 3.0);
    float powered_step_fac = pow(step_fac, 3.0);

    if (texture_value.outcome == Collision) {
        color = from_array(texture_value.color) * texture_value.light *
                (1.0f + 5.0f * geo_powered_step_fac * float(globals.use_step_glow_on_foreground)) +
                vec3(texture_value.depth * 0.00001f);
        // color = from_array(texture_value.color);
        // color = vec3(texture_value.depth * 0.00005f);
    } else if (texture_value.outcome == DepthLimit) {
        color = vec3(vec3(0.2f, 0.4f, 1.0f) * 3.0f * 0.0f + texture_value.steps * 0.01f);
        color = vec3(0.0f) + 1.0f * powered_step_fac * float(globals.use_step_glow_on_background);
    }

    color = hdr_map_aces_tone(max(color, 0.0f));
    // color = vec3(texture_value.depth * 0.001f);

    unsigned int rgba = ((unsigned int) (255.0f * color.x) & 0xff) |
                        (((unsigned int) (255.0f * color.y) & 0xff) << 8) |
                        (((unsigned int) (255.0f * color.z) & 0xff) << 16) |
        ((unsigned int) 255 << 24);

    render_texture.texture[index_2d(texture_coord, render_texture)] = rgba;
}
