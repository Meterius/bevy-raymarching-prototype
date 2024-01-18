#include "../includes/bindings.h"
#include "../includes/libraries/glm/glm.hpp"
#include "../includes/ray_marching.cu"
#include "../includes/rendering.cu"
#include "../includes/utils.h"

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

template <typename Func, typename Texture>
__device__ auto fetch_2d(ivec2 p, Texture &texture, Func map) {
    return map(
        texture.texture[
            min(max(p.x, 0), texture.size[0] - 1)
            + min(max(p.y, 0), texture.size[1] - 1) * texture.size[0]
        ]
    );
}

__device__ float cubic_interpolate(float y0, float y1, float y2, float y3, float rx1) {
    return y1 + 0.5f * rx1*(y2 - y0 + rx1*(2.0f*y0 - 5.0f*y1 + 4.0f*y2 - y3 + rx1*(3.0f*(y1 - y2) + y3 - y0)));
}

template <typename Func, typename Texture>
__device__ auto ndc_to_interpolated_value(vec2 p, Texture &texture, Func map) {
    vec2 t = (p * vec2((float) texture.size[0], (float) texture.size[1])) - vec2(0.5f, 0.5f);
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

    return cubic_interpolate(interps[0], interps[1], interps[2], interps[3], t.y - (float) tc.y);
}

__device__ vec2 ndc_to_camera(vec2 p, vec2 render_screen_size) {
    return {(2 * p.x - 1) * (render_screen_size.x / render_screen_size.y), 1 - 2 * p.y};
}

__device__ vec3 camera_to_ray(
    vec2 p,
    CameraBuffer CAMERA,
    vec2 screen_size,
    vec2 texture_size
) {
    float width_factor = (screen_size.x / texture_size.x) * (texture_size.y / screen_size.y);

    float fov_fac = tan(CAMERA.fov / 2);
    return normalize(
        vec3(CAMERA.forward[0], CAMERA.forward[1], CAMERA.forward[2])
        + p.y * fov_fac * vec3(CAMERA.up[0], CAMERA.up[1], CAMERA.up[2])
        + p.x * fov_fac * width_factor * vec3(CAMERA.right[0], CAMERA.right[1], CAMERA.right[2])
    );
}

// scene

__shared__ SdRuntimeScene runtime_scene;

__device__ auto make_sd_scene(
    GlobalsBuffer &globals,
    CameraBuffer &camera
) {
    return [globals](vec3 p){
        float sd = sd_box(p, vec3(-30.0f, 0.0f, 0.0f), vec3(1.0f, 2.0f, 10.0f));

        //p.x = wrap(p.x, -40.0f, 40.0f);
        //sd = min(sd_mandelbulb((p + vec3(0.0f, 0.0f, 15.0f)) / 20.0f, globals.time) * 20.0f, sd);

        for (int i = 0; i < runtime_scene.sphere_count; i++) {
            sd = min(
                sd,
                length(p - from_array(runtime_scene.spheres[i].translation)) - runtime_scene.spheres[i].radius
            );
        }

        return sd;
    };
}

// ray-marching

#include <cuda_runtime.h>

#ifndef DISABLE_CONE_MARCH
extern "C" __global__ void compute_compressed_depth(
     unsigned int level,
     RenderDataTexture render_data_texture,
     ConeMarchTextures cm_textures,
     GlobalsBuffer globals,
     CameraBuffer camera,
     SdRuntimeScene runtime_scene_param
) {
    runtime_scene.sphere_count = runtime_scene_param.sphere_count;

    int perThread = runtime_scene.sphere_count / blockDim.x;
    for (int i = 0; i < perThread; i++) {
        runtime_scene.spheres[threadIdx.x * perThread + i] = runtime_scene_param.spheres[threadIdx.x * perThread + i];
    }

    __syncthreads();

    u32 id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id > cm_textures.textures[level].size[0] * cm_textures.textures[level].size[1]) {
        return;
    }

    const auto texture_to_dir = [&camera, &cm_textures, &level, &globals](vec2 p) {
        vec2 ndc_coord = texture_to_ndc(p, {cm_textures.textures[level].size[0], cm_textures.textures[level].size[1] });
        vec2 cam_coord = ndc_to_camera(ndc_coord, { cm_textures.textures[level].size[0], cm_textures.textures[level].size[1] });
        return camera_to_ray(cam_coord, camera, from_array(globals.render_screen_size), vec2(globals.render_texture_size[0], globals.render_texture_size[1]));
    };

    uvec2 cm_texture_coord = uvec2(id % cm_textures.textures[level].size[0], id / cm_textures.textures[level].size[0]);
    vec2 ndc_coord = texture_to_ndc(cm_texture_coord, {cm_textures.textures[level].size[0], cm_textures.textures[level].size[1] });
    vec2 cam_coord = ndc_to_camera(ndc_coord, { cm_textures.textures[level].size[0], cm_textures.textures[level].size[1] });
    Ray ray {
        { camera.position[0], camera.position[1], camera.position[2] },
        camera_to_ray(cam_coord, camera, from_array(globals.render_screen_size), vec2(globals.render_texture_size[0], globals.render_texture_size[1]))
    };

    vec3 border_dirs[4] = {
        texture_to_dir({ (float) cm_texture_coord[0] - SQRT_INV, (float) cm_texture_coord[1] - SQRT_INV }),
        texture_to_dir({ (float) cm_texture_coord[0] - SQRT_INV, (float) cm_texture_coord[1] + SQRT_INV }),
        texture_to_dir({ (float) cm_texture_coord[0] + SQRT_INV, (float) cm_texture_coord[1] - SQRT_INV }),
        texture_to_dir({ (float) cm_texture_coord[0] + SQRT_INV, (float) cm_texture_coord[1] + SQRT_INV }),
    };

    float cone_radius_at_unit = max(
        max(length(ray.direction - border_dirs[0]), length(ray.direction - border_dirs[1])),
        max(length(ray.direction - border_dirs[2]), length(ray.direction - border_dirs[3]))
    );

    ConeMarchTextureValue entry { 0.0f, 0, Collision };
    if (level > 0) {
        uvec2 lower_cm_texture_coord = ndc_to_texture(
        ndc_coord,
        { (float) cm_textures.textures[level - 1].size[0], (float) cm_textures.textures[level - 1].size[1] }
        );

        entry = cm_textures.textures[level - 1].texture[
            lower_cm_texture_coord.x + cm_textures.textures[level - 1].size[0] * lower_cm_texture_coord.y
        ];

        if (COMPRESSION_STEP_INTERPOLATION) {
            entry.steps = ndc_to_interpolated_value(
                ndc_coord, cm_textures.textures[level - 1], [](ConeMarchTextureValue entry) { return entry.steps; }
            );
        }
    }

    RayMarchHit hit = ray_march<true>(make_sd_scene(globals, camera), ray, entry, cone_radius_at_unit);

    if (RELATIVIZE_STEP_COUNT) {
        float compression_factor = (float) (render_data_texture.size[0] * render_data_texture.size[1]) / (float) (cm_textures.textures[level].size[0] * cm_textures.textures[level].size[1]);
        hit.steps = (int) ceil((float) hit.steps / compression_factor);
    }

    cm_textures.textures[level].texture[id] = ConeMarchTextureValue { hit.depth, (float) hit.steps + entry.steps, hit.outcome };

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

extern "C" __global__ void compute_render(
    RenderDataTexture render_data_texture,
    ConeMarchTextures cm_textures,
    GlobalsBuffer globals,
    CameraBuffer camera,
    SdRuntimeScene runtime_scene_param,
    bool compression_enabled
) {
    // store runtime scene data in shared memory

    runtime_scene.sphere_count = runtime_scene_param.sphere_count;

    int perThread = runtime_scene.sphere_count / blockDim.x;
    for (int i = 0; i < perThread; i++) {
        runtime_scene.spheres[threadIdx.x * perThread + i] = runtime_scene_param.spheres[threadIdx.x * perThread + i];
    }

    __syncthreads();

    // calculate ray

    u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    uvec2 texture_coord = uvec2(id % render_data_texture.size[0], id / render_data_texture.size[0]);
    vec2 ndc_coord = texture_to_ndc(texture_coord, { render_data_texture.size[0], render_data_texture.size[1] });
    vec2 cam_coord = ndc_to_camera(ndc_coord, { render_data_texture.size[0], render_data_texture.size[1] });
    Ray ray {
        { camera.position[0], camera.position[1], camera.position[2] },
        camera_to_ray(cam_coord, camera, from_array(globals.render_screen_size), vec2(globals.render_texture_size[0], globals.render_texture_size[1]))
    };

    // if enabled, fetch cone march compression starting point

    #ifndef DISABLE_CONE_MARCH
        ConeMarchTextureValue entry = { 0.0f, 0, Collision };

        if (compression_enabled) {
            uvec2 cm_texture_coord = ndc_to_texture(
                ndc_coord,
                {(float) cm_textures.textures[CONE_MARCH_LEVELS - 1].size[0],
                 (float) cm_textures.textures[CONE_MARCH_LEVELS - 1].size[1]}
            );

            entry = cm_textures.textures[CONE_MARCH_LEVELS - 1].texture[
                cm_texture_coord.x + cm_textures.textures[CONE_MARCH_LEVELS - 1].size[0] * cm_texture_coord.y
            ];

            if (COMPRESSION_STEP_INTERPOLATION) {
                entry.steps = ndc_to_interpolated_value(
                    ndc_coord, cm_textures.textures[CONE_MARCH_LEVELS - 1],
                    [](ConeMarchTextureValue entry) { return entry.steps; }
                );
            }
        }
    #else
        float interpolated_cm_steps = 0.0f;
        ConeMarchTextureValue entry = { 0.0f, 0, Collision };
    #endif

    // ray march and fill preliminary values in render data texture

    RayMarchHit hit = ray_march<false>(make_sd_scene(globals, camera), ray, entry);

    render_data_texture.texture[id] = {
        hit.depth, (float) hit.steps + entry.steps, hit.outcome, { 1.0f, 1.0f, 1.0f }, 1.0f
    };
}

#define STEP_GAUSSIAN_SIZE 6
#define STEP_GAUSSIAN_DEV 12.0f

__device__ float step_gaussian_value(int i, int j) {
    return exp(
        -((float) (i * i + j * j) / (2.0f * STEP_GAUSSIAN_DEV * STEP_GAUSSIAN_DEV)))
        / (2.0f * PI * STEP_GAUSSIAN_DEV * STEP_GAUSSIAN_DEV
    );
}

extern "C" __global__ void compute_render_finalize(
    Texture render_texture,
    RenderDataTexture render_data_texture,
    GlobalsBuffer globals,
    bool compression_enabled
) {
    u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    ivec2 texture_coord = ivec2(id % render_data_texture.size[0], id / render_data_texture.size[0]);

    float blended_steps = 0.0f;

    if (compression_enabled && false) {
        float total = 0.0f;

        for (int i = -STEP_GAUSSIAN_SIZE; i <= STEP_GAUSSIAN_SIZE; i++) {
            for (int j = -STEP_GAUSSIAN_SIZE; j <= STEP_GAUSSIAN_SIZE; j++) {
                int px = texture_coord.x + i;
                int py = texture_coord.y + j;

                if (0 <= px && px <= render_texture.size[0] && 0 <= py && py <= render_texture.size[1]) {
                    float fac;

                    if (
                        render_data_texture.texture[id].outcome == Collision
                        && render_data_texture.texture[render_data_texture.size[0] * py + px].outcome == Collision
                    ) {
                        fac = clamp(1.0f - 4.0f * abs(
                                render_data_texture.texture[id].depth -
                                render_data_texture.texture[render_data_texture.size[0] * py + px].depth
                        ), 0.0f, 1.0f);
                    } else if (
                        render_data_texture.texture[id].outcome == DepthLimit
                        && render_data_texture.texture[render_data_texture.size[0] * py + px].outcome == DepthLimit
                    ) {
                        fac = 1.0f;
                    } else {
                        fac = 0.0f;
                    }

                    float val = step_gaussian_value(i, j) * fac;
                    total += val;
                    blended_steps += val * render_data_texture.texture[render_data_texture.size[0] * py + px].steps;
                }
            }
        }

        blended_steps /= total;
    } else {
        blended_steps = render_data_texture.texture[id].steps;
    }

    vec3 color = clamp(
        vec3(blended_steps * 0.001f),
        0.0f, 1.0f
    );

    unsigned int rgba = ((unsigned int)(255.0f * color.x) & 0xff) |
                        (((unsigned int)(255.0f * color.y) & 0xff) << 8) |
                        (((unsigned int)(255.0f * color.z) & 0xff) << 16) |
                        (255 << 24);

    render_texture.texture[id] = rgba;
}
