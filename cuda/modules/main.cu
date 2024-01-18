#include "../includes/bindings.h"
#include "../includes/libraries/glm/glm.hpp"
#include "../includes/ray_marching.cu"
#include "../includes/rendering.cu"
#include "../includes/utils.h"

using namespace glm;

#define RELATIVIZE_STEP_COUNT false

// coordinate system conversion

__device__ vec2 texture_to_ndc(uvec2 p, vec2 texture_size) {
    return ((vec2) p + vec2(0.5f, 0.5f)) / texture_size;
}

__device__ uvec2 ndc_to_texture(vec2 p, vec2 texture_size) {
    return uvec2(round((p * texture_size) - vec2(0.5f, 0.5f)));
}

__device__ vec2 ndc_to_camera(vec2 p, vec2 render_screen_size) {
    return {(2 * p.x - 1) * (render_screen_size.x / render_screen_size.y), 1 - 2 * p.y};
}

__device__ vec3 camera_to_ray(vec2 p, CameraBuffer CAMERA) {
    float fov_fac = tan(CAMERA.fov / 2);
    return normalize(
        vec3(CAMERA.forward[0], CAMERA.forward[1], CAMERA.forward[2])
        + p.y * fov_fac * vec3(CAMERA.up[0], CAMERA.up[1], CAMERA.up[2])
        + p.x * fov_fac * vec3(CAMERA.right[0], CAMERA.right[1], CAMERA.right[2])
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

    uvec2 cm_texture_coord = uvec2(id % cm_textures.textures[level].size[0], id / cm_textures.textures[level].size[0]);
    vec2 ndc_coord = texture_to_ndc(cm_texture_coord, {cm_textures.textures[level].size[0], cm_textures.textures[level].size[1] });
    vec2 cam_coord = ndc_to_camera(ndc_coord, { cm_textures.textures[level].size[0], cm_textures.textures[level].size[1] });
    Ray ray { { camera.position[0], camera.position[1], camera.position[2] }, camera_to_ray(cam_coord, camera) };

    float aspect_ratio = (float) render_data_texture.size[0] / (float) render_data_texture.size[1];
    float fov_fac = tan(camera.fov / 2);
    float cone_radius = length(vec2(
        (2.0f * aspect_ratio * fov_fac) / (float) cm_textures.textures[level].size[0],
        (2.0f * fov_fac) / (float) cm_textures.textures[level].size[1]
    ));

    ConeMarchTextureValue entry { 0.0f, 0, Collision };
    if (level > 0) {
        uvec2 lower_cm_texture_coord = ndc_to_texture(
        ndc_coord,
        { (float) cm_textures.textures[level - 1].size[0], (float) cm_textures.textures[level - 1].size[1] }
        );

        entry = cm_textures.textures[level - 1].texture[
            lower_cm_texture_coord.x + cm_textures.textures[level - 1].size[0] * lower_cm_texture_coord.y
        ];
    }

    RayMarchHit hit = ray_march<true>(make_sd_scene(globals, camera), ray, entry, cone_radius);

    if (RELATIVIZE_STEP_COUNT) {
        float compression_factor = (float) (render_data_texture.size[0] * render_data_texture.size[1]) / (float) (cm_textures.textures[level].size[0] * cm_textures.textures[level].size[1]);
        hit.steps = (int) ceil((float) hit.steps / compression_factor);
    }

    cm_textures.textures[level].texture[id] = ConeMarchTextureValue { hit.depth, hit.steps, hit.outcome };
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
    runtime_scene.sphere_count = runtime_scene_param.sphere_count;

    int perThread = runtime_scene.sphere_count / blockDim.x;
    for (int i = 0; i < perThread; i++) {
        runtime_scene.spheres[threadIdx.x * perThread + i] = runtime_scene_param.spheres[threadIdx.x * perThread + i];
    }

    __syncthreads();

    u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    uvec2 texture_coord = uvec2(id % render_data_texture.size[0], id / render_data_texture.size[0]);
    vec2 ndc_coord = texture_to_ndc(texture_coord, { render_data_texture.size[0], render_data_texture.size[1] });
    vec2 cam_coord = ndc_to_camera(ndc_coord, { render_data_texture.size[0], render_data_texture.size[1] });
    Ray ray { { camera.position[0], camera.position[1], camera.position[2] }, camera_to_ray(cam_coord, camera) };

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
        }
    #else
        ConeMarchTextureValue entry = { 0.0f, 0, Collision };
    #endif

    RayMarchHit hit = ray_march<false>(make_sd_scene(globals, camera), ray, entry);

    render_data_texture.texture[id] = {
            hit.depth, (float) hit.steps, hit.outcome, { 1.0f, 1.0f, 1.0f }, 1.0f
    };
}

extern "C" __global__ void compute_render_finalize(
    Texture render_texture,
    RenderDataTexture render_data_texture,
    GlobalsBuffer globals
) {
    u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    uvec2 texture_coord = uvec2(id % render_data_texture.size[0], id / render_data_texture.size[0]);

    vec3 color = clamp(
            vec3(render_data_texture.texture[id].steps * 0.001f),
            0.0f, 1.0f
    );

    unsigned int rgba = ((unsigned int)(255.0f * color.x) & 0xff) |
                        (((unsigned int)(255.0f * color.y) & 0xff) << 8) |
                        (((unsigned int)(255.0f * color.z) & 0xff) << 16) |
                        (255 << 24);

    render_texture.texture[id] = rgba;
}
