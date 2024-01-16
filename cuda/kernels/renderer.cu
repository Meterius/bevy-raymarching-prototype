#include "../includes/bindings.h"
#include "../includes/libraries/glm/glm.hpp"
#include "./signed_distance.cu"
#include "./ray_marching.cu"

using namespace glm;

// coordinate system conversion

__device__ vec2 texture_to_ndc(uvec2 p, vec2 texture_size) {
    return ((vec2) p + vec2(0.5f, 0.5f)) / texture_size;
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

// ray-marching

__forceinline__ __device__ vec3 render_ray(Ray ray, float time, DepthTextureEntry starting) {
    RayMarchHit hit = ray_march(ray, time, starting);
    return vec3(hit.depth * 0.001f, f32(hit.outcome == StepLimit), (float) hit.steps * 0.001f);
}


extern "C" __global__ void render_depth(char *render_texture, DepthTexture depth_texture, GlobalsBuffer globals, CameraBuffer camera) {
    u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    uvec2 depth_texture_coord = uvec2(id % depth_texture.size[0], id / depth_texture.size[0]);
    vec2 ndc_coord = texture_to_ndc(depth_texture_coord, {depth_texture.size[0], depth_texture.size[1] });
    vec2 cam_coord = ndc_to_camera(ndc_coord, { depth_texture.size[0], depth_texture.size[1] });
    Ray ray { { camera.position[0], camera.position[1], camera.position[2] }, camera_to_ray(cam_coord, camera) };

    float aspect_ratio = (float) globals.render_texture_size[0] / (float) globals.render_texture_size[1];
    float fov_fac = tan(camera.fov / 2);
    float cone_radius = length(vec2(
            (2.0f * aspect_ratio * fov_fac) / (float) depth_texture.size[0],
            (2.0f * fov_fac) / (float) depth_texture.size[1]
    ));

    RayMarchHit hit = cone_march(ray, cone_radius, globals.time, DepthTextureEntry { 0.0, 0 });
    depth_texture.texture[id] = DepthTextureEntry { hit.depth, hit.steps };
}

extern "C" __global__ void render(char *render_texture, DepthTexture depth_texture, GlobalsBuffer globals, CameraBuffer camera)
{
    u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    uvec2 texture_coord = uvec2(id % globals.render_texture_size[0], id / globals.render_texture_size[0]);
    vec2 ndc_coord = texture_to_ndc(texture_coord, {globals.render_texture_size[0], globals.render_texture_size[1] });
    vec2 cam_coord = ndc_to_camera(ndc_coord, { globals.render_texture_size[0], globals.render_texture_size[1] });
    Ray ray { { camera.position[0], camera.position[1], camera.position[2] }, camera_to_ray(cam_coord, camera) };

    DepthTextureEntry entry = depth_texture.texture[texture_coord.x / 8 + depth_texture.size[0] * (texture_coord.y / 8)];

    vec3 color = render_ray(ray, globals.time, entry);
    vec3 mapped_color = clamp(color, 0.0f, 1.0f);

    unsigned int rgba = ((unsigned int)(255.0f * mapped_color.x) & 0xff) |
                        (((unsigned int)(255.0f * mapped_color.y) & 0xff) << 8) |
                        (((unsigned int)(255.0f * mapped_color.z) & 0xff) << 16) |
                        (255 << 24);
    ((unsigned int*)render_texture)[id] = rgba;

    /*auto intensity = (unsigned int) (
        255.0 * clamp(depth_texture.texture[texture_coord.x / 16 + depth_texture.size[0] * (texture_coord.y / 16)] / 100.0, 0.0, 1.0)
    );
    unsigned int rgba = (intensity) | (intensity << 8) | (intensity << 16) | (255 << 24);
    ((unsigned int*)render_texture)[id] = rgba;*/
}
