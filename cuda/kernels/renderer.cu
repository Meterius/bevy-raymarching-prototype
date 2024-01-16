#include "../includes/bindings.h"
#include "../includes/libraries/glm/glm.hpp"

using namespace glm;

__device__ CameraBuffer CAMERA;
__device__ GlobalsBuffer GLOBALS;

__device__ float minimum(vec3 p) {
    return min(min(p.x, p.y), p.z);
}

__device__ float minimum(vec2 p) {
    return min(p.x, p.y);
}

__device__ float maximum(vec3 p) {
    return max(max(p.x, p.y), p.z);
}

__device__ float maximum(vec2 p) {
    return max(p.x, p.y);
}

// coordinate system conversion

__device__ vec2 texture_to_ndc(uvec2 p) {
    return ((vec2) p + vec2(0.5f, 0.5f)) / vec2(GLOBALS.render_texture_size[0], GLOBALS.render_texture_size[1]);
}

__device__ vec2 ndc_to_camera(vec2 p) {
    return {(2 * p.x - 1) * (GLOBALS.render_screen_size[0] / GLOBALS.render_screen_size[1]), 1 - 2 * p.y};
}

__device__ vec3 camera_to_ray(vec2 p) {
    return normalize(
        vec3(CAMERA.forward[0], CAMERA.forward[1], CAMERA.forward[2])
        + p.y * tan(CAMERA.fov / 2) * vec3(CAMERA.up[0], CAMERA.up[1], CAMERA.up[2])
        + p.x * tan(CAMERA.fov / 2) * vec3(CAMERA.right[0], CAMERA.right[1], CAMERA.right[2])
    );
}

__device__ u32 texture_to_index(uvec2 p) {
    return 4 * (GLOBALS.render_texture_size[0] * p.y + p.x);
}

// signed-distance scene

__device__ float wrap(float x, float lower, float higher) {
    return lower + glm::mod(x - lower, higher - lower);
}

__device__ float sd_cube(vec3 p) {
    return maximum(glm::abs(p)) - 0.5f;
}

__device__ float sd_scene(vec3 p) {
    p.x = wrap(p.x, -5.0f, 5.0f);
    float sd = sd_cube(p - vec3(0, 0.5f, 0));

    for (int i = 0; i < 1000; i++) {
        p.y -= 2.0f;
        sd = min(sd, sd_cube(p));
    }

    return sd;
}

// ray-marching

struct RayMarchHit {
    int steps;
    vec3 position;
    float depth;
    bool collision;
};

struct Ray {
    vec3 position;
    vec3 direction;
};

__device__ RayMarchHit ray_march(Ray ray) {
    RayMarchHit hit { 0, ray.position, 0.0f, false };

    float coll_dist = 0.001f;

    for (; hit.steps < 1000; hit.steps++) {
        float d = sd_scene(hit.position);

        if (d < coll_dist) {
            hit.collision = true;
            break;
        }

        hit.depth += d;
        hit.position += d * ray.direction;

        if (hit.depth > 10000) {
            break;
        }
    }

    return hit;
}

__device__ vec3 render_ray(Ray ray) {
    RayMarchHit hit = ray_march(ray);
    return vec3(hit.depth * 0.001f, 0.0f, (float) hit.steps / 100.0f);
}

extern "C" __global__ void render(char *render_texture, GlobalsBuffer globals, CameraBuffer camera)
{
    GLOBALS = globals;
    CAMERA = camera;

    uvec2 texture_coord = uvec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    vec2 ndc_coord = texture_to_ndc(texture_coord);
    vec2 cam_coord = ndc_to_camera(ndc_coord);
    Ray ray {
        vec3(CAMERA.position[0], CAMERA.position[1], CAMERA.position[2]),
        camera_to_ray(cam_coord)
    };

    vec3 color = render_ray(ray);
    vec3 mapped_color = clamp(color, 0.0f, 1.0f);

    u32 texture_index = texture_to_index(texture_coord);
    render_texture[texture_index+0] = (char) (255.0 * mapped_color.x);
    render_texture[texture_index+1] = (char) (255.0 * mapped_color.y);
    render_texture[texture_index+2] = (char) (255.0 * mapped_color.z);
    render_texture[texture_index+3] = (char) 255;
}
