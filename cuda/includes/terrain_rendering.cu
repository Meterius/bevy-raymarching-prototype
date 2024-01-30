#include "../includes/libraries/glm/glm.hpp"
#include "../includes/bindings.h"
#include "../includes/color.cu"
#include "../includes/utils.cu"
#include "../includes/signed_distance.cu"
#include "../includes/rendering.cu"
#include "../includes/ray_marching.cu"

__device__ float smooth_step(const float x, const float a, const float b) {
    const float l = min(1.0f, max(0.0f, (x - a) / (b - a)));
    return 3.0f * pow(l, 2.0f) - 2.0f * pow(l, 3.0f);
}

__device__ float smoothed_linear_unit_tile(const vec2 p, const float a, const float b, const float c, const float d) {
    return a + (b - a) * smooth_step(p.x, 0.0f, 1.0f)
           + (c - a) * smooth_step(p.y, 0.0f, 1.0f)
           + (a - b - c + d) * smooth_step(p.x, 0.0f, 1.0f) * smooth_step(p.y, 0.0f, 1.0f);
}

__device__ float sd_terrain(const vec3 p) {
    return p.y - 10.0f * smoothed_linear_unit_tile(p.xz * 0.01f, -1.0f, 1.0f, 2.0f, 1.25f);
}

__device__ float ray_march_terrain(const Ray ray) {

}

__device__ RayRender terrain_render_ray(
    const Ray ray,
    const float cone_radius_at_unit
) {
    return RayRender {
        {}, vec3(0.25f), 0.0f
    };
}
