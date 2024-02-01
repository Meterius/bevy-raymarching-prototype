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
    return p.y;
}

__device__ const float TR_MAX_DIST = 1000.0f;
__device__ const float TR_COLL_DIST = 0.001f;

__device__ RayMarchHit ray_march_terrain(const Ray ray) {
    RayMarchHit hit {
        0,
        vec3(ray.origin),
        0.0f,
        RayMarchHitOutcome::StepLimit,
        0
    };

    for (; hit.steps < 100; hit.steps++) {
        float dist = sd_terrain(hit.position);

        if (dist < TR_COLL_DIST) {
            hit.outcome = RayMarchHitOutcome::Collision;
            break;
        }

        hit.position += dist * ray.direction;
        hit.depth += dist;

        if (hit.depth > TR_MAX_DIST) {
            float overshoot = TR_MAX_DIST - hit.depth;
            hit.position -= overshoot * ray.direction;
            hit.depth -= overshoot;
            hit.outcome = RayMarchHitOutcome::DepthLimit;
            break;
        }
    }

    return hit;
}

__device__ RayRender terrain_render_ray(
    const Ray ray,
    const float cone_radius_at_unit
) {
    RayMarchHit hit = ray_march_terrain(ray);

    vec3 color;

    if (hit.outcome == RayMarchHitOutcome::DepthLimit) {
        color = vec3(0.4f, 0.3f, 0.8f);
    } else if (hit.outcome == RayMarchHitOutcome::Collision) {
        color = vec3(0.2f, 0.3f, 0.1f) + vec3(1.0f) * (hit.depth / TR_MAX_DIST);
    } else {
        color = vec3(0.7f, 0.1f, 0.1f);
    }

    color = clamp(color, 0.0f, 1.0f);

    return RayRender {
        {}, color, 0.0f
    };
}
