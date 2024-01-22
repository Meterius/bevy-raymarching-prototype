#pragma once

#include "../includes/libraries/glm/glm.hpp"
#include "../includes/bindings.h"
#include "../includes/types.cu"

using namespace glm;

#define RAY_MARCH_STEP_LIMIT 500
#define RAY_MARCH_DEPTH_LIMIT 1000.0f
#define RAY_MARCH_COLLISION_DISTANCE 0.001f

__shared__ float collision_distance[BLOCK_SIZE];

template<bool useConeMarch, typename SdSceneFunc>
__device__ RayMarchHit ray_march(
    SdSceneFunc sd_scene,
    Ray ray,
    ConeMarchTextureValue starting = ConeMarchTextureValue {},
    float cone_radius_at_unit = 0.0
) {
    RayMarchHit hit {
        (int) starting.steps,
        ray.position + ray.direction * starting.depth,
        starting.depth,
        StepLimit
    };

    collision_distance[threadIdx.x] = RAY_MARCH_COLLISION_DISTANCE;

    if (starting.outcome == DepthLimit) {
        hit.outcome = starting.outcome;
    } else {
        for (; hit.steps < RAY_MARCH_STEP_LIMIT; hit.steps++) {
            collision_distance[threadIdx.x] = cone_radius_at_unit * hit.depth;

            float d = sd_scene(hit.position);

            if (useConeMarch) {
                if (d < cone_radius_at_unit * hit.depth) {
                    hit.outcome = Collision;
                    break;
                }
            } else {
                if (d < RAY_MARCH_COLLISION_DISTANCE) {
                    hit.outcome = Collision;
                    break;
                }
            }

            float diff = d;

            if (useConeMarch) {
                hit.depth += diff - cone_radius_at_unit * hit.depth;
            } else {
                hit.depth += diff;
            }

            hit.position += diff * ray.direction;

            if (hit.depth > RAY_MARCH_DEPTH_LIMIT) {
                hit.outcome = DepthLimit;
                break;
            }
        }
    }

    hit.steps -= (int) starting.steps;

    return hit;
}

template<typename SdSceneFunc>
__device__ float soft_shadow_ray_march(
    SdSceneFunc sd_scene,
    Ray ray,
    float w
) {
    float res = 1.0f;
    float ph = 1e20f;
    float depth = 0.0f;

    for (int i = 0; i < RAY_MARCH_STEP_LIMIT; i++) {
        float sd = sd_scene(ray.position + ray.direction * depth);

        if (sd <= RAY_MARCH_COLLISION_DISTANCE) {
            return 0.0;
        }

        float y = sd * sd / (2.0 * ph);
        float d = sqrt(sd * sd - y * y);
        res = min(res, d / (w * max(0.0, depth - y)));
        ph = sd;
        depth += sd;

        if (depth > RAY_MARCH_DEPTH_LIMIT) {
            break;
        }
    }

    return res;
}
