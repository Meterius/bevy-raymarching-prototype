#pragma once

#include "../includes/libraries/glm/glm.hpp"
#include "../includes/bindings.h"
#include "../includes/types.cu"

using namespace glm;

#define RAY_MARCH_STEP_LIMIT 500
#define RAY_MARCH_DEPTH_LIMIT 1000.0f
#define RAY_MARCH_COLLISION_DISTANCE 0.001f

template<bool useConeMarch, typename SdSceneFunc>
__device__ RayMarchHit ray_march(
    SdSceneFunc sd_scene,
    Ray ray,
    ConeMarchTextureValue starting = ConeMarchTextureValue {},
    float cone_radius_at_unit = 0.0
) {
    RayMarchHit hit {
        0,
        ray.position + ray.direction * starting.depth,
        starting.depth,
        StepLimit
    };

    if (starting.outcome == DepthLimit) {
        hit.outcome = starting.outcome;
    } else {
        for (; hit.steps < RAY_MARCH_STEP_LIMIT; hit.steps++) {
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

            hit.position.x = glm::fma(diff, ray.direction.x, hit.position.x);
            hit.position.y = glm::fma(diff, ray.direction.y, hit.position.y);
            hit.position.z = glm::fma(diff, ray.direction.z, hit.position.z);

            if (hit.depth > RAY_MARCH_DEPTH_LIMIT) {
                hit.outcome = DepthLimit;
                break;
            }
        }
    }

    return hit;
}
