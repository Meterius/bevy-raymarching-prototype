#pragma once

#include "../includes/libraries/glm/glm.hpp"
#include "../includes/bindings.h"
#include "../includes/types.cu"
#include "../includes/signed_distance.cu"

using namespace glm;

#define RAY_MARCH_STEP_LIMIT 256
#define RAY_MARCH_DEPTH_LIMIT 500.0f
#define RAY_MARCH_COLLISION_DISTANCE 0.001f

__device__ RayMarchHit ray_march(
    const SignedDistanceScene& scene,
    const Ray ray
) {
    RayMarchHit hit {
        (int) 0,
        ray.origin + ray.direction,
        0,
        StepLimit,
        clock64()
    };

    for (; hit.steps < RAY_MARCH_STEP_LIMIT; hit.steps++) {
        float d = scene.distance(hit.position);

        if (d <= RAY_MARCH_COLLISION_DISTANCE) {
            hit.outcome = Collision;
            break;
        }

        hit.depth += d;
        hit.position += d * ray.direction;

        if (hit.depth > RAY_MARCH_DEPTH_LIMIT) {
            hit.outcome = DepthLimit;
            break;
        }
    }

    hit.cycles = clock64() - hit.cycles;

    return hit;
}
