#pragma once

#include "../includes/libraries/glm/glm.hpp"
#include "../includes/bindings.h"
#include "../includes/types.cu"
#include "../includes/signed_distance.cu"

using namespace glm;

#define RAY_MARCH_STEP_LIMIT 256
#define RAY_MARCH_DEPTH_LIMIT 500.0f
#define RAY_MARCH_COLLISION_DISTANCE 0.001f

template<typename SdSceneFunc>
__device__ RayMarchHit ray_march(
    const SdSceneFunc sdi_scene,
    const Ray ray,
    const ConeMarchTextureValue starting = ConeMarchTextureValue {},
    const float cone_radius_at_unit = 0.0f
) {
    RayMarchHit hit {
        (int) 0,
        ray.position + ray.direction * starting.depth,
        starting.depth,
        StepLimit,
        clock64()
    };

    if (starting.outcome == DepthLimit) {
        hit.outcome = starting.outcome;
    } else {
        for (; hit.steps < RAY_MARCH_STEP_LIMIT; hit.steps++) {
            float collision_distance = cone_radius_at_unit * hit.depth;
            float d = sdi_scene(
                SdInvocation<SdInvocationType::ConeType> { Ray { hit.position, ray.direction }, collision_distance }
            );

            if (d <= collision_distance + RAY_MARCH_COLLISION_DISTANCE) {
                hit.outcome = Collision;
                break;
            }

            hit.depth += (d - collision_distance);
            hit.position += (d - collision_distance) * ray.direction;

            if (hit.depth > RAY_MARCH_DEPTH_LIMIT) {
                hit.outcome = DepthLimit;
                break;
            }
        }
    }

    hit.cycles = clock64() - hit.cycles;

    return hit;
}

template<typename SdSceneFunc>
__device__ float soft_shadow_ray_march(
    const SdSceneFunc sdi_scene,
    const Ray ray,
    const float w
) {
    float res = 1.0f;
    float ph = 1e20f;
    float depth = 0.0f;

    for (int i = 0; i < RAY_MARCH_STEP_LIMIT; i++) {
        float sd = sdi_scene(
            SdInvocation<SdInvocationType::RayType> { ray.position + ray.direction * depth, ray.direction }
        );

        if (sd <= RAY_MARCH_COLLISION_DISTANCE) {
            return 0.0f;
        }

        float y = sd * sd / (2.0f * ph);
        float d = sqrt(sd * sd - y * y);
        res = min(res, d / (w * max(0.0f, depth - y)));
        ph = sd;
        depth += sd;

        if (depth > RAY_MARCH_DEPTH_LIMIT) {
            break;
        }
    }

    return res;
}
