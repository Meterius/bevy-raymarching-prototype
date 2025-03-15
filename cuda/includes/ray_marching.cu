#pragma once

#include "../includes/libraries/glm/glm.hpp"
#include "../includes/bindings.h"
#include "../includes/types.cu"
#include "../includes/signed_distance.cu"

using namespace glm;

#define RAY_MARCH_STEP_LIMIT 256
#define RAY_MARCH_DEPTH_LIMIT 500.0f
#define RAY_MARCH_COLLISION_DISTANCE 0.001f

#define RAY_MARCH_AO_STEP_COUNT 10
#define RAY_MARCH_AO_STEP_LIMIT 1.0f

#define RAY_MARCH_SS_MIN_T 0.025f
#define RAY_MARCH_SS_MAX_T 50.0f
#define RAY_MARCH_SS_W 0.1f

__device__ float ray_march_softshadow(const SignedDistanceScene& scene, const Ray ray) {
    float res = 1.0f;
    float t = RAY_MARCH_SS_MIN_T;

    for(int i = 0; i < RAY_MARCH_STEP_LIMIT && t < RAY_MARCH_SS_MAX_T; i++) {
        float h = scene.distance(ray.origin + vec3(t) * ray.direction);
        res = min( res, h/(RAY_MARCH_SS_W*t) );
        t += clamp(h, 0.005f, 0.50f);
        if( res < -1.0f) break;
    }

    res = max(res,-1.0f);
    return clamp(0.25f*(1.0f+res)*(1.0f+res)*(2.0f-res), 0.0f, 1.0f);
}

__device__ float ray_march_ambient_occlusion(
    const SignedDistanceScene& scene,
    const Ray ray
) {
    const auto occlusion = [&](float d) {
        return abs(max(0.0f, scene.distance(ray.origin + ray.direction * d)) - d) / d;
    };

    const auto falloff = [](float d) {
        return 1.0f;
    };

    float limit = RAY_MARCH_AO_STEP_LIMIT;
    for (int i = 0; i < 3; i++) {
        if (scene.distance(ray.origin + ray.direction * limit / 2.0f) <= RAY_MARCH_COLLISION_DISTANCE) {
            limit /= 2.0f;
        }
    }

    float size = limit / float(RAY_MARCH_AO_STEP_COUNT);

    float total_occlusion = 0.0f;
    float total_falloff = 0.0f;

    float prev_occlusion = 0.0f;
    float prev_falloff = falloff(size);

    for (int i = 0; i < RAY_MARCH_AO_STEP_COUNT; i++) {
        const float d = float(i + 1) * size;
        const float next_occlusion = occlusion(d);
        const float occlusion_growth = (next_occlusion - prev_occlusion) / size;
        const float next_falloff = falloff(d);
        const float falloff_growth = (next_falloff - prev_falloff) / size;

        total_occlusion += size * (prev_falloff * prev_occlusion + size * ((falloff_growth * prev_occlusion + prev_falloff * occlusion_growth) / 2.0f + size * (falloff_growth * occlusion_growth) / 3.0f));
        total_falloff += size * (prev_falloff + size * falloff_growth / 2.0f);

        prev_occlusion = next_occlusion;
        prev_falloff = next_falloff;
    }

    if (limit != RAY_MARCH_AO_STEP_LIMIT) {
        const float size_rem = RAY_MARCH_AO_STEP_LIMIT - limit;

        const float next_occlusion = 1.0f;
        const float next_falloff = falloff(RAY_MARCH_AO_STEP_COUNT);
        const float falloff_growth = (next_falloff - prev_falloff) / size_rem;

        total_occlusion += size_rem * (prev_falloff * next_occlusion + size_rem * (falloff_growth * next_occlusion) / 2.0f);
        total_falloff += size_rem * (prev_falloff + size_rem * falloff_growth / 2.0f);
    }

    return clamp(1.0f - pow(total_occlusion / total_falloff, 1.25f), 0.0f, 1.0f);
}

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
        hit.depth += d;
        hit.position += d * ray.direction;

        if (d <= RAY_MARCH_COLLISION_DISTANCE) {
            hit.outcome = Collision;
            break;
        }

        if (hit.depth > RAY_MARCH_DEPTH_LIMIT) {
            hit.outcome = DepthLimit;
            break;
        }
    }

    hit.cycles = clock64() - hit.cycles;

    return hit;
}
