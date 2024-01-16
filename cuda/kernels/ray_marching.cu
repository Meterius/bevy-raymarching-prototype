#pragma once

#include "../includes/libraries/glm/glm.hpp"
#include "./signed_distance.cu"
#include "../includes/bindings.h"

using namespace glm;

enum RayMarchHitOutcome {
    Collision, StepLimit, DepthLimit
};

struct __align__(32) RayMarchHit {
    int steps;
    vec3 position;
    float depth;
    RayMarchHitOutcome outcome;
};

struct __align__(32) Ray {
    vec3 position;
    vec3 direction;
};

__forceinline__ __device__ RayMarchHit cone_march(Ray ray, float cone_radius, float time, DepthTextureEntry starting) {
    RayMarchHit hit { starting.steps, ray.position + ray.direction * starting.depth, starting.depth, StepLimit };

    for (; hit.steps < 1000; hit.steps++) {
        float curr_cone_radius = cone_radius * (1.0f + hit.depth);
        float d = sd_scene(hit.position, time);

        if (d < curr_cone_radius) {
            hit.outcome = Collision;
            break;
        }

        float diff = d;// - curr_cone_radius;
        hit.depth += diff;
        hit.position.x = fma(diff, ray.direction.x, hit.position.x);
        hit.position.y = fma(diff, ray.direction.y, hit.position.y);
        hit.position.z = fma(diff, ray.direction.z, hit.position.z);

        if (hit.depth > 10000) {
            hit.outcome = DepthLimit;
            break;
        }
    }

    return hit;
}

__forceinline__ __device__ RayMarchHit ray_march(Ray ray, float time, DepthTextureEntry starting) {
    RayMarchHit hit { 0, ray.position + ray.direction * starting.depth, starting.depth, StepLimit };

    for (; hit.steps < 1000; hit.steps++) {
        float d = sd_scene(hit.position, time);

        if (d < 0.001f) {
            hit.outcome = Collision;
            break;
        }

        hit.depth += d;
        hit.position.x = fma(d, ray.direction.x, hit.position.x);
        hit.position.y = fma(d, ray.direction.y, hit.position.y);
        hit.position.z = fma(d, ray.direction.z, hit.position.z);

        if (hit.depth > 10000) {
            hit.outcome = DepthLimit;
            break;
        }
    }

    return hit;
}