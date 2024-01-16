#pragma once

#include "../includes/libraries/glm/glm.hpp"
#include "./signed_distance.cu"
#include "../includes/bindings.h"

using namespace glm;

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

__forceinline__ __device__ RayMarchHit cone_march(Ray ray, float cone_radius, float time, ConeMarchTextureValue starting) {
    RayMarchHit hit { 0, ray.position + ray.direction * starting.depth, starting.depth, StepLimit };

    if (starting.outcome == DepthLimit) {
        hit.outcome = starting.outcome;
        return hit;
    }

    for (; hit.steps < 500; hit.steps++) {
        float curr_cone_radius = cone_radius * (1.0f + hit.depth);
        float d = sd_scene(hit.position, time);

        if (d < curr_cone_radius) {
            hit.outcome = Collision;
            break;
        }

        float diff = d;// - curr_cone_radius;
        hit.depth += diff;
        hit.position.x = glm::fma(diff, ray.direction.x, hit.position.x);
        hit.position.y = glm::fma(diff, ray.direction.y, hit.position.y);
        hit.position.z = glm::fma(diff, ray.direction.z, hit.position.z);

        if (hit.depth > 100000) {
            hit.outcome = DepthLimit;
            break;
        }
    }

    hit.steps += starting.steps;
    return hit;
}

__forceinline__ __device__ RayMarchHit ray_march(Ray ray, float time, ConeMarchTextureValue starting) {
    RayMarchHit hit { 0, ray.position + ray.direction * starting.depth, starting.depth, StepLimit };

    if (starting.outcome == DepthLimit) {
        hit.outcome = starting.outcome;
        return hit;
    }

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

        if (hit.depth > 100000) {
            hit.outcome = DepthLimit;
            break;
        }
    }

    hit.steps += starting.steps;
    return hit;
}