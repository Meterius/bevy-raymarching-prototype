#pragma once

#include "../includes/libraries/glm/glm.hpp"
#include "./signed_distance.cu"

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

__forceinline__ __device__ RayMarchHit ray_march(Ray ray, float cone_radius, float time) {
    RayMarchHit hit { 0, ray.position, 0.0f, StepLimit };

    float coll_dist = max(cone_radius, 0.001f);

    for (; hit.steps < 1000; hit.steps++) {
        float d = sd_scene(hit.position, time);

        if (d < coll_dist) {
            hit.outcome = Collision;
            break;
        }

        float diff = d - cone_radius;
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