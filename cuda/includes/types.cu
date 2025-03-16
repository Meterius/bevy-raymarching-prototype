#pragma once

#include "./libraries/glm/glm.hpp"
#include "../includes/bindings.h"

using namespace glm;

struct __align__(32) RayMarchHit {
    int steps;
    vec3 position;
    float depth;
    RayMarchHitOutcome outcome;
    long long cycles;
};

struct __align__(32) Ray {
    vec3 origin;
    vec3 direction;
};

struct __align__(32) Line {
    vec3 origin;
    vec3 direction;
    float distance;

    __device__ Ray as_ray() const {
        return Ray { origin, direction };
    }
};

struct RayRender {
    struct RayMarchHit hit;
    vec3 color;
};

struct Material {
    vec3 color;
    float roughness;
};
