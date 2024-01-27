#pragma once

#include "./libraries/glm/glm.hpp"
#include "../includes/bindings.h"

using namespace glm;

struct RayMarchHit {
int steps;
vec3 position;
float depth;
RayMarchHitOutcome outcome;
};

struct Ray {
vec3 position;
vec3 direction;
};

__device__ void __shfl_sync_ray(const uint32_t mask, const Ray &value, const int offset, Ray &output) {
    output.position[0] = __shfl_sync(mask, value.position[0], offset);
    output.position[1] = __shfl_sync(mask, value.position[1], offset);
    output.position[2] = __shfl_sync(mask, value.position[2], offset);
    output.direction[0] = __shfl_sync(mask, value.direction[0], offset);
    output.direction[1] = __shfl_sync(mask, value.direction[1], offset);
    output.direction[2] = __shfl_sync(mask, value.direction[2], offset);
}

struct RayRender {
    struct RayMarchHit hit;
    vec3 color;
    float light;
};

struct RenderSurfaceData {
    vec3 color;
};
