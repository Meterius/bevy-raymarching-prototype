#pragma once

#include "./libraries/glm/glm.hpp"
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

struct RayRender {
    struct RayMarchHit hit;
    vec3 color;
    float light;
};

struct RenderSurfaceData {
    vec3 color;
};