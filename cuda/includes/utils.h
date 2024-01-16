#pragma once

#include "./libraries/glm/glm.hpp"

using namespace glm;

#define PI 3.14159265358979323846264338327950288f
#define PI_HALF 1.5707963267948966f

float minimum(vec3 p) {
    return min(min(p.x, p.y), p.z);
}

float minimum(vec2 p) {
    return min(p.x, p.y);
}

float maximum(vec3 p) {
    return max(max(p.x, p.y), p.z);
}

float maximum(vec2 p) {
    return max(p.x, p.y);
}