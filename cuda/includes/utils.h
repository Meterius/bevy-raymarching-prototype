#pragma once

#include "./libraries/glm/glm.hpp"

using namespace glm;

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