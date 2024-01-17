#pragma once

#include "./libraries/glm/glm.hpp"

using namespace glm;

#define PI 3.14159265358979323846264338327950288f
#define PI_HALF 1.5707963267948966f

__device__ float minimum(const vec3 p) {
    return min(min(p.x, p.y), p.z);
}

__device__ float minimum(const vec2 p) {
    return min(p.x, p.y);
}

__device__ float maximum(const vec3 p) {
    return max(max(p.x, p.y), p.z);
}

__device__ float maximum(const vec2 p) {
    return max(p.x, p.y);
}

__device__ vec3 from_array(const float p[3]) {
    return vec3(p[0], p[1], p[2]);
}
