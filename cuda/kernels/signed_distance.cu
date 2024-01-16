#pragma once

#include "../includes/libraries/glm/glm.hpp"
#include "../includes/utils.h"


// signed-distance scene

__device__ float wrap(float x, float lower, float higher) {
    return lower + glm::mod(x - lower, higher - lower);
}

__device__ vec3 wrap(vec3 p, vec3 lower, vec3 higher) {
    return {
            wrap(p.x, lower.x, higher.x),
            wrap(p.y, lower.y, higher.y),
            wrap(p.z, lower.z, higher.z)
    };
}

// fractals

#define POWER 8.0f

__forceinline__ __device__ float sd_mandelbulb(glm::vec3 p, float time) {
    glm::vec3 z = p;
    float dr = 1.0f;
    float r;

    float power = POWER * (1.0f + time * 0.001f);

    for (int i = 0; i < 20; i++) {
        r = length(z);
        if (r > 4.0f) {
            break;
        }

        float theta = acos(z.z / r) * power;
        float phi = atan2(z.y, z.x) * power;
        float zr = pow(r, power);
        dr = pow(r, power - 1) * power * dr + 1;

        float s_theta = sin(theta);
        z.x = glm::fma(zr, s_theta * cos(phi), p.x);
        z.y = glm::fma(zr, sin(phi) * s_theta, p.y);
        z.z = glm::fma(zr, cos(theta), p.z);
    }

    return 0.5f * log(r) * r / dr;
}

// scene

__forceinline__ __device__ float sd_axes(glm::vec3 p) {
    p.x = wrap(p.x, -0.5f, 0.5f);
    p.z = wrap(p.z, -0.5f, 0.5f);
    return length(p) - 0.05;
}

__forceinline__ __device__ float sd_scene(glm::vec3 p, float time) {
    vec3 q = p;
    q.x = wrap(q.x, -250.0f, 250.0f);
    q.z = wrap(q.z, -250.0f, 250.0f);
    // float axes = sd_axes(p);
    return min(p.y + 0.5f, sd_mandelbulb(q / 200.0f, time) * 200.0f);
}
