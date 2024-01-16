#pragma once

#include "../includes/libraries/glm/glm.hpp"
#include "../includes/utils.h"


// signed-distance scene

__device__ float wrap(float x, float lower, float higher) {
    return lower + glm::mod(x - lower, higher - lower);
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

__forceinline__ __device__ float sd_scene(glm::vec3 p, float time) {
    //p.x = wrap(p.x, -10.0f, 10.0f);
    //return length(p) - 1.0f;

    return sd_mandelbulb(p / 2000.0f - glm::vec3(0.0, 0.0, -1.5f), time) * 2000.0f;
}
