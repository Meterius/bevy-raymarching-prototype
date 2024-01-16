#pragma once

#include "../includes/libraries/glm/glm.hpp"

using namespace glm;


// signed-distance scene

__device__ float wrap(float x, float lower, float higher) {
    return lower + glm::mod(x - lower, higher - lower);
}

// fractals

#define POWER 8.0f

__forceinline__ __device__ float sd_mandelbulb(vec3 p, float time) {
    vec3 z = p;
    float dr = 1.0f;
    float r;

    float power = POWER * (1.0f + time * 0.001f);

    for (int i = 0; i < 20; i++) {
        r = length(z);
        if (r > 2.0f) {
            break;
        }

        float theta = acos(z.z / r) * power;
        float phi = atan2(z.y, z.x) * power;
        float zr = pow(r, power);
        dr = pow(r, power - 1) * power * dr + 1;

        z.x = fma(zr, sin(theta) * cos(phi), p.x);
        z.y = fma(zr, sin(phi) * sin(theta), p.y);
        z.z = fma(zr, cos(theta), p.z);
    }

    return 0.5f * log(r) * r / dr;
}

// scene

__forceinline__ __device__ float sd_scene(vec3 p, float time) {
    //p.x = wrap(p.x, -10.0f, 10.0f);
    //return length(p) - 1.0f;

    return sd_mandelbulb(p / 20.0f - vec3(0.0, 0.0, -1.5f), time) * 20.0f;
}
