#pragma once

#include "../includes/libraries/glm/glm.hpp"
#include "../includes/rendering.cu"
#include "../includes/utils.h"

using namespace glm;

#define SURFACE_DISTANCE 0.001f

__device__ float wrap(float x, float lower, float higher) {
    return lower + glm::mod(x - lower, higher - lower);
}

__device__ vec3 wrap(vec3 p, vec3 lower, vec3 higher) {
    return {wrap(p.x, lower.x, higher.x), wrap(p.y, lower.y, higher.y),
            wrap(p.z, lower.z, higher.z)};
}

// fractals

#define POWER 8.0f

__device__ float sd_mandelbulb(vec3 p, float time) {
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

// primitives

__device__ float sd_box(vec3 p, vec3 bp, vec3 bs) {
    vec3 q = abs(p - bp) - bs / 2.0f;
    float udst = length(max(q, vec3(0.0f)));
    float idst = maximum(min(q, vec3(0.0f)));
    return udst + idst;
}

//

__device__ float sd_axes(vec3 p) {
    vec3 px = p;
    px.x = wrap(p.x, -0.5f, 0.5f);
    vec3 py = p;
    px.y = wrap(p.y, -0.5f, 0.5f);
    vec3 pz = p;
    px.z = wrap(p.z, -0.5f, 0.5f);

    return min(min(length(px) - 0.05f, length(py) - 0.05f), length(pz) - 0.05f);
}

// surface

template <typename SFunc>
__device__ auto make_generic_sds(SFunc sd_func, RenderSurfaceData surface) {
    return [=](vec3 p, RenderSurfaceData &surface_output) {
        float sd = sd_func(p);
        if (sd <= SURFACE_DISTANCE) {
            surface_output.color = surface.color;
        }
        return sd;
    };
}

template <typename SFunc, typename SurfFunc>
__device__ auto make_generic_location_dependent_sds(SFunc sd_func,
                                                    SurfFunc surface_func) {
    return [surface_func, sd_func](vec3 p, RenderSurfaceData &surface_output) {
        float sd = sd_func(p);
        if (sd <= SURFACE_DISTANCE) {
            surface_output = surface_func(p);
        }
        return sd;
    };
}

#define NORMAL_EPSILON 0.01f

template <typename SFunc> __device__ vec3 sd_normal(vec3 p, SFunc sd_func) {
    float dx = (-sd_func(vec3(p.x + 2.0 * NORMAL_EPSILON, p.y, p.z)) +
                8.0 * sd_func(vec3(p.x + NORMAL_EPSILON, p.y, p.z)) -
                8.0 * sd_func(vec3(p.x - NORMAL_EPSILON, p.y, p.z)) +
                sd_func(vec3(p.x - 2.0 * NORMAL_EPSILON, p.y, p.z)));

    float dy = (-sd_func(vec3(p.x, p.y + 2.0 * NORMAL_EPSILON, p.z)) +
                8.0 * sd_func(vec3(p.x, p.y + NORMAL_EPSILON, p.z)) -
                8.0 * sd_func(vec3(p.x, p.y - NORMAL_EPSILON, p.z)) +
                sd_func(vec3(p.x, p.y - 2.0 * NORMAL_EPSILON, p.z)));

    float dz = (-sd_func(vec3(p.x, p.y, p.z + 2.0 * NORMAL_EPSILON)) +
                8.0 * sd_func(vec3(p.x, p.y, p.z + NORMAL_EPSILON)) -
                8.0 * sd_func(vec3(p.x, p.y, p.z - NORMAL_EPSILON)) +
                sd_func(vec3(p.x, p.y, p.z - 2.0 * NORMAL_EPSILON)));

    return normalize(vec3(dx, dy, dz));
}
