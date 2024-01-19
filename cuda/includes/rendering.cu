#pragma once

#include "./libraries/glm/glm.hpp"
#include "./bindings.h"
#include "./types.cu"
#include "./utils.h"
#include "./ray_marching.cu"

using namespace glm;

template<typename SdsFunc>
__device__ RayRender render_ray(Ray ray, SdsFunc sds_func, SdRuntimeSceneLighting lighting) {
    RenderSurfaceData surface {
        { 0.0f, 0.0f, 0.0f }
    };
    RayMarchHit hit = ray_march<false>([&](vec3 p) { return sds_func(p, surface); }, ray);

    float light = 0.0f;
    vec3 normal = sd_normal(hit.position, [&](vec3 p) { return sds_func(p, surface); });

    for (int i = 0; i < lighting.sun_light_count; i++) {
        Ray light_ray = { hit.position, -from_array(lighting.sun_lights[i].direction) };
        light_ray.position += light_ray.direction * 0.1f;
        RayMarchHit light_hit = ray_march<false>([&](vec3 p) { return sds_func(p, surface); }, light_ray);

        light = light_hit.outcome == Collision ? 0.5f : 1.0f;
    }

    sds_func(hit.position, surface);

    return RayRender {
        hit, surface.color, light
    };
}

