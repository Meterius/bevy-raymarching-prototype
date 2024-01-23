#pragma once

#include "./libraries/glm/glm.hpp"
#include "./bindings.h"
#include "./types.cu"
#include "./utils.h"
#include "./ray_marching.cu"

using namespace glm;

template<typename SdsFunc>
__device__ RayRender
render_ray(
    Ray ray, float cone_radius_at_unit, SdsFunc sds_func, SdRuntimeSceneLighting lighting,
    ConeMarchTextureValue starting
) {
    RenderSurfaceData surface {
        { 0.0f, 0.0f, 0.0f }
    };
    RayMarchHit hit = ray_march<true>(
        [&](vec3 p, float cd) { return sds_func(p, cd, surface); }, ray, starting, cone_radius_at_unit
    );

    RenderSurfaceData light_surface {};

    float light = 2.0f;
    vec3 normal = sd_normal(hit.position, [&](vec3 p, float cd) { return sds_func(p, cd, surface); });

    for (int i = 0; i < 0 * lighting.sun_light_count; i++) {
        Ray light_ray = { hit.position, -from_array(lighting.sun_lights[i].direction) };
        light_ray.position += light_ray.direction * 0.01f;
        light = max(
            0.05f,
            soft_shadow_ray_march([&](vec3 p, float cd) { return sds_func(p, cd, light_surface); }, light_ray, 0.1f)
        );
    }

    float looking_to_light = (1.0f + dot(-from_array(lighting.sun_lights[0].direction), normal)) * 0.5f;

    surface.color = mix(
        0.5f * (surface.color + vec3(0.0f, 0.0f, 1.0f)), 0.5f * (surface.color + vec3(1.0f, 0.0f, 0.0f)),
        looking_to_light
    );

    return RayRender {
        hit, surface.color, light
    };
}

