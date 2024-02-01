#pragma once

#include "./libraries/glm/glm.hpp"
#include "./bindings.h"
#include "./types.cu"
#include "./utils.cu"
#include "./signed_distance.cu"
#include "./ray_marching.cu"

using namespace glm;

template<typename SdiFunc, typename SdiSurfFunc, typename SdiPointFunc, typename SdiRayFunc>
__device__ RayRender
render_ray(
    const Ray ray,
    const float cone_radius_at_unit,
    const SdiFunc sdc_func,
    const SdiSurfFunc sds_func,
    const SdiPointFunc sdp_func,
    const SdiRayFunc sdr_func,
    const SdRuntimeSceneLighting lighting,
    const ConeMarchTextureValue starting
) {
    RenderSurfaceData surface {
        { 0.0f, 0.0f, 0.0f }
    };
    RayMarchHit hit = ray_march(
        [&](auto inv) { return sdc_func(inv); }, ray, starting, cone_radius_at_unit
    );

    RenderSurfaceData light_surface {};

    float light = 1.0f;
    vec3 normal = sdi_normal(SdInvocation<SdInvocationType::PointType> {{ hit.position, vec3(0.0f) }}, sdp_func);

    for (int i = 0; i < 0 * lighting.sun_light_count; i++) {
        Ray light_ray = { hit.position, -from_array(lighting.sun_lights[i].direction) };
        light_ray.origin += light_ray.direction * 0.01f;
        light = max(
            0.05f,
            soft_shadow_ray_march(sdr_func, light_ray, 0.1f)
        );
    }

    float looking_to_light = (1.0f + dot(-from_array(lighting.sun_lights[0].direction), normal)) * 0.5f;

    surface.color = vec3(0.4f, 0.5f, 0.7f);
    surface.color = 1.0f * mix(
        surface.color * vec3(0.0f, 0.0f, 1.0f), surface.color * vec3(1.0f, 0.0f, 0.0f),
        looking_to_light
    );

    return RayRender {
        hit, surface.color, light
    };
}

