#pragma once

#include "./libraries/glm/glm.hpp"
#include "./bindings.h"
#include "./types.cu"
#include "./utils.cu"
#include "./signed_distance.cu"
#include "./ray_marching.cu"

using namespace glm;

__device__ RayRender
render_ray(
    const Ray ray,
    const SignedDistanceScene& scene
) {
    RenderSurfaceData surface {
        { 0.0f, 0.0f, 0.0f }
    };
    RayMarchHit hit = ray_march(scene, ray);

    float light = 1.0f;
    vec3 normal = scene.normal(hit.position);

    float looking_to_light = (1.0f + dot(-normalize(vec3(1.0f, -0.25f, 0.25f)), normal)) * 0.5f;

    surface.color = vec3(0.4f, 0.5f, 0.7f);
    surface.color = 1.0f * mix(
        surface.color * 0.01f, surface.color * 1.0f,
        looking_to_light
    );

    return RayRender {
        hit, surface.color, light
    };
}

