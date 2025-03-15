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
    const SceneBuffer& scene,
    const SignedDistanceScene& sd_scene
) {
    vec3 sun_dir = { scene.sun_direction[0], scene.sun_direction[1], scene.sun_direction[2] };

    RenderSurfaceData surface {
        { 0.0f, 0.0f, 0.0f }
    };
    RayMarchHit hit = ray_march(sd_scene, ray);

    vec3 normal = sd_scene.normal(hit.position);

    float looking_to_light = (1.0f + dot(-sun_dir, normal)) * 0.5f;
    float occlusion = hit.outcome == RayMarchHitOutcome::Collision
        ? ray_march_ambient_occlusion(sd_scene, Ray { hit.position, normal }) : 1.0f;
    float shadow = ray_march_softshadow(sd_scene, Ray { hit.position, -sun_dir });

    float light = (0.8f + 0.2f * looking_to_light) * (0.3f + 0.7f * occlusion) * (0.1f + 0.9f * shadow);

    surface.color = vec3(0.4f, 0.5f, 0.7f);

    return RayRender {
        hit, surface.color, light
    };
}

