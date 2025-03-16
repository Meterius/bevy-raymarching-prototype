#pragma once

#include "./libraries/glm/glm.hpp"
#include "./bindings.h"
#include "./types.cu"
#include "./utils.cu"
#include "./signed_distance.cu"
#include "./ray_marching.cu"

using namespace glm;

__device__ vec3 skybox(const Texture& environment, vec3 dir, vec3 sun_dir) {
    const float sun_proj = dot(dir, -sun_dir);
    const float sun_angle = acos(sun_proj);

    float SUN_DISK_ANGLE = PI * 0.5f / 180.0f;
    float SUN_FADE_ANGLE = PI * 3.0f / 180.0f;

    vec3 sun_color = vec3(20.0f);

    float azimuth = clamp(0.5f + atan2(dir.z, dir.x) / (2.0f * PI), 0.0f, 1.0f);
    float elevation = clamp(0.5f + atan2(dir.y, sqrtf(dir.x * dir.x + dir.z * dir.z)) / PI, 0.0f, 1.0f);

    ivec2 environment_pos = {
        min(int(float(environment.size[0]) * azimuth), environment.size[0] - 1),
        min(int(float(environment.size[1]) * (1.0f - elevation)), environment.size[1] - 1)
    };

    unsigned int sky_color_rgba = environment.texture[environment_pos.y * environment.size[0] + environment_pos.x];
    vec3 sky_color = 4.0f * vec3 { float(sky_color_rgba & 0xff) / 255.0f, float(sky_color_rgba >> 8 & 0xff) / 255.0f, float(sky_color_rgba >> 16 & 0xff) / 255.0f };

    if (sun_angle <= SUN_DISK_ANGLE) {
        return sun_color;
    } else if (sun_angle <= SUN_FADE_ANGLE) {
        float fac = 1.0f - (sun_angle - SUN_DISK_ANGLE) / (SUN_FADE_ANGLE - SUN_DISK_ANGLE);
        return mix(sky_color, sun_color, pow(fac, 2.0f));
    } else {
        return sky_color;
    }
}

__device__ RayRender
render_ray(
    const Ray ray,
    const SceneBuffer& scene,
    const SignedDistanceScene& sd_scene,
    const Texture& environment
) {
    vec3 sun_dir = { scene.sun_direction[0], scene.sun_direction[1], scene.sun_direction[2] };

    RayMarchHit hit = ray_march(sd_scene, ray);

    vec3 material_color = vec3(0.4f, 0.5f, 0.7f);

    float light = 1.0f;
    vec3 color = vec3(0.0f);

    if (hit.outcome == Collision) {
        vec3 normal = sd_scene.normal(hit.position);

        float looking_to_light = (1.0f + dot(-sun_dir, normal)) * 0.5f;
        float occlusion = ray_march_ambient_occlusion(sd_scene, Ray { hit.position, normal });
        float shadow = ray_march_softshadow(sd_scene, Ray { hit.position, -sun_dir });

        light = (0.8f + 0.2f * looking_to_light) * (0.3f + 0.7f * occlusion) * (0.1f + 0.9f * shadow);
        color = material_color * (0.1f + 0.9f * light);
    } else {
        color = skybox(environment, ray.direction, sun_dir);
    }

    return RayRender {
        hit, color
    };
}

