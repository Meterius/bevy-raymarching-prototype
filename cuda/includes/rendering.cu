#pragma once

#include "./libraries/glm/glm.hpp"
#include "./bindings.h"
#include "./types.cu"
#include "./utils.cu"
#include "./signed_distance.cu"
#include "./ray_marching.cu"

using namespace glm;

struct RayRenderParameters {
    const SceneBuffer& scene;
    const SignedDistanceScene& sd_scene;
};

__device__ vec3 skybox(
    const RayRenderParameters parameters,
    const vec3 direction
) {
    const float sun_proj = dot(direction, -from_array(parameters.scene.sun.direction));
    const float sun_angle = acos(sun_proj);

    float SUN_DISK_ANGLE = PI * 0.5f / 180.0f;
    float SUN_FADE_ANGLE = PI * 3.0f / 180.0f;

    vec3 sun_color = from_array(parameters.scene.sun.color) * parameters.scene.sun.intensity;

    float azimuth = clamp(0.5f + atan2(direction.z, direction.x) / (2.0f * PI), 0.0f, 1.0f);
    float elevation = clamp(0.5f + atan2(direction.y, sqrtf(direction.x * direction.x + direction.z * direction.z)) / PI, 0.0f, 1.0f);

    auto environment = parameters.scene.environment_texture;

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

struct LightingParameters {
    Ray surface_ray;
    Line surface_light_line;
    Line view_line;
};

__device__ float phong_diffuse(
    const LightingParameters l_param
) {
    float n_dot_l = dot(l_param.surface_light_line.direction, l_param.surface_ray.direction);
    return clamp(n_dot_l, 0.0f, 1.0f);
}

__device__ float phong_specular(
    const LightingParameters l_param, const float specular_power
) {
    vec3 h = reflect(l_param.surface_light_line.direction, l_param.surface_ray.direction);
    float n_dot_h = dot(l_param.view_line.direction, h);
    return pow(clamp(n_dot_h, 0.0f, 1.0f), specular_power);
}

__device__ vec3 render_surface_point_light(
    const RayRenderParameters parameters, const PointLight light, const LightingParameters l_param, const Material surface_material
) {
    float surface_light_angle = dot(l_param.surface_light_line.direction, l_param.surface_ray.direction);

    if (surface_light_angle >= 0.0f) {
        RayMarchHit shadow = ray_march(parameters.sd_scene, l_param.surface_light_line.as_ray(), l_param.surface_light_line.distance, true);

        if (shadow.outcome == RayMarchHitOutcome::DepthLimit) {
            float diffuse = phong_diffuse(l_param);
            float specular = 2.0f * phong_specular(l_param, mix(200.0f, 30.0f, pow(surface_material.roughness, 0.5f)));

            float falloff = 1.0f / (l_param.surface_light_line.distance * l_param.surface_light_line.distance);
            float diff_spec = 0.3f + mix(specular, diffuse, 0.1f + 0.9f * surface_material.roughness);

            return from_array(light.color) * light.intensity * surface_material.color * falloff * diff_spec;
        } else {
            return vec3(0.0f);
        }
    } else {
        return vec3(0.0f);
    }
}

__device__ vec3 render_surface_sun_light(
    const RayRenderParameters parameters, const SunLight light, const LightingParameters l_param, const Material surface_material
) {
    float diffuse = phong_diffuse(l_param);
    float specular = 2.0f * phong_specular(l_param, mix(30.0f, 10.0f, surface_material.roughness));

    float shadow = ray_march_softshadow(parameters.sd_scene, l_param.surface_light_line.as_ray());

    const float diff_spec_factor = 0.9f;
    float diff_spec = 0.3f + mix(specular * shadow, diffuse, 0.1f + 0.9f * surface_material.roughness);

    return from_array(light.color) * light.intensity * surface_material.color * mix(1.0f, diff_spec, diff_spec_factor) * (0.2f + 0.8f * shadow);
}

__device__ RayRender
render_ray(const RayRenderParameters parameters, const Ray ray) {
    RayMarchHit hit = ray_march(parameters.sd_scene, ray, RAY_MARCH_DEPTH_LIMIT, false);

    vec3 color = vec3(0.0f);
    if (hit.outcome == Collision) {
        Material surface_material { vec3(0.4f, 0.5f, 0.7f), 0.5f };

        vec3 normal = parameters.sd_scene.normal(hit.position);

        float occlusion = ray_march_ambient_occlusion(parameters.sd_scene, Ray { hit.position, normal });

        LightingParameters l_params {
            Ray { hit.position, normal },
            Line { hit.position, -from_array(parameters.scene.sun.direction), 0.0f },
            Line { ray.origin, ray.direction, hit.depth },
        };

        vec3 sun_color = (0.3f + 0.7f * occlusion) * render_surface_sun_light(parameters, parameters.scene.sun, l_params, surface_material);
        color += sun_color;

        for (int i = 0; i < parameters.scene.point_light_count; i++) {
            PointLight light = parameters.scene.point_lights[i];

            vec3 light_difference = from_array(light.position) - hit.position;
            float light_distance = length(light_difference);

            l_params.surface_light_line = Line { hit.position, light_difference / light_distance, light_distance };

            color += render_surface_point_light(parameters, light, l_params, surface_material);
        }
    } else {
        color = skybox(parameters, ray.direction);
    }

    return RayRender {
        hit, color
    };
}

