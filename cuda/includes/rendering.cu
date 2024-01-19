#pragma once

#include "./bindings.h"
#include "./ray_marching.cu"

struct RenderSurfaceData {
    vec3 color;
};

struct RayRender {
    RayMarchHit hit;
    vec3 color;
    float light;
};

template<typename SdsFunc>
__device__ RayRender render_ray(Ray ray, SdsFunc sds_func) {
    RenderSurfaceData surface {
            { 0.0f, 0.0f, 0.0f }
    };
    RayMarchHit hit = ray_march<false>(sds_func, ray);

    surface = { { 0.0f, 0.0f, 0.0f } };
    sds_func(hit.position, surface);
}

