#pragma once

#include "./bindings.h"
#include "./ray_marching.cu"

template<typename SdSceneFunc>
__device__ RenderDataTextureValue render_ray(SdSceneFunc sd_scene, Ray ray, ConeMarchTextureValue starting) {
    RayMarchHit hit = ray_march<false>(sd_scene, ray, starting);

    return {
        hit.depth, (float) hit.steps, hit.outcome, { 1.0f, 1.0f, 1.0f }, 1.0f
    };
}
