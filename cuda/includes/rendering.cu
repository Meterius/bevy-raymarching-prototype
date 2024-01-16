#pragma once

#include "./bindings.h"
#include "./ray_marching.cu"

template<typename SdSceneFunc>
__device__ vec3 render_ray(SdSceneFunc sd_scene, Ray ray, ConeMarchTextureValue starting) {
    RayMarchHit hit = ray_march<false>(sd_scene, ray, starting);
    return vec3(hit.depth * 0.001f, f32(hit.outcome == StepLimit), (float) hit.steps * 0.001f);
}
