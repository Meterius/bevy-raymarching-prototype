#pragma once

#include "../includes/libraries/glm/glm.hpp"
#include "../includes/types.cu"
#include "../includes/utils.cu"
#include "../includes/ray_marching.cu"
#include <assert.h>

using namespace glm;

#define SURFACE_DISTANCE 0.001f

__device__ float wrap(float x, float lower, float higher) {
    return lower + glm::mod(x - lower, higher - lower);
}

__device__ vec3 wrap(vec3 p, vec3 lower, vec3 higher) {
    return {
        wrap(p.x, lower.x, higher.x), wrap(p.y, lower.y, higher.y),
        wrap(p.z, lower.z, higher.z)
    };
}

// fractals

#define POWER 7.0f

__device__ float sd_mandelbulb(vec3 p, float time) {
    vec3 z = p;
    float dr = 1.0f;
    float r;

    float power = POWER * (1.0f + time * 0.001f);

    for (int i = 0; i < 25; i++) {
        r = length(z);
        if (r > 2.0f) {
            break;
        }

        float theta = acos(z.z / r) * power;
        float phi = atan2(z.y, z.x) * power;
        float zr = pow(r, power);
        dr = pow(r, power - 1.0f) * power * dr + 1.0f;

        float s_theta = sin(theta);
        z = zr * vec3(s_theta * cos(phi), sin(phi) * s_theta, cos(theta));
        z += p;
    }

    return 0.5f * log(r) * r / dr;
}

// primitives

__device__ float sd_unit_sphere(vec3 p) {
    return length(p) - 0.5;
}

__device__ float sd_box(vec3 p, vec3 bp, vec3 bs) {
    vec3 q = abs(p - bp) - bs / 2.0f;
    float udst = length(max(q, vec3(0.0f)));
    float idst = maximum(min(q, vec3(0.0f)));
    return udst + idst;
}

__device__ float sd_simple_box(vec3 p, vec3 bp, vec3 bs) {
    vec3 q = abs(p - bp) - bs / 2.0f;
    return maximum(min(q, vec3(0.0f)));
}

__device__ float sd_simple_bounding_box(vec3 p, vec3 bb_min, vec3 bb_max) {
    return max(
        max(
            bb_min.x - p.x,
            max(bb_min.y - p.y, bb_min.z - p.z)
        ),
        max(
            p.x - bb_max.x,
            max(p.y - bb_max.y, p.z - bb_max.z)
        )
    );
}

__device__ float sd_unit_cube(vec3 p) {
    return sd_box(p, vec3(0.0f), vec3(1.0f));
}

//

__device__ float sd_axes(vec3 p) {
    vec3 px = p;
    px.x = wrap(p.x, -0.5f, 0.5f);
    vec3 py = p;
    px.y = wrap(p.y, -0.5f, 0.5f);
    vec3 pz = p;
    px.z = wrap(p.z, -0.5f, 0.5f);

    return min(min(length(px) - 0.05f, length(py) - 0.05f), length(pz) - 0.05f);
}

// runtime

struct RuntimeStackNode {
    float sd;
};

#define USE_SHARED_RUNTIME_STACK false
#define SD_RUNTIME_STACK_MAX_DEPTH 32
#define SD_SHARED_RUNTIME_STACK_MAX_DEPTH 18

#if USE_SHARED_RUNTIME_STACK == true
__shared__ RuntimeStackNode sd_runtime_stack[BLOCK_SIZE * SD_SHARED_RUNTIME_STACK_MAX_DEPTH];
#endif

__device__ float sd_composition(
    vec3 p,
    float cd,
    SdRuntimeSceneGeometry geometry,
    int composition_index
) {
    int stack_index = 0;
    int index = composition_index;

    float sd = 0.0f;
    bool returning = false;

    BitSet<32> second_child {};
    BitSet<32> pos_mirrored {};

#if USE_SHARED_RUNTIME_STACK == false
    RuntimeStackNode sd_runtime_stack[SD_RUNTIME_STACK_MAX_DEPTH];
#endif

    const auto get_stack_node = [&](int i) {
        return &sd_runtime_stack[
#if USE_SHARED_RUNTIME_STACK == true
            threadIdx.x + i * BLOCK_SIZE
#else
            i
#endif
        ];
    };

    vec3 position = p;

    *get_stack_node(stack_index) = { MAX_POSITIVE_F32 };

    while (stack_index >= 0) {
#if CUDA_DEBUG == true
        assert(stack_index >= 0);
#if USE_SHARED_RUNTIME_STACK == true
            assert(stack_index < SD_SHARED_RUNTIME_STACK_MAX_DEPTH);
#else
            assert(stack_index < SD_RUNTIME_STACK_MAX_DEPTH);
#endif
#endif

        SdComposition node = geometry.compositions[index];
        RuntimeStackNode *stack_node = get_stack_node(stack_index);

        vec3 scale = from_array(node.bound_max) - from_array(node.bound_min);
        vec3 center = 0.5f * (from_array(node.bound_max) + from_array(node.bound_min));

        float bound_distance;
        if (returning) {
            switch (node.variant) {
                case SdCompositionVariant::Mirror:
                    position = pos_mirrored.get(stack_index)
                               ? center - 2.0f * dot(position - center, from_array(node.composition_par0)) *
                                          from_array(node.composition_par0)
                               : position;
                    break;
            }
        } else {
            bound_distance = sd_simple_bounding_box(
                position, from_array(node.bound_min), from_array(node.bound_max)
            );

            if (bound_distance > cd + 1.0f) {
                sd = bound_distance;

                index = node.parent;
                stack_index -= 1;
                returning = true;
                continue;
            }
        }

        if (node.primitive_variant != SdPrimitiveVariant::None) {
            switch (node.primitive_variant) {
                case SdPrimitiveVariant::Empty:
                    sd = MAX_POSITIVE_F32;
                    break;

                case SdPrimitiveVariant::Sphere:
                    sd = sd_unit_sphere(
                        (position - center) / scale
                    ) * minimum(scale);
                    break;

                case SdPrimitiveVariant::Cube:
                    sd = sd_box(
                        position, center, scale
                    );
                    break;

                case SdPrimitiveVariant::Mandelbulb:
                    sd = sd_mandelbulb((position - center) / (0.4f * scale), 0.0f) * 0.4f * minimum(scale);
                    break;
            }

            index = node.parent;
            stack_index -= 1;
            returning = true;
        } else {
            if (returning && second_child.get(stack_index)) {
                switch (node.variant) {
                    case SdCompositionVariant::Difference:
                        sd = max(stack_node->sd, -sd);
                        break;

                    case SdCompositionVariant::Intersect:
                        sd = max(stack_node->sd, sd);
                        break;

                    case SdCompositionVariant::Mirror:
                    case SdCompositionVariant::Union:
                        sd = min(stack_node->sd, sd);
                        break;
                }

                index = node.parent;
                stack_index -= 1;
                returning = true;
            } else {
                index = node.child;

                if (returning) {
                    stack_node->sd = sd;
                    second_child.set(stack_index, true);
                    index += 1;
                } else {
                    switch (node.variant) {
                        case SdCompositionVariant::Mirror: {
                            float diff = dot(p - center, from_array(node.composition_par0));

                            pos_mirrored.set(stack_index, diff < 0.0f);
                            position =
                                diff < 0.0f ? position - 2.0f * diff * from_array(node.composition_par0) : position;

                            break;
                        }
                    }
                }

                stack_index += 1;
                returning = false;

                *get_stack_node(stack_index) = {
                    MAX_POSITIVE_F32
                };
                second_child.set(stack_index, false);
            }
        }
    }

    return sd;
}

// surface

template<typename SFunc>
__device__ auto make_generic_sds(SFunc sd_func, RenderSurfaceData surface) {
    return [=](vec3 p, float cd, RenderSurfaceData &surface_output) {
        float sd = sd_func(p, cd);
        if (sd <= cd) {
            surface_output.color = surface.color;
        }
        return sd;
    };
}

template<typename SFunc, typename SurfFunc>
__device__ auto make_generic_location_dependent_sds(
    SFunc sd_func,
    SurfFunc surface_func
) {
    return [surface_func, sd_func](vec3 p, float cd, RenderSurfaceData &surface_output) {
        float sd = sd_func(p, cd);
        if (sd <= cd) {
            surface_output = surface_func(p, cd);
        }
        return sd;
    };
}

#define NORMAL_EPSILON 0.01f
#define NORMAL_EPSILON_CD NORMAL_EPSILON * 4.0f

template<typename SFunc>
__device__ vec3 sd_normal(vec3 p, SFunc sd_func) {
    float dx = (-sd_func(vec3(p.x + 2.0f * NORMAL_EPSILON, p.y, p.z), NORMAL_EPSILON_CD) +
                8.0f * sd_func(vec3(p.x + NORMAL_EPSILON, p.y, p.z), NORMAL_EPSILON_CD) -
                8.0f * sd_func(vec3(p.x - NORMAL_EPSILON, p.y, p.z), NORMAL_EPSILON_CD) +
                sd_func(vec3(p.x - 2.0f * NORMAL_EPSILON, p.y, p.z), NORMAL_EPSILON_CD));

    float dy = (-sd_func(vec3(p.x, p.y + 2.0f * NORMAL_EPSILON, p.z), NORMAL_EPSILON_CD) +
                8.0f * sd_func(vec3(p.x, p.y + NORMAL_EPSILON, p.z), NORMAL_EPSILON_CD) -
                8.0f * sd_func(vec3(p.x, p.y - NORMAL_EPSILON, p.z), NORMAL_EPSILON_CD) +
                sd_func(vec3(p.x, p.y - 2.0f * NORMAL_EPSILON, p.z), NORMAL_EPSILON_CD));

    float dz = (-sd_func(vec3(p.x, p.y, p.z + 2.0f * NORMAL_EPSILON), NORMAL_EPSILON_CD) +
                8.0f * sd_func(vec3(p.x, p.y, p.z + NORMAL_EPSILON), NORMAL_EPSILON_CD) -
                8.0f * sd_func(vec3(p.x, p.y, p.z - NORMAL_EPSILON), NORMAL_EPSILON_CD) +
                sd_func(vec3(p.x, p.y, p.z - 2.0f * NORMAL_EPSILON), NORMAL_EPSILON_CD));

    return normalize(vec3(dx, dy, dz));
}
