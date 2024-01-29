#pragma once

#include "../includes/libraries/glm/glm.hpp"
#include "../includes/types.cu"
#include "../includes/utils.cu"

using namespace glm;

#define SURFACE_DISTANCE 0.001f

__device__ float wrap(const float x, const float lower, const float higher) {
    return lower + glm::mod(x - lower, higher - lower);
}

__device__ vec3 wrap(const vec3 p, const vec3 lower, const vec3 higher) {
    return {
        wrap(p.x, lower.x, higher.x), wrap(p.y, lower.y, higher.y),
        wrap(p.z, lower.z, higher.z)
    };
}

enum SdInvocationType {
    ConeType,
    RayType,
    PointType,
    SurfaceType,
};

template<SdInvocationType type>
struct SdInvocation {
};

template<>
struct SdInvocation<SdInvocationType::ConeType> {
    Ray ray;
    float radius;
};

template<>
struct SdInvocation<SdInvocationType::RayType> {
    Ray ray;
};

template<>
struct SdInvocation<SdInvocationType::PointType> {
    Ray ray;
    float radius;
};

__device__ SdInvocation<SdInvocationType::PointType> adjust_point_invocation(
    const SdInvocation<SdInvocationType::PointType> inv,
    const vec3 offset,
    const float min_radius
) {
    return SdInvocation<SdInvocationType::PointType> {
        Ray { inv.ray.position + offset, inv.ray.direction },
        max(inv.radius, min_radius)
    };
}

template<>
struct SdInvocation<SdInvocationType::SurfaceType> {
    Ray ray;
    RenderSurfaceData &surface;
};

// fractals

#define POWER 7.0f

__device__ float sd_mandelbulb(const vec3 p, const float time) {
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

__device__ float sd_unit_mandelbulb(const vec3 p) {
    return sd_mandelbulb(p / 0.4f, 0.0f) * 0.4f;
}

// primitives

__device__ float sd_unit_sphere(const vec3 p) {
    return length(p) - 0.5f;
}

__device__ float sd_box(const vec3 p, const vec3 bp, const vec3 bs) {
    vec3 q = abs(p - bp) - bs / 2.0f;
    float udst = length(max(q, vec3(0.0f)));
    float idst = maximum(min(q, vec3(0.0f)));
    return udst + idst;
}

__device__ float sd_simple_box(const vec3 p, const vec3 bp, const vec3 bs) {
    vec3 q = abs(p - bp) - bs / 2.0f;
    return maximum(min(q, vec3(0.0f)));
}

__device__ float sd_simple_bounding_box(const vec3 p, const vec3 bb_min, const vec3 bb_max) {
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

__device__ float sd_unit_cube(const vec3 p) {
    return sd_box(p, vec3(0.0f), vec3(1.0f));
}

bool inside_aabb(const vec3 p, const vec3 bb_min, const vec3 bb_max) {
    return bb_min.x <= p.x && p.x <= bb_max.x && bb_min.y <= p.y && p.y <= bb_max.y && bb_min.z <= p.z &&
           p.z <= bb_max.z;
}

float ray_distance_to_bb(const Ray &ray, const vec3 &bb_min, const vec3 &bb_max) {
    if (inside_aabb(ray.position, bb_min, bb_max)) {
        return 0.0f;
    }

    float tmin = std::numeric_limits<float>::lowest();
    float tmax = std::numeric_limits<float>::max();

    for (int i = 0; i < 3; ++i) {
        if (abs(ray.direction[i]) < std::numeric_limits<float>::epsilon()) {
            // Ray is parallel to the slab. No hit if origin not within slab
            if (ray.position[i] < bb_min[i] || ray.position[i] > bb_max[i])
                return std::numeric_limits<float>::max();
        } else {
            // Compute intersection t value of ray with near and far plane of slab
            float ood = 1.0f / ray.direction[i];
            float t1 = (bb_min[i] - ray.position[i]) * ood;
            float t2 = (bb_max[i] - ray.position[i]) * ood;

            // Make t1 be intersection with near plane, t2 with far plane
            if (t1 > t2) std::swap(t1, t2);

            // Compute the intersection of slab intersection intervals
            tmin = max(tmin, t1);
            tmax = min(tmax, t2);

            // Exit with no collision as soon as slab intersection becomes empty
            if (tmin > tmax) return std::numeric_limits<float>::max();
        }
    }

    // Ray intersects all 3 slabs. Return distance to first hit
    return tmin > 0 ? tmin : tmax;
}

//

__device__ float sd_axes(const vec3 p) {
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

#define SD_RUNTIME_STACK_MAX_DEPTH 32

__device__ float dot2(const vec3 x) {
    return dot(x, x);
}

__device__ float sd_triangle(const vec3 p, const vec3 a, const vec3 b, const vec3 c) {
    vec3 ba = b - a;
    vec3 pa = p - a;
    vec3 cb = c - b;
    vec3 pb = p - b;
    vec3 ac = a - c;
    vec3 pc = p - c;
    vec3 nor = cross(ba, ac);

    return sqrt(
        (sign(dot(cross(ba, nor), pa)) +
         sign(dot(cross(cb, nor), pb)) +
         sign(dot(cross(ac, nor), pc)) < 2.0f)
        ?
        min(
            min(
                dot2(ba * clamp(dot(ba, pa) / dot2(ba), 0.0f, 1.0f) - pa),
                dot2(cb * clamp(dot(cb, pb) / dot2(cb), 0.0f, 1.0f) - pb)),
            dot2(ac * clamp(dot(ac, pc) / dot2(ac), 0.0f, 1.0f) - pc))
        :
        dot(nor, pa) * dot(nor, pa) / dot(nor, nor));
}

__device__ float sd_mesh(const vec3 p, const unsigned int mesh_id, const SdRuntimeSceneGeometry &geometry) {
    auto mesh = &geometry.meshes.meshes[mesh_id];

    float sd = MAX_POSITIVE_F32;

    for (unsigned int i = mesh->triangle_start_id; i <= mesh->triangle_end_id; i++) {
        vec3 v0 = from_array(geometry.meshes.vertices[geometry.meshes.triangles[i].vertex_ids[0]].pos);
        vec3 v1 = from_array(geometry.meshes.vertices[geometry.meshes.triangles[i].vertex_ids[1]].pos);
        vec3 v2 = from_array(geometry.meshes.vertices[geometry.meshes.triangles[i].vertex_ids[2]].pos);
        sd = min(sd, sd_triangle(p, v0, v1, v2));
    }

    return sd;
}

__shared__ unsigned int composition_traversal_count[BLOCK_SIZE];

template<SdInvocationType type>
__device__ bool sdi_composition_should_use_bounding_box_termination(
    const float bb_dist,
    const SdInvocation<type> &inv
) {
    return bb_dist >= 0.1f;
}

template<>
__device__ bool sdi_composition_should_use_bounding_box_termination<SdInvocationType::PointType>(
    const float bb_dist,
    const SdInvocation<SdInvocationType::PointType> &inv
) {
    return bb_dist >= 0.1f + inv.radius;
}

template<>
__device__ bool sdi_composition_should_use_bounding_box_termination<SdInvocationType::ConeType>(
    const float bb_dist,
    const SdInvocation<SdInvocationType::ConeType> &inv
) {
    return bb_dist >= 0.1f + inv.radius;
}

template<SdInvocationType type>
__device__ float sdi_composition(
    const SdInvocation<type> inv,
    const SdRuntimeSceneGeometry geometry,
    const int composition_index
) {
    int stack_index = 0;
    unsigned int index = composition_index;

    float sd = 0.0f;
    bool returning = false;

    BitSet<32> second_child {};
    BitSet<32> pos_mirrored {};

    RuntimeStackNode sd_runtime_stack[SD_RUNTIME_STACK_MAX_DEPTH];

    vec3 position = inv.ray.position;

    sd_runtime_stack[stack_index] = { MAX_POSITIVE_F32 };

    while (stack_index >= 0) {
#if CUDA_DEBUG == true
        assert(stack_index >= 0);
        assert(stack_index < SD_RUNTIME_STACK_MAX_DEPTH);
#endif

        const SdComposition *const node = &geometry.compositions[index];
        RuntimeStackNode *const stack_node = &sd_runtime_stack[stack_index];

        // determine bb distance when invoking the node
        float bound_distance = 0.0f;

        if (!returning) {
            bound_distance = sd_simple_bounding_box(
                position, from_array(node->bound_min), from_array(node->bound_max)
            );
        }

        if (sdi_composition_should_use_bounding_box_termination<type>(bound_distance, inv)) {
            // early-bounding box return
            sd = bound_distance;

            index = node->parent;
            stack_index -= 1;
            returning = true;
        } else if (node->primitive_variant != SdPrimitiveVariant::None) {
            // primitive return
            const vec3 center = 0.5f * (from_array(node->bound_min) + from_array(node->bound_max));

            const auto appendix = reinterpret_cast<SdCompositionPrimitiveAppendix *>(&geometry.compositions[index + 1]);

            const quat rot = from_quat_array(appendix->rotation);
            const vec3 scale = from_array(appendix->scale);
            const vec3 primitive_position = rotate(inverse(rot), (position - center));

            switch (node->primitive_variant) {
                default:
                case SdPrimitiveVariant::None:
                case SdPrimitiveVariant::Empty:
                    sd = MAX_POSITIVE_F32;
                    break;

                case SdPrimitiveVariant::Sphere:
                    sd = sd_unit_sphere(primitive_position / scale) * minimum(scale);
                    break;

                case SdPrimitiveVariant::Cube:
                    sd = sd_box(primitive_position, vec3(0.0f), scale);
                    break;
            }

            index = node->parent;
            stack_index -= 1;
            returning = true;
        } else {
            // node handling

            if (returning && second_child.get(stack_index)) {
                // returning from second child
                switch (node->variant) {
                    case SdCompositionVariant::Difference:
                        sd = max(stack_node->sd, -sd);
                        break;

                    case SdCompositionVariant::Intersect:
                        sd = max(stack_node->sd, sd);
                        break;

                    default:
                    case SdCompositionVariant::Mirror:
                    case SdCompositionVariant::Union:
                        sd = min(stack_node->sd, sd);
                        break;
                }

                // when returning reverse position modification
                switch (node->variant) {
                    default:
                    case SdCompositionVariant::Union:
                    case SdCompositionVariant::Intersect:
                    case SdCompositionVariant::Difference:
                        break;

                    case SdCompositionVariant::Mirror: {
                        auto appendix = reinterpret_cast<SdCompositionMirrorAppendix *>(&geometry.compositions[index +
                                                                                                               1]);

                        vec3 dir = from_array(appendix->direction);

                        position = pos_mirrored.get(stack_index)
                                   ? position - 2.0f * dot(position - from_array(appendix->translation), dir) *
                                                dir
                                   : position;
                        break;
                    };
                }

                index = node->parent;
                stack_index -= 1;
                returning = true;
            } else {
                index = node->child;

                if (!returning) {
                    // when invoking the node apply position modification
                    switch (node->variant) {
                        default:
                        case SdCompositionVariant::Union:
                        case SdCompositionVariant::Intersect:
                        case SdCompositionVariant::Difference:
                            break;

                        case SdCompositionVariant::Mirror: {
                            auto appendix = &reinterpret_cast<SdCompositionMirrorAppendix *>(
                                geometry.compositions
                            )[index + 1];

                            vec3 dir = from_array(appendix->direction);
                            float diff = dot(position - from_array(appendix->translation), dir);

                            pos_mirrored.set(stack_index, diff < 0.0f);
                            position =
                                diff < 0.0f ? position - 2.0f * diff * dir : position;
                            break;
                        }
                    }
                } else {
                    // when returning from the first child update the stack sd using the returned sd,
                    // also offset the child index if the child has indicated it used two entries for its node storage
                    stack_node->sd = sd;
                    second_child.set(stack_index, true);
                    index += node->second_child_offset;
                }

                stack_index += 1;
                returning = false;

                // reset stack values when entering the next node, child offset needs not be written as it will
                // be overwritten by the child's child regardless
                sd_runtime_stack[stack_index] = { MAX_POSITIVE_F32 };
                second_child.set(stack_index, false);
            }
        }
    }

    return sd;
}

// surface

#define NORMAL_EPSILON 0.001f
#define NORMAL_EPSILON_CD (NORMAL_EPSILON * 4.0f)

template<typename SFunc>
__device__ vec3 sdi_normal(
    const SdInvocation<SdInvocationType::PointType> inv,
    const SFunc sdi_func
) {
    float dx = (-sdi_func(adjust_point_invocation(inv, vec3(2.0f * NORMAL_EPSILON, 0.0f, 0.0f), NORMAL_EPSILON_CD)) +
                8.0f * sdi_func(adjust_point_invocation(inv, vec3(NORMAL_EPSILON, 0.0f, 0.0f), NORMAL_EPSILON_CD)) -
                8.0f * sdi_func(adjust_point_invocation(inv, vec3(-NORMAL_EPSILON, 0.0f, 0.0f), NORMAL_EPSILON_CD)) +
                sdi_func(adjust_point_invocation(inv, vec3(-2.0f * NORMAL_EPSILON, 0.0f, 0.0f), NORMAL_EPSILON_CD)));

    float dy = (-sdi_func(adjust_point_invocation(inv, vec3(0.0f, 2.0f * NORMAL_EPSILON, 0.0f), NORMAL_EPSILON_CD)) +
                8.0f * sdi_func(adjust_point_invocation(inv, vec3(0.0f, NORMAL_EPSILON, 0.0f), NORMAL_EPSILON_CD)) -
                8.0f * sdi_func(adjust_point_invocation(inv, vec3(0.0f, -NORMAL_EPSILON, 0.0f), NORMAL_EPSILON_CD)) +
                sdi_func(adjust_point_invocation(inv, vec3(0.0f, -2.0f * NORMAL_EPSILON, 0.0f), NORMAL_EPSILON_CD)));

    float dz = (-sdi_func(adjust_point_invocation(inv, vec3(0.0f, 0.0f, 2.0f * NORMAL_EPSILON), NORMAL_EPSILON_CD)) +
                8.0f * sdi_func(adjust_point_invocation(inv, vec3(0.0f, 0.0f, NORMAL_EPSILON), NORMAL_EPSILON_CD)) -
                8.0f * sdi_func(adjust_point_invocation(inv, vec3(0.0f, 0.0f, -NORMAL_EPSILON), NORMAL_EPSILON_CD)) +
                sdi_func(adjust_point_invocation(inv, vec3(0.0f, 0.0f, -2.0f * NORMAL_EPSILON), NORMAL_EPSILON_CD)));

    return normalize(vec3(dx, dy, dz));
}
