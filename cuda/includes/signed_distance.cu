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
    if (inside_aabb(ray.origin, bb_min, bb_max)) {
        return 0.0f;
    }

    float tmin = std::numeric_limits<float>::lowest();
    float tmax = std::numeric_limits<float>::max();

    for (int i = 0; i < 3; ++i) {
        if (abs(ray.direction[i]) < std::numeric_limits<float>::epsilon()) {
            // Ray is parallel to the slab. No hit if origin not within slab
            if (ray.origin[i] < bb_min[i] || ray.origin[i] > bb_max[i])
                return std::numeric_limits<float>::max();
        } else {
            // Compute intersection t value of ray with near and far plane of slab
            float ood = 1.0f / ray.direction[i];
            float t1 = (bb_min[i] - ray.origin[i]) * ood;
            float t2 = (bb_max[i] - ray.origin[i]) * ood;

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

//

#define NORMAL_EPSILON 0.001f

class SignedDistanceScene {
public:
    __device__ virtual float distance(const vec3 p) const;

    __device__ vec3 normal(const vec3 p) const {
        float dx = (-distance(p +  vec3(2.0f * NORMAL_EPSILON, 0.0f, 0.0f)) +
            8.0f * distance(p +  vec3(NORMAL_EPSILON, 0.0f, 0.0f)) -
            8.0f * distance(p +  vec3(-NORMAL_EPSILON, 0.0f, 0.0f)) +
            distance(p +  vec3(-2.0f * NORMAL_EPSILON, 0.0f, 0.0f)));

        float dy = (-distance(p +  vec3(0.0f, 2.0f * NORMAL_EPSILON, 0.0f)) +
                    8.0f * distance(p +  vec3(0.0f, NORMAL_EPSILON, 0.0f)) -
                    8.0f * distance(p +  vec3(0.0f, -NORMAL_EPSILON, 0.0f)) +
                    distance(p +  vec3(0.0f, -2.0f * NORMAL_EPSILON, 0.0f)));

        float dz = (-distance(p +  vec3(0.0f, 0.0f, 2.0f * NORMAL_EPSILON)) +
                    8.0f * distance(p +  vec3(0.0f, 0.0f, NORMAL_EPSILON)) -
                    8.0f * distance(p +  vec3(0.0f, 0.0f, -NORMAL_EPSILON)) +
                    distance(p +  vec3(0.0f, 0.0f, -2.0f * NORMAL_EPSILON)));

        return normalize(vec3(dx, dy, dz));
    }
};

class DefaultSignedDistanceScene : public SignedDistanceScene {
public:
    __device__ float distance(const vec3 p) const {
        return min(
            sd_box(p, vec3(0.0f, -0.5f, 0.0f), vec3(30.0f, 1.0f, 30.0f)),
            length(p - vec3(15.0f, 1.5f, 0.0f)) - 1.0f,
            length(p - vec3(13.0f, 0.75f, 2.0f)) - 1.0f
        );
    }
};
