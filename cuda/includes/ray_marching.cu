#pragma once

#include "../includes/libraries/glm/glm.hpp"
#include "../includes/bindings.h"
#include "../includes/types.cu"

using namespace glm;

#define RAY_MARCH_STEP_LIMIT 500
#define RAY_MARCH_DEPTH_LIMIT 10000.0f
#define RAY_MARCH_COLLISION_DISTANCE 0.001f

#define RAY_MARCH_LANE_PAIR_DELTA 11
#define RAY_MARCH_LANE_PAIR_PRED_FAC 0.95f

template<bool useConeMarch, typename SdSceneFunc>
__device__ RayMarchHit ray_march(
    const SdSceneFunc sd_scene,
    const Ray ray,
    const ConeMarchTextureValue starting = ConeMarchTextureValue {},
    const float cone_radius_at_unit = 0.0f
) {
    RayMarchHit hit {
        (int) starting.steps,
        ray.position + ray.direction * starting.depth,
        starting.depth,
        StepLimit
    };

    float unadjusted_depth = starting.depth;

    int pair_lane =
        (threadIdx.x + (threadIdx.x % 2 == 0 ? RAY_MARCH_LANE_PAIR_DELTA : -RAY_MARCH_LANE_PAIR_DELTA)) % 32;

    Ray pair_ray;
    __shfl_sync_ray(0xffffffff, ray, pair_lane, pair_ray);

    bool finished = false;
    bool pair_finished = false;

    float pair_prev_sd = 0.0f;

    if (starting.outcome == DepthLimit) {
        finished = true;
        hit.outcome = DepthLimit;
    }

    if (blockIdx.x == 20 && (threadIdx.x == 0 || threadIdx.x == RAY_MARCH_LANE_PAIR_DELTA)) {
//        printf(
//            "START; %d; Dir: %f %f %f Pair Dir: %f %f %f", threadIdx.x, ray.direction.x,
//            ray.direction.y, ray.direction.z, pair_ray.direction.x, pair_ray.direction.y, pair_ray.direction.z
//        );
    }

    float prev_sd = 0.0f;
    for (unsigned int steps = 0; (steps < RAY_MARCH_STEP_LIMIT) && (!finished || !pair_finished); steps++) {
        pair_finished = __shfl_sync(0xffffffff, finished, pair_lane);

        const float collision_distance = useConeMarch ? max(
            RAY_MARCH_COLLISION_DISTANCE, cone_radius_at_unit * hit.depth
        ) : RAY_MARCH_COLLISION_DISTANCE;

        const float pair_collision_distance = __shfl_sync(0xffffffff, collision_distance, pair_lane);
        const float pair_depth = __shfl_sync(0xffffffff, unadjusted_depth, pair_lane);

        float d = sd_scene(
            finished
            ? pair_ray.position + (pair_depth + pair_prev_sd * RAY_MARCH_LANE_PAIR_PRED_FAC) * pair_ray.direction
            : hit.position,
            finished ? pair_collision_distance : collision_distance
        );

        const float pair_d = __shfl_sync(0xffffffff, d, pair_lane);

        if (blockIdx.x == 20 && (threadIdx.x == 0 || threadIdx.x == RAY_MARCH_LANE_PAIR_DELTA)) {
//            vec3 p_pair =
//                pair_ray.position + pair_depth * pair_ray.direction;
//            printf(
//                "iter: %d; %d; sd_pos: %f %f %f; sd_pair_pos: %f %f %f; sd_dist: %f\n", steps, threadIdx.x,
//                hit.position.x, hit.position.y, hit.position.z, p_pair.x, p_pair.y, p_pair.z, collision_distance
//            );
//            printf(
//                "iter: %d; %d; finished: %d; pair_finished: %d; d: %f; coll_d: %f depth: %f; pair_depth: %f;\n", steps,
//                threadIdx.x, finished, pair_finished, d, collision_distance, unadjusted_depth, pair_depth,
//                pair_depth
//            );

//if (!finished && pair_finished) {
//    printf("iter: %d; %d; d: %f pair_d: %f overshoot: %f\n\n", steps, threadIdx.x, d, pair_d, prev_sd * RAY_MARCH_LANE_PAIR_PRED_FAC);
//}
        }

        if (!finished && pair_finished && (pair_d + d) > pair_prev_sd * RAY_MARCH_LANE_PAIR_PRED_FAC) {
            composition_traversal_count[threadIdx.x] += 1.0f;
        }

        d = !finished && pair_finished && (pair_d + d) > prev_sd * RAY_MARCH_LANE_PAIR_PRED_FAC
            ? max(d, prev_sd * RAY_MARCH_LANE_PAIR_PRED_FAC + pair_d) : d;

        pair_prev_sd = __shfl_sync(0xffffff, d, pair_lane);
        prev_sd = d;

        if (!finished && d <= collision_distance) {
            hit.outcome = Collision;
            hit.steps = steps;
            finished = true;
        }

        if (!finished) {
            if (useConeMarch) {
                hit.depth += d - collision_distance;
                unadjusted_depth += d;
                hit.position += d * ray.direction;
            } else {
                hit.depth += d;
                unadjusted_depth += d;
                hit.position += d * ray.direction;
            }

            if (hit.depth > RAY_MARCH_DEPTH_LIMIT) {
                hit.outcome = DepthLimit;
                hit.steps = steps;
                finished = true;
            } else if (steps == RAY_MARCH_STEP_LIMIT - 1) {
                hit.steps = steps;
                finished = true;
            }
        }
    }

//    if (blockIdx.x == 20) {
//        hit.outcome = RayMarchHitOutcome::StepLimit;
//    }
//
//    if (blockIdx.x == 20 && (threadIdx.x == 0 || threadIdx.x == RAY_MARCH_LANE_PAIR_DELTA)) {
//        hit.outcome = RayMarchHitOutcome::DepthLimit;
//    }

    // assert(blockIdx.x != 207 || threadIdx.x != 32);

    return hit;
}

template<typename SdSceneFunc>
__device__ float soft_shadow_ray_march(
    const SdSceneFunc sd_scene,
    const Ray ray,
    const float w
) {
    float res = 1.0f;
    float ph = 1e20f;
    float depth = 0.0f;

    for (int i = 0; i < RAY_MARCH_STEP_LIMIT; i++) {
        float sd = sd_scene(ray.position + ray.direction * depth, RAY_MARCH_COLLISION_DISTANCE);

        if (sd <= RAY_MARCH_COLLISION_DISTANCE) {
            return 0.0f;
        }

        float y = sd * sd / (2.0f * ph);
        float d = sqrt(sd * sd - y * y);
        res = min(res, d / (w * max(0.0f, depth - y)));
        ph = sd;
        depth += sd;

        if (depth > RAY_MARCH_DEPTH_LIMIT) {
            break;
        }
    }

    return res;
}
