// #import "shaders/compiled/render_scene_sd.wgsl"::{sd_scene, sd_scene_normal, SdSceneData} 

// Ray Marching

const RAY_MARCHING_MAX_STEP_DEPTH = 4000;
const RAY_MARCHER_COLLISION_DISTANCE = 0.0001;
const RAY_MARCHER_MAX_DEPTH = 1000000.0;

const CUTOFF_REASON_NONE = 0u;
const CUTOFF_REASON_DISTANCE = 1u;
const CUTOFF_REASON_STEPS = 2u;

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct RayMarchHit {
    pos: vec3<f32>,
    depth: f32,
    step_depth: i32,
    shortest_distance: f32,
    soft_shadow_factor: f32,
    cutoff_reason: u32,
    average_sd_scene_data: SdSceneData,
    minimal_sd_scene_data: SdSceneData,
    maximal_sd_scene_data: SdSceneData,
}

fn is_colliding_distance(sd: f32) -> bool {
    return sd <= RAY_MARCHER_COLLISION_DISTANCE;
}

const APPROX_AO_SAMPLE_COUNT = 5;
const APPROX_AO_SAMPLE_STEP = 0.1;

fn ray_march_hit_approx_soft_shadow(ray: Ray, hit: RayMarchHit, sun_dir: vec3<f32>) -> f32 {
    let light_hit = ray_march_with(Ray(hit.pos, sun_dir), RayMarchOptions(RAY_MARCHER_MAX_DEPTH, true));
    return f32(light_hit.cutoff_reason != CUTOFF_REASON_NONE) * clamp(
        pow(light_hit.soft_shadow_factor, 0.8), 0.0, 1.0,
    );
}

fn ray_march_hit_approx_ao(ray: Ray, hit: RayMarchHit) -> f32 {
    if (hit.cutoff_reason != CUTOFF_REASON_NONE) {
        return 0.0;
    }

    let normal = sd_scene_normal(hit.pos);

    var total = 0.0;

    for (var i = 1; i < APPROX_AO_SAMPLE_COUNT; i += 1) {
        let delta = f32(i) * APPROX_AO_SAMPLE_STEP;
        let sd = sd_scene(hit.pos + normal * delta).sd;
        total += pow(2.0, f32(-i)) * (delta - sd);
    }

    return 1.0 - clamp(5.0 * total, 0.0, 1.0);
}

struct RayMarchOptions {
    depth_limit: f32,
    use_hit_escape: bool,
}

fn ray_march_with(ray: Ray, options: RayMarchOptions) -> RayMarchHit {
    var depth = f32(options.use_hit_escape) * 0.01;
    var position = ray.origin + depth * ray.direction;
    var shortest_distance = 3.40282346638528859812e+38f;
    var soft_shadow_factor = 3.40282346638528859812e+38f;

    var step_depth = 0;
    var cutoff_reason = 0u;

    var sd_scene_data_total = SdSceneData(0.0, 0.0);
    var sd_scene_data_minimal = SdSceneData(3.40282346638528859812e+38f, 3.40282346638528859812e+38f);
    var sd_scene_data_maximal = SdSceneData(-3.40282346638528859812e+38f, -3.40282346638528859812e+38f);

    for (; step_depth < RAY_MARCHING_MAX_STEP_DEPTH; step_depth += 1) {
        let sd_scene_data = sd_scene(position);

        sd_scene_data_total.sd += sd_scene_data.sd;
        sd_scene_data_total.iterations += sd_scene_data.iterations;
        sd_scene_data_minimal.sd = min(sd_scene_data_minimal.sd, sd_scene_data.sd);
        sd_scene_data_minimal.iterations = min(sd_scene_data_minimal.iterations, sd_scene_data.iterations);
        sd_scene_data_maximal.sd = max(sd_scene_data_maximal.sd, sd_scene_data.sd);
        sd_scene_data_maximal.iterations = max(sd_scene_data_maximal.iterations, sd_scene_data.iterations);

        let sd = sd_scene_data.sd;

        shortest_distance = min(sd, shortest_distance);

        if (depth >= 0.1) {
            soft_shadow_factor = min(
                soft_shadow_factor,
                sd / depth,
            );
        }

        if (sd <= RAY_MARCHER_COLLISION_DISTANCE + max(0.0, depth - 100.0) * (0.01 / 100.0)) {
            step_depth += 1;
            break;
        }

        depth += sd;
        let ray_offset = vec3<f32>(0.0); // vec3<f32>(0.0, sin(depth) * 0.35, 0.0);
        position = ray.origin + depth * ray.direction + ray_offset;

        if (depth >= options.depth_limit) {
            cutoff_reason = CUTOFF_REASON_DISTANCE;
            step_depth += 1;
            break;
        }
    }

    if (step_depth == RAY_MARCHING_MAX_STEP_DEPTH) {
        cutoff_reason = CUTOFF_REASON_STEPS;
    }

    let sd_scene_data_average = SdSceneData(
        sd_scene_data_total.sd / f32(step_depth),
        sd_scene_data_total.iterations / f32(step_depth)
    );

    return RayMarchHit(
        position, depth, step_depth, shortest_distance,
        soft_shadow_factor, cutoff_reason, sd_scene_data_average,
        sd_scene_data_minimal, sd_scene_data_maximal
    );
}

fn ray_march(ray: Ray) -> RayMarchHit {
    return ray_march_with(ray, RayMarchOptions(RAY_MARCHER_MAX_DEPTH, false));
}
