// #import "shaders/compiled/render_scene_sd.wgsl"::{sd_scene, sd_scene_normal, SdSceneData} 

// Ray Marching

var<private> is_main_ray: bool = true;
var<workgroup> workgroup_main_ray_step_depth: i32;
var<workgroup> workgroup_main_ray_step_pos: array<vec3<f32>, RAY_MARCHING_MAX_STEP_DEPTH>;
var<workgroup> workgroup_main_ray_step_sd: array<f32, RAY_MARCHING_MAX_STEP_DEPTH>;

const RAY_MARCHING_MAX_STEP_DEPTH = 200;
const RAY_MARCHER_COLLISION_DISTANCE = 0.0001;
const RAY_MARCHER_MAX_DEPTH = 10000.0;

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
    if (true) {
        return 0.0;
    }

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

    var ref_step_depth = 0;
    var max_step_depth = RAY_MARCHING_MAX_STEP_DEPTH;
    var skipped_depth = 0.0;

    for (; step_depth < max_step_depth; step_depth += 1) {
        var sd_scene_data: SdSceneData;

        if (is_main_ray) {
            if (is_main_invocation) {
                sd_scene_data = sd_scene(position);
                workgroup_main_ray_step_pos[step_depth] = position;
                workgroup_main_ray_step_sd[step_depth] = sd_scene_data.sd;
                workgroup_main_ray_step_depth += 1;
            } else {
                if (true) {
                    loop {
                        loop {
                            if (ref_step_depth + 1 < workgroup_main_ray_step_depth) {
                                let dist = distance(workgroup_main_ray_step_pos[ref_step_depth], position);
                                let next_dist = distance(position, workgroup_main_ray_step_pos[ref_step_depth + 1]);

                                if (next_dist < dist) {
                                    ref_step_depth += 1;
                                } else {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }

                        let dist_pos = distance(workgroup_main_ray_step_pos[ref_step_depth], position);
                        let dist_sd = workgroup_main_ray_step_sd[ref_step_depth];
                        let diff = dist_sd - dist_pos;

                        if (dist_sd > 0.01 && diff > 0.01) {
                            // let sp_center = workgroup_main_ray_step_pos[step_depth_ref];
                            // let sp_diff = position - sp_center;
                            // let uoc = dot(ray.direction, sp_diff);
                            // let delta = pow(uoc, 2.0) - (dot(sp_diff, sp_diff) - pow(workgroup_main_ray_step_sd[step_depth_ref], 2.0));
                            // delta > 0.0 && -uoc - sqrt(delta) < 0.0 && -uoc + sqrt(delta) > 0.0
                            // sd_scene_data = SdSceneData(sqrt(delta), 0.0);
                            // sd_scene_data = sd_scene(position);
                            depth += diff - 0.005;
                            skipped_depth += diff - 0.005;
                            position = ray.origin + depth * ray.direction;
                            step_depth += 1;
                            max_step_depth += 1;
                        } else {
                            sd_scene_data = sd_scene(position);
                            break;
                        }
                    }
                } else {
                    sd_scene_data = sd_scene(position);
                }
            }
        } else {
            sd_scene_data = sd_scene(position);
        }

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

        if (sd <= RAY_MARCHER_COLLISION_DISTANCE + max(0.0, depth - 50.0) * (0.01 / 100.0)) {
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

    if (step_depth == max_step_depth) {
        cutoff_reason = CUTOFF_REASON_STEPS;
    }

    let sd_scene_data_average = SdSceneData(
        sd_scene_data_total.sd / f32(step_depth),
        sd_scene_data_total.iterations / f32(step_depth)
    );

    
    if (is_main_ray) {
        // pixel_color_override = vec3<f32>(0.0, 0.0, skipped_depth / depth);
    }

    is_main_ray = false;

    return RayMarchHit(
        position, depth, step_depth, shortest_distance,
        soft_shadow_factor, cutoff_reason, sd_scene_data_average,
        sd_scene_data_minimal, sd_scene_data_maximal
    );
}

fn ray_march(ray: Ray) -> RayMarchHit {
    return ray_march_with(ray, RayMarchOptions(RAY_MARCHER_MAX_DEPTH, false));
}
