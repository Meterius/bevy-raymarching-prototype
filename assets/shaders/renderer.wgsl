#import "shaders/compiled/utils.wgsl"::{PI, max_comp3, euclid_mod, smooth_min, wrap, wrap_reflect, min4, min5, min3}
#import "shaders/compiled/color.wgsl"::{color_map_default, color_map_a, color_map_temp}
#import "shaders/compiled/phong_reflection_model.wgsl"::{PhongReflectionMaterial, mix_material, phong_reflect_color, PhongReflectionLight}
#import "shaders/compiled/signed_distance.wgsl"::{sdSphere, sdUnion, sdPostSmoothUnion, sdRecursiveTetrahedron, sdBox, sdPreCheapBend, sdPreMirrorB}

@group(0) @binding(0) var texture: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(1) var<uniform> frame: RayMarcherFrameData;

struct RayMarcherFrameData {
    time: f32,

    texture_size: vec2<f32>,
    screen_size: vec2<f32>,
    aspect_ratio: f32,

    cam_unit_plane_dist: f32,

    cam_pos: vec3<f32>,
    cam_forward: vec3<f32>,
    cam_up: vec3<f32>,
    cam_right: vec3<f32>,

    world_scale: f32,
}

//

fn invocationIdToTextureCoord(invocation_id: vec3<u32>) -> vec2<i32> {
    return vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
}

fn textureCoordToViewportCoord(texture_coord: vec2<f32>) -> vec2<f32> {
    let flipped = vec2<f32>(2.0) * texture_coord / vec2<f32>(frame.texture_size) - vec2<f32>(1.0);
    return flipped * vec2<f32>(1.0, -1.0);
}

fn viewportCoordToRayDir(viewport_coord: vec2<f32>) -> vec3<f32> {
    return normalize(
        frame.cam_forward * frame.cam_unit_plane_dist
        + frame.cam_right * viewport_coord.x * 0.5 * frame.aspect_ratio
        + frame.cam_up * viewport_coord.y * 0.5
    );
}

// Scene

fn sdSceneAxes(p: vec3<f32>) -> f32 {
    return min3(
        sdSphere(vec3<f32>(p.x, wrap(p.y, -0.5, 0.5), p.z), vec3<f32>(0.0), 0.1),
        sdSphere(vec3<f32>(wrap(p.x, -0.5, 0.5), p.y, p.z), vec3<f32>(0.0), 0.1),
        sdSphere(vec3<f32>(p.x, p.y, wrap(p.z, -0.5, 0.5)), vec3<f32>(0.0), 0.1),
    );
}

fn sdSceneColumn(p: vec3<f32>) -> f32 {
    return min3(
        sdBox(p, vec3<f32>(0.0, 3.0, 0.0), vec3<f32>(0.5, 3.0, 0.5)),
        sdBox(p, vec3<f32>(0.0, 5.0, 0.0), vec3<f32>(0.75, 0.15, 0.75)),
        sdBox(p, vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.7, 1.0, 0.7)),
    );
}

struct SdSceneData {
    sd: f32,
    iterations: f32,
}

fn sdScene(p: vec3<f32>) -> SdSceneData {
    var q = p;

    // Tetrahedron
    let tetrahedron_scale = 400.0;
    let tetrahedron_wrapped_pos = mix(vec3<f32>(
        wrap(q.x, -tetrahedron_scale*2.0, tetrahedron_scale*2.0),
        q.y,
        wrap(q.z, -tetrahedron_scale*2.0, tetrahedron_scale*2.0),
    ), q, 1.0);

    let tetrahedron_translated_pos = tetrahedron_wrapped_pos - vec3<f32>(0.0, tetrahedron_scale, 0.0);
    let tetrahedron_scaled_pos = tetrahedron_translated_pos / tetrahedron_scale;

    let sd_tetrahedron_data = sdRecursiveTetrahedron(tetrahedron_scaled_pos);
    let sd_tetrahedron = sd_tetrahedron_data.x * tetrahedron_scale;

    // Columns

    let columns_wrapped_pos = vec3<f32>(
        wrap(q.x, -4.0, 4.0),
        q.y,
        wrap(q.z, -2.0, 2.0),
    );

    let sd_columns = sdSceneColumn(columns_wrapped_pos);

    // Axes

    let sd_axes = sdSceneAxes(q);

    // Plane

    let sd_plane = q.y;

    return SdSceneData(min4(
        sd_tetrahedron,
        sd_columns,
        sd_plane,
        sd_axes,
    ), sd_tetrahedron_data.y);
}

fn sdSceneMaterial(p: vec3<f32>, base_color: vec3<f32>) -> PhongReflectionMaterial {
    if (isCollidingDistance(sdSceneAxes(p)) && length(p) > 0.5) {
        var color: vec3<f32>;
        if (abs(p.x) > 0.5) {
            color = vec3<f32>(0.5 + f32(p.x > 0.0) * 1.0, 0.2, 0.2);
        } else if (abs(p.y) > 0.5) {
            color = vec3<f32>(0.2, 0.5 + f32(p.y > 0.0) * 1.0, 0.2);
        } else {
            color = vec3<f32>(0.2, 0.2, 0.5 + f32(p.z > 0.0) * 1.0);
        }

        return PhongReflectionMaterial(color, vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.9), 1.0);
    }

    return PhongReflectionMaterial(base_color, vec3<f32>(0.0), vec3<f32>(0.7), vec3<f32>(0.05), 30.0);
}

const NORMAL_EPSILON = 0.001;

fn sdSceneNormal(p: vec3<f32>) -> vec3<f32> {
    return normalize(vec3(
        sdScene(vec3<f32>(p.x + NORMAL_EPSILON, p.y, p.z)).sd - sdScene(vec3<f32>(p.x - NORMAL_EPSILON, p.y, p.z)).sd,
        sdScene(vec3<f32>(p.x, p.y + NORMAL_EPSILON, p.z)).sd - sdScene(vec3<f32>(p.x, p.y - NORMAL_EPSILON, p.z)).sd,
        sdScene(vec3<f32>(p.x, p.y, p.z  + NORMAL_EPSILON)).sd - sdScene(vec3<f32>(p.x, p.y, p.z - NORMAL_EPSILON)).sd
    ));
}

const SUN_DIR = vec3<f32>(0.5, 1.0, 3.0);

// Ray Marching

const RAY_MARCHING_MAX_STEP_DEPTH = 2000;
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
    cutoff_reason: u32,
    average_sd_scene_data: SdSceneData,
    minimal_sd_scene_data: SdSceneData,
    maximal_sd_scene_data: SdSceneData,
}

fn isCollidingDistance(sd: f32) -> bool {
    return sd <= RAY_MARCHER_COLLISION_DISTANCE;
}

const APPROX_AO_SAMPLE_COUNT = 5;
const APPROX_AO_SAMPLE_STEP = 0.1;

fn rayMarchHitApproxSoftShadow(ray: Ray, hit: RayMarchHit, sun_dir: vec3<f32>) -> f32 {
    let light_hit = rayMarchWith(Ray(hit.pos, sun_dir), RayMarchOptions(RAY_MARCHER_MAX_DEPTH, true));
    return f32(light_hit.cutoff_reason != CUTOFF_REASON_NONE);
}

fn rayMarchHitApproxAO(ray: Ray, hit: RayMarchHit) -> f32 {
    if (hit.cutoff_reason != CUTOFF_REASON_NONE) {
        return 0.0;
    }

    let normal = sdSceneNormal(hit.pos);

    var total = 0.0;

    for (var i = 1; i < APPROX_AO_SAMPLE_COUNT; i += 1) {
        let delta = f32(i) * APPROX_AO_SAMPLE_STEP;
        let sd = sdScene(hit.pos + normal * delta).sd;
        total += pow(2.0, f32(-i)) * (delta - sd);
    }

    return 1.0 - clamp(5.0 * total, 0.0, 1.0);
}

struct RayMarchOptions {
    depth_limit: f32,
    use_hit_escape: bool,
}

fn rayMarchWith(ray: Ray, options: RayMarchOptions) -> RayMarchHit {
    var depth = f32(options.use_hit_escape) * 0.01;
    var position = ray.origin + depth * ray.direction;
    var shortest_distance = 3.40282346638528859812e+38f;

    var step_depth = 0;
    var cutoff_reason = 0u;

    var sd_scene_data_total = SdSceneData(0.0, 0.0);
    var sd_scene_data_minimal = SdSceneData(3.40282346638528859812e+38f, 3.40282346638528859812e+38f);
    var sd_scene_data_maximal = SdSceneData(-3.40282346638528859812e+38f, -3.40282346638528859812e+38f);

    for (; step_depth < RAY_MARCHING_MAX_STEP_DEPTH; step_depth += 1) {
        let sd_scene_data = sdScene(position);

        sd_scene_data_total.sd += sd_scene_data.sd;
        sd_scene_data_total.iterations += sd_scene_data.iterations;
        sd_scene_data_minimal.sd = min(sd_scene_data_minimal.sd, sd_scene_data.sd);
        sd_scene_data_minimal.iterations = min(sd_scene_data_minimal.iterations, sd_scene_data.iterations);
        sd_scene_data_maximal.sd = max(sd_scene_data_maximal.sd, sd_scene_data.sd);
        sd_scene_data_maximal.iterations = max(sd_scene_data_maximal.iterations, sd_scene_data.iterations);

        let sd = sd_scene_data.sd;

        shortest_distance = min(sd, shortest_distance);

        if (sd <= RAY_MARCHER_COLLISION_DISTANCE * max(1.0, 1.0 + depth * 0.5)) {
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
        position, depth, step_depth, shortest_distance, cutoff_reason, sd_scene_data_average,
        sd_scene_data_minimal, sd_scene_data_maximal
    );
}

fn rayMarch(ray: Ray) -> RayMarchHit {
    return rayMarchWith(ray, RayMarchOptions(RAY_MARCHER_MAX_DEPTH, false));
}

// Rendering

const PIXEL_SAMPLING_RATE = 1;
const PIXEL_SAMPLING_BORDER = 0.4;

fn renderHit(ray: Ray, hit: RayMarchHit) -> vec3<f32> {
    var color: vec3<f32>;
    if (hit.cutoff_reason == CUTOFF_REASON_DISTANCE) {
        color = vec3<f32>(1.0); // mix(vec3<f32>(0.5), vec3<f32>(1.0, 0.2, 0.2), clamp(pow(f32(hit.step_depth) * 0.01, 2.0), 0.0, 1.0));
    } else if (hit.cutoff_reason == CUTOFF_REASON_STEPS) {
        color = vec3<f32>(1.0); //color_map_a(hit.depth * 0.1);
    } else {
        let normal = sdSceneNormal(hit.pos);

        let scene_light = PhongReflectionLight(
            normalize(SUN_DIR) * 10000.0,
            vec3<f32>(1.0, 1.0, 1.0),
            vec3<f32>(1.0, 1.0, 1.0),
        );

        let base_material_color = color_map_a(0.0* hit.depth * 0.001 + 0.0 * (f32(hit.step_depth) / f32(RAY_MARCHING_MAX_STEP_DEPTH)) * 0.1);

        let material_reflect_color = phong_reflect_color(
            frame.cam_pos, hit.pos, normal,
            sdSceneMaterial(hit.pos, base_material_color),
            scene_light,
        );

        let shadow = rayMarchHitApproxSoftShadow(ray, hit, normalize(SUN_DIR));
        let ao = rayMarchHitApproxAO(ray, hit);
        let light = vec3<f32>(mix(shadow, ao, 0.5));

        color = mix(vec3<f32>(0.0), material_reflect_color, light);
    }

    // let iter = hit.average_sd_scene_data.iterations;
    // color = vec3<f32>((color.x + color.y + color.z) / 3.0);
    // color = mix(color, vec3<f32>(1.0), 0.4);
    //color *= color_map_temp(iter / 10.0);

    return color;
}

fn renderRay(ray: Ray) -> vec3<f32> {
    let hit = rayMarch(ray);
    return renderHit(ray, hit);
}

fn renderPixel(texture_coord: vec2<i32>) -> vec3<f32> {
    if (PIXEL_SAMPLING_RATE == 1) {
        return renderRay(Ray(frame.cam_pos, viewportCoordToRayDir(textureCoordToViewportCoord(vec2<f32>(texture_coord)))));
    } else {
        var color = vec3<f32>(0.0);

        for (var i: i32 = 0; i < PIXEL_SAMPLING_RATE; i += 1) {
            for (var j: i32 = 0; j < PIXEL_SAMPLING_RATE; j += 1) {
                let offset = vec2<f32>(
                    -0.5 + PIXEL_SAMPLING_BORDER / 2.0 + (1.0 - PIXEL_SAMPLING_BORDER) * (f32(i) / (f32(PIXEL_SAMPLING_RATE - 1))),
                    -0.5 + PIXEL_SAMPLING_BORDER / 2.0 + (1.0 - PIXEL_SAMPLING_BORDER) * (f32(j) / (f32(PIXEL_SAMPLING_RATE - 1))),
                );

                let sub_pixel_viewport_coord = textureCoordToViewportCoord(vec2<f32>(texture_coord) + offset);
                let sub_pixel_ray_dir = viewportCoordToRayDir(sub_pixel_viewport_coord);
                color += renderRay(
                    Ray(frame.cam_pos, sub_pixel_ray_dir)
                ) / vec3<f32>(f32(PIXEL_SAMPLING_RATE * PIXEL_SAMPLING_RATE));
            }
        }

        return color;
    }
}

@compute @workgroup_size(8, 8, 1)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let texture_coord = invocationIdToTextureCoord(invocation_id);
    let viewport_coord = textureCoordToViewportCoord(vec2<f32>(texture_coord));

    var color = renderPixel(texture_coord);
    // let color = vec3<f32>(viewport_coord * vec2<f32>(0.5) + vec2<f32>(0.5), 0.0);

    if (viewport_coord.y <= -0.9) {
        color = color_map_temp(0.5 * viewport_coord.x + 0.5) * vec3<f32>(f32(viewport_coord.y <= -0.95));
    }

    textureStore(texture, texture_coord, vec4<f32>(color, 1.0));
}