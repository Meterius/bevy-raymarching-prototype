#import "shaders/compiled/utils.wgsl"::{PI, max_comp3, euclid_mod, smooth_min, wrap, wrap_reflect, min4, min5, min3}
#import "shaders/compiled/color.wgsl"::{color_map_default, color_map_a}
#import "shaders/compiled/phong_reflection_model.wgsl"::{PhongReflectionMaterial, mix_material, phong_reflect_color, PhongReflectionLight}
#import "shaders/compiled/signed_distance.wgsl"::{sdSphere, sdUnion, sdSmoothUnion, sdRecursiveTetrahedron}

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
}

//

const SCENE_LIGHT = PhongReflectionLight(
    vec3<f32>(3.0, 4.0, 3.0),
    vec3<f32>(1.0, 1.0, 1.0),
    vec3<f32>(1.0, 1.0, 1.0),
);

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

fn sdSceneLight(p: vec3<f32>) -> f32 {
    return sdSphere(p, SCENE_LIGHT.pos, 0.1);
}

fn sdSceneAxes(p: vec3<f32>) -> f32 {
    return min3(
        sdSphere(vec3<f32>(p.x, wrap(p.y, -0.5, 0.5), p.z), vec3<f32>(0.0), 0.1),
        sdSphere(vec3<f32>(wrap(p.x, -0.5, 0.5), p.y, p.z), vec3<f32>(0.0), 0.1),
        sdSphere(vec3<f32>(p.x, p.y, wrap(p.z, -0.5, 0.5)), vec3<f32>(0.0), 0.1),
    );
}

fn sdScene(p: vec3<f32>) -> f32 {
    var q = p;

    let scale = 200.0;

    let tq = vec3<f32>(
        wrap(q.x, -scale*2.0, scale*2.0),
        q.y,
        wrap(q.z, -scale*2.0, scale*2.0),
    );
    let t = sdRecursiveTetrahedron(tq / scale - vec3<f32>(0.0, 1.0, 0.0)) * scale;

    return min3(
        t,
        sdSceneAxes(q),
        p.y,
    );
}

fn sdSceneMaterial(p: vec3<f32>, base_color: vec3<f32>) -> PhongReflectionMaterial {
    if (isCollidingDistance(sdSceneLight(p))) {
        return PhongReflectionMaterial(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.9), 1.0);
    } else if (isCollidingDistance(sdSceneAxes(p)) && length(p) > 0.5) {
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
        sdScene(vec3<f32>(p.x + NORMAL_EPSILON, p.y, p.z)) - sdScene(vec3<f32>(p.x - NORMAL_EPSILON, p.y, p.z)),
        sdScene(vec3<f32>(p.x, p.y + NORMAL_EPSILON, p.z)) - sdScene(vec3<f32>(p.x, p.y - NORMAL_EPSILON, p.z)),
        sdScene(vec3<f32>(p.x, p.y, p.z  + NORMAL_EPSILON)) - sdScene(vec3<f32>(p.x, p.y, p.z - NORMAL_EPSILON))
    ));
}

const SUN_DIR = vec3<f32>(0.5, 1.0, 3.0);

fn canSeeSun(p: vec3<f32>) -> bool {
    let dir = normalize(SUN_DIR);
    let hit = rayMarch(recastRayFromHit(Ray(p, dir)));
    return hit.cutoff_reason != CUTOFF_REASON_NONE;
}

fn canSeeSceneLight(p: vec3<f32>) -> bool {
    let dir = SCENE_LIGHT.pos - p;
    let dir_len = length(dir);
    let hit = rayMarchWithLimitedDepth(recastRayFromHit(Ray(p, dir / dir_len)), dir_len);
    return hit.depth >= dir_len;
}

// Ray Marching

const RAY_MARCHING_MAX_STEP_DEPTH = 2000;
const RAY_MARCHER_COLLISION_DISTANCE = 0.0001;
const RAY_MARCHER_RECAST_SKIP = 0.01;
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
    cutoff_reason: u32,
}

fn isCollidingDistance(sd: f32) -> bool {
    return sd <= RAY_MARCHER_COLLISION_DISTANCE;
}

fn recastRayFromHit(ray: Ray) -> Ray {
    return Ray(ray.origin + ray.direction * RAY_MARCHER_RECAST_SKIP, ray.direction);
}

fn rayMarchWithLimitedDepth(ray: Ray, depth_limit: f32) -> RayMarchHit {
    var position = ray.origin;
    var depth = 0.0;
    var shortest_distance = 3.40282346638528859812e+38f;

    var step_depth = 0;
    var cutoff_reason = 0u;

    for (; step_depth < RAY_MARCHING_MAX_STEP_DEPTH; step_depth += 1) {
        let sd = sdScene(position);
        shortest_distance = min(sd, shortest_distance);

        if (sd <= RAY_MARCHER_COLLISION_DISTANCE) {
            break;
        }

        depth += sd;
        let ray_offset = vec3<f32>(0.0); // vec3<f32>(0.0, sin(depth) * 0.35, 0.0);
        position = ray.origin + depth * ray.direction + ray_offset;

        if (depth >= depth_limit) {
            cutoff_reason = CUTOFF_REASON_DISTANCE;
            break;
        }
    }

    if (step_depth == RAY_MARCHING_MAX_STEP_DEPTH) {
        cutoff_reason = CUTOFF_REASON_STEPS;
    }

    return RayMarchHit(
        position, depth, step_depth, shortest_distance, cutoff_reason,
    );
}

fn rayMarch(ray: Ray) -> RayMarchHit {
    return rayMarchWithLimitedDepth(ray, RAY_MARCHER_MAX_DEPTH);
}

// Rendering

const PIXEL_SAMPLING_RATE = 1;
const PIXEL_SAMPLING_BORDER = 0.4;

fn renderHit(hit: RayMarchHit) -> vec3<f32> {
    if (hit.cutoff_reason == CUTOFF_REASON_DISTANCE) {
        return vec3<f32>(1.0);
    } else if (hit.cutoff_reason == CUTOFF_REASON_STEPS) {
        return vec3<f32>(1.0);
    } else {
        let normal = sdSceneNormal(hit.pos);

        let color1 = phong_reflect_color(
            frame.cam_pos, hit.pos, normal,
            sdSceneMaterial(hit.pos, color_map_a(hit.depth * 0.001 + (f32(hit.step_depth) / f32(RAY_MARCHING_MAX_STEP_DEPTH)) * 0.1)),
            SCENE_LIGHT,
        );

        let color2 = mix(color1, vec3<f32>(0.0), min(f32(hit.step_depth) / 100.0, 0.7));
        // let color3 = mix(color1, vec3<f32>(0.0), 0.7 * (1.0 - f32(canSeeSun(hit.pos))));

        return mix(color2, vec3<f32>(1.0), min(hit.depth / 1000.0, 1.0));
    }
}

fn renderRay(ray: Ray) -> vec3<f32> {
    let hit = rayMarch(ray);
    return renderHit(hit);
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
        // color = color_map_default(0.5 * viewport_coord.x + 0.5);
    }

    textureStore(texture, texture_coord, vec4<f32>(color, 1.0));
}