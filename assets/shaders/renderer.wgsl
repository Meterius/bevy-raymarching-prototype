#import "shaders/compiled/utils.wgsl"::{PI, max_comp3, euclid_mod, smooth_min, wrap, wrap_reflect}
#import "shaders/compiled/color.wgsl"::{color_map_default, color_map_a}
#import "shaders/compiled/phong_reflection_model.wgsl"::{PhongReflectionMaterial, mix_material, phong_reflect_color}
#import "shaders/compiled/signed_distance.wgsl"::{sdSphere, sdUnion, sdSmoothUnion}

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

const ray_marcher_max_steps = 100;
const ray_marcher_hit_cutoff_dist = 0.01;
const scene_cutoff_dist = 1000.0;
const pixel_sampling_rate = 1;

fn invocationIdToTextureCoord(invocation_id: vec3<u32>) -> vec2<i32> {
    return vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
}

fn textureCoordToViewportCoord(texture_coord: vec2<i32>) -> vec2<f32> {
    let flipped = vec2<f32>(2.0) * vec2<f32>(texture_coord) / vec2<f32>(frame.texture_size) - vec2<f32>(1.0);
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

fn sdScene(p: vec3<f32>) -> f32 {
    // sphere1

    var q = p;
    q.x = wrap(q.x, -0.75, 0.75);
    q.y = wrap(q.y, -3.0, 3.0);
    q.z = wrap(q.z, -3.0, 3.0);

    return sdSphere(q, vec3<f32>(0.0, 0.0, 0.0), 0.5);
    // let sd2 = sdBox(q, vec3(0.0, 0.5, 0.0), vec3(0.25, 1.0, 0.25));
}

fn sdSceneMaterial(p: vec3<f32>, depth: f32) -> PhongReflectionMaterial {
    return PhongReflectionMaterial(color_map_default(depth / 100.0), vec3<f32>(1.0), vec3<f32>(0.7), vec3<f32>(0.05), 30.0);
    /*let m1 = PhongReflectionMaterial(vec3<f32>(1.0, 0.1, 0.1), vec3<f32>(1.0), vec3<f32>(0.7), vec3<f32>(0.05), 30.0);
    let m2 = PhongReflectionMaterial(vec3<f32>(0.1, 0.1, 1.0), vec3<f32>(0.6), vec3<f32>(0.7), vec3<f32>(0.05), 30.0);
    let m3 = PhongReflectionMaterial(vec3<f32>(0.1, 1.0, 0.1), vec3<f32>(0.6), vec3<f32>(3.0), vec3<f32>(0.5), 30.0);

    var q = p;
    q.x = wrap(q.x, -3.0, 3.0);
    q.z = wrap(q.z, -2.0, 2.0);
    q.y = wrap(q.y, -3.0, 3.0);

    let sd1 = sdSphere(q, vec3(sin(frame.time * 2.0 * pi / 5.0) * 2.0, 0.5, 0.0), 0.5);
    let sd2 = sdBox(q, vec3(0.0, 0.5, 0.0), vec3(0.25, 2.0, 0.25));

    let grid_dist = 0.75;
    let grid_width = 0.05;
    let grid = euclid_mod(q.x, grid_dist);
    let f_grid = max(max(1.0 - grid / grid_width, (grid - grid_dist) / grid_width + 1.0), 0.0);
    let f_grid_mapped = 1.0 - pow(1.0 - f_grid, 1.25);

    let f = 1.0 - sd2 / (sd1 + sd2);

    return mix_material(mix_material(m1, m2, f), m3, f_grid_mapped);*/
}

const normal_eps = 0.001;

fn sdSceneNormal(p: vec3<f32>) -> vec3<f32> {
    return normalize(vec3(
        sdScene(vec3<f32>(p.x + normal_eps, p.y, p.z)) - sdScene(vec3<f32>(p.x - normal_eps, p.y, p.z)),
        sdScene(vec3<f32>(p.x, p.y + normal_eps, p.z)) - sdScene(vec3<f32>(p.x, p.y - normal_eps, p.z)),
        sdScene(vec3<f32>(p.x, p.y, p.z  + normal_eps)) - sdScene(vec3<f32>(p.x, p.y, p.z - normal_eps))
    ));
}

// Ray Marching

fn rayMarch(ray_pos: vec3<f32>, ray_dir: vec3<f32>) -> vec3<f32> {
    var curr_ray_pos = ray_pos;
    var depth = 0.0;
    var shortest_distance = 3.40282346638528859812e+38f;

    var step_count: i32 = 0;
    var step_cutoff = true;
    var dist_cutoff = false;

    var color = vec3<f32>(0.0);

    for (; step_count < ray_marcher_max_steps; step_count += 1) {
        let sd = sdScene(curr_ray_pos);
        shortest_distance = min(sd, shortest_distance);

        if (sd <= ray_marcher_hit_cutoff_dist) {
            step_cutoff = false;
            break;
        }

        depth += sd;
        curr_ray_pos = ray_pos + depth * ray_dir + vec3<f32>(0.0, sin(depth) * 0.35, 0.0);

        if (depth > scene_cutoff_dist) {
            step_cutoff = false;
            dist_cutoff = true;
            break;
        }
    }

    if (dist_cutoff) {
        // color = vec3<f32>(1.0);
        color = vec3<f32>(1.0);
    } else if (step_cutoff) {
        color = vec3<f32>(1.0);
    } else {
        let normal = sdSceneNormal(curr_ray_pos);
        color = phong_reflect_color(frame.cam_pos, curr_ray_pos, normal, sdSceneMaterial(curr_ray_pos, depth));
        // color = vec3<f32>(depth * 0.001);
    }

    color = color_map_a(depth * 0.01 + (f32(step_count) / f32(ray_marcher_max_steps)) * 0.1);

    /*color = mix(mix(
        color_map_default(1.0 - pow(1.0 - shortest_distance, 4.0)),
        color_map_default(f32(step_count) * 0.01),
        0.25,
    ), color_map_default(depth * 0.001), 0.5);*/


    return color;
}

fn rayMarchPixel(viewport_coord: vec2<f32>) -> vec3<f32> {
    var color = vec3<f32>(0.0);

    for (var i: i32 = 0; i < pixel_sampling_rate; i += 1) {
        for (var j: i32 = 0; j < pixel_sampling_rate; j += 1) {
            /*let viewport_sub_pixel_coord = vec2<f32>(
                viewport_coord.x + 2.0 * (- 0.5 + f32(i) / max(f32(pixel_sampling_rate - 1), 1.0)) / f32(texture_size.x),
                viewport_coord.y + 2.0 * (- 0.5 + f32(j) / max(f32(pixel_sampling_rate - 1), 1.0)) / f32(texture_size.y),
            );
            let sub_pixel_ray_dir = viewportCoordToRayDir(viewport_sub_pixel_coord);*/
            let sub_pixel_ray_dir = viewportCoordToRayDir(viewport_coord);
            color += rayMarch(frame.cam_pos, sub_pixel_ray_dir) / vec3<f32>(f32(pixel_sampling_rate * pixel_sampling_rate));
        }
    }

    return color;
}

@compute @workgroup_size(8, 8, 1)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let texture_coord = invocationIdToTextureCoord(invocation_id);
    let viewport_coord = textureCoordToViewportCoord(texture_coord);

    var color = rayMarchPixel(viewport_coord);
    // let color = vec3<f32>(viewport_coord * vec2<f32>(0.5) + vec2<f32>(0.5), 0.0);

    if (viewport_coord.y <= -0.9) {
        color = color_map_default(0.5 * viewport_coord.x + 0.5);
    }

    textureStore(texture, texture_coord, vec4<f32>(color, 1.0));
}