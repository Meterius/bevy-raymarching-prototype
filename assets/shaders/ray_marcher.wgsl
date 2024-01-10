@group(0) @binding(0) var texture: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(1) var<uniform> frame: RayMarcherFrameData;

struct RayMarcherFrameData {
    time: f32,
}

const texture_size: vec2<i32> = vec2<i32>(2560, 1440);
const camera_plane_width: f32 = 2.0;

const ray_marcher_max_steps = 80;
const ray_marcher_hit_cutoff_dist = 0.001;
const scene_cutoff_dist = 100.0;
const pixel_sampling_rate = 1;

fn hash(value: u32) -> u32 {
    var state = value;
    state = state ^ 2747636419u;
    state = state * 2654435769u;
    state = state ^ state >> 16u;
    state = state * 2654435769u;
    state = state ^ state >> 16u;
    state = state * 2654435769u;
    return state;
}

fn cameraPlane() -> vec2<f32> {
    return vec2<f32>(camera_plane_width, camera_plane_width * (f32(texture_size.y) / f32(texture_size.x)));
}

fn randomFloat(value: u32) -> f32 {
    return f32(hash(value)) / 4294967295.0;
}

fn invocationIdToTextureCoord(invocation_id: vec3<u32>) -> vec2<i32> {
    return vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
}

fn textureCoordToViewportCoord(texture_coord: vec2<i32>) -> vec2<f32> {
    return vec2<f32>(2.0) * vec2<f32>(texture_coord) / vec2<f32>(texture_size) - vec2<f32>(1.0);
}

fn viewportCoordToRayDir(viewport_coord: vec2<f32>) -> vec3<f32> {
    return normalize(vec3<f32>(viewport_coord, 1.0) * vec3<f32>(cameraPlane(), 1.0));
}

// SD Primitives

fn sdUnitSphere(p: vec3<f32>) -> f32 {
    return length(p) - 1.0;
}

// SD Operators

fn sdPreTranslate(p: vec3<f32>, translation: vec3<f32>) -> vec3<f32> {
    return p - translation;
}

fn sdPostTranslate(sd: f32, translation: vec3<f32>) -> f32 {
    return sd;
}

fn sdPreScale(p: vec3<f32>, scale_origin: vec3<f32>, scale: vec3<f32>) -> vec3<f32> {
    return (p - scale_origin) * scale + scale_origin;
}

fn sdPostScale(sd: f32, scale_origin: vec3<f32>, scale: vec3<f32>) -> f32 {
    return sd / scale.x;
}

fn sdPreTransformUnit(p: vec3<f32>, translation: vec3<f32>, scale: vec3<f32>) -> vec3<f32> {
    let p2 = sdPreScale(p, vec3<f32>(0.0), scale);
    let p3 = sdPreTranslate(p, translation);
    return p3;
}

fn sdPostTransformUnit(sd3: f32, translation: vec3<f32>, scale: vec3<f32>) -> f32 {
    let sd2 = sdPostTranslate(sd3, translation);
    let sd = sdPostScale(sd2, vec3<f32>(0.0), scale);
    return sd;
}

fn sdPostUnion(sd1: f32, sd2: f32) -> f32 {
    return min(sd1, sd2);
}

fn sdPostIntersect(sd1: f32, sd2: f32) -> f32 {
    return max(sd1, sd2);
}

// Scene

fn sdScene(p: vec3<f32>) -> f32 {
    // sphere1

    let tSphere1 = vec3<f32>(0.0, 0.0, 3.0);
    let sSphere1 = vec3<f32>(4.0);
    let pSphere1 = sdPreTransformUnit(p, tSphere1, sSphere1);
    //let sdSphere1 = sdPostTransformUnit(sdUnitSphere(pSphere1), tSphere1, sSphere1);
    let sdSphere1 = sdUnitSphere((p - vec3(0.0, 0.0, 3.0)) * vec3(1.25));

    return sdSphere1;
}

// Ray Marching

fn rayMarch(ray_pos: vec3<f32>, ray_dir: vec3<f32>) -> vec3<f32> {
    var curr_ray_pos = ray_pos;
    var depth = length(curr_ray_pos - ray_pos);
    var cutoff = true;

    var color = vec3<f32>(0.0);

    for (var i: i32 = 0; i < ray_marcher_max_steps; i += 1) {
        let sd = sdScene(curr_ray_pos);
        curr_ray_pos += sd * ray_dir;
        depth += sd;

        let i_fac = f32(i - 1) / f32(ray_marcher_max_steps);

        if (sd <= ray_marcher_hit_cutoff_dist) {
            cutoff = false;
            break;
        } else if (depth > scene_cutoff_dist) {
            break;
        }
    }

    if (cutoff) {
        color = vec3<f32>(1.0);
    } else {
        color = vec3<f32>(0.0);
    }

    return color;
}

fn rayMarchPixel(ray_pos: vec3<f32>, viewport_coord: vec2<f32>) -> vec3<f32> {
    var color = vec3<f32>(0.0);

    for (var i: i32 = 0; i < pixel_sampling_rate; i += 1) {
        for (var j: i32 = 0; j < pixel_sampling_rate; j += 1) {
            let viewport_sub_pixel_coord = vec2<f32>(
                viewport_coord.x + 2.0 * (- 0.5 + f32(i) / max(f32(pixel_sampling_rate - 1), 1.0)) / f32(texture_size.x),
                viewport_coord.y + 2.0 * (- 0.5 + f32(j) / max(f32(pixel_sampling_rate - 1), 1.0)) / f32(texture_size.y),
            );
            let sub_pixel_ray_dir = viewportCoordToRayDir(viewport_sub_pixel_coord);
            color += rayMarch(ray_pos, sub_pixel_ray_dir) / vec3<f32>(f32(pixel_sampling_rate * pixel_sampling_rate));
        }
    }

    return color;
}

fn euclid_mod(x: f32, n: f32) -> f32 {
    return x - floor(x * n) / n;
}

@compute @workgroup_size(8, 8, 1)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let texture_coord = invocationIdToTextureCoord(invocation_id);
    let viewport_coord = textureCoordToViewportCoord(texture_coord);

    let ray_pos = vec3<f32>(0.0);

    var color = vec3<f32>(euclid_mod(frame.time, 1.0)); // rayMarchPixel(ray_pos, viewport_coord);

    textureStore(texture, texture_coord, vec4<f32>(color, 1.0));
}