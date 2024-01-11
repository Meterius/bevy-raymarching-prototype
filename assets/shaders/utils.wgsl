const PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164062;

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

fn randomFloat(value: u32) -> f32 {
    return f32(hash(value)) / 4294967295.0;
}

fn max_comp3(v: vec3<f32>) -> f32 {
    return max(v.x, max(v.y, v.z));
}

fn smooth_min(x1: f32, x2: f32, k: f32) -> f32 {
    let h = max(k - abs(x1 - x2), 0.0) / k;
    return min(x1, x2) - h * h * h * k * (1.0 / 6.0);
}

fn euclid_mod(x: f32, n: f32) -> f32 {
    return x - floor(x / n) * n;
}

fn wrap(x: f32, lower: f32, upper: f32) -> f32 {
    let offset = euclid_mod(x - lower, (upper - lower));
    return lower + offset;
}

fn wrap_reflect(x: f32, lower: f32, upper: f32) -> f32 {
    let diff = upper - lower;
    let offset = euclid_mod(x - lower, 2.0 * diff);

    if (offset > diff) {
        return lower + 2.0 * diff - offset;
    } else {
        return lower + offset;
    }
}