fn color_map_cos(x: f32, c1: vec3<f32>, c2: vec3<f32>, c3: vec3<f32>, c4: vec3<f32>) -> vec3<f32> {
    return c1 + c2 * cos(6.28318 * (c3 * x + c4));
}

fn color_map_4_phases(x: f32, c1: vec3<f32>, c2: vec3<f32>, c3: vec3<f32>, c4: vec3<f32>) -> vec3<f32> {
    let count = 4.0;
    let step = 1.0 / (count - 1.0);

    if (x < 0.0) {
        return c1;
    } else if (x < step) {
        return mix(c1, c2, x / step);
    } else if (x < 2.0 * step) {
        return mix(c2, c3, (x - step) / step);
    } else if (x < 3.0 * step) {
      return mix(c3, c4, (x - 2.0 * step) / step);
    } else {
        return c4;
    }
}

// cosine color maps: https://iquilezles.org/articles/palettes/

fn color_map_0(x: f32) -> vec3<f32> {
    return color_map_cos(
        x,
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(1.0, 1.0, 1.0),
        vec3<f32>(0.00, 0.33, 0.67),
    );
}

fn color_map_1(x: f32) -> vec3<f32> {
    return color_map_cos(
        x,
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(1.0, 1.0, 1.0),
        vec3<f32>(0.00, 0.10, 0.20),
    );
}

fn color_map_2(x: f32) -> vec3<f32> {
    return color_map_cos(
        x,
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(1.0, 1.0, 1.0),
        vec3<f32>(0.30, 0.20, 0.20),
    );
}

fn color_map_3(x: f32) -> vec3<f32> {
    return color_map_cos(
        x,
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(1.0, 1.0, 0.5),
        vec3<f32>(0.80, 0.90, 0.30),
    );
}

fn color_map_4(x: f32) -> vec3<f32> {
    return color_map_cos(
        x,
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(1.0, 0.7, 0.4),
        vec3<f32>(0.00, 0.15, 0.20),
    );
}

fn color_map_5(x: f32) -> vec3<f32> {
    return color_map_cos(
        x,
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(2.0, 1.0, 0.0),
        vec3<f32>(0.50, 0.20, 0.25),
    );
}

fn color_map_6(x: f32) -> vec3<f32> {
    return color_map_cos(
        x,
        vec3<f32>(0.8, 0.5, 0.4),
        vec3<f32>(0.2, 0.4, 0.2),
        vec3<f32>(2.0, 1.0, 1.0),
        vec3<f32>(0.00, 0.25, 0.25),
    );
}

// custom

fn color_map_a(x: f32) -> vec3<f32> {
    return color_map_cos(
        x,
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(0.8, 0.8, 0.5),
        vec3<f32>(0.0, 0.2, 0.5),
    );
}


// linear phase color maps

fn color_map_mako(x: f32) -> vec3<f32> {
    return color_map_4_phases(
        x,
        vec3<f32>(0.045, 0.015, 0.221),
        vec3<f32>(0.750, 0.186, 0.357),
        vec3<f32>(0.221, 0.666, 0.675),
        vec3<f32>(0.872, 0.960, 0.897),
    );
}

//

fn color_map_default(x: f32) -> vec3<f32> {
    return color_map_3(x);
}