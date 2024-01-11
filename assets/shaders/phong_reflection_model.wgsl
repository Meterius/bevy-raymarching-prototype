const I_A = vec3<f32>(1.0, 1.0, 1.0);

struct PhongReflectionLight {
    pos: vec3<f32>,
    i_s: vec3<f32>,
    i_d: vec3<f32>,
}

struct PhongReflectionMaterial {
    color: vec3<f32>,
    k_s: vec3<f32>,
    k_d: vec3<f32>,
    k_a: vec3<f32>,
    a: f32,
}

fn mix_material(m1: PhongReflectionMaterial, m2: PhongReflectionMaterial, fac: f32) -> PhongReflectionMaterial {
    return PhongReflectionMaterial(
        mix(m1.color, m2.color, fac),
        mix(m1.k_s, m2.k_s, fac),
        mix(m1.k_d, m2.k_d, fac),
        mix(m1.k_a, m2.k_a, fac),
        mix(m1.a, m2.a, fac),
    );
}

fn phong_reflect_color(
    origin: vec3<f32>,
    hit: vec3<f32>,
    normal: vec3<f32>,
    material: PhongReflectionMaterial,
) -> vec3<f32> {
    let light = PhongReflectionLight(
        origin,
        vec3<f32>(1.0),
        vec3<f32>(1.0),
    );

    var color = material.color * material.k_a * I_A;
    let v = normalize(origin - hit);

    let l_m = normalize(light.pos - hit);
    let r_m = 2.0 * dot(l_m, normal) * normal - l_m;
    color += material.color * material.k_d * max(0.0, dot(l_m, normal)) * light.i_d + material.k_s * pow(max(0.0, dot(r_m, v)), material.a) * light.i_s;

    return color;
}
