#import "shaders/compiled/utils.wgsl"::{max_comp3, smooth_min}

// SD Primitives

fn sdOctahedron(p: vec3<f32>, op: vec3<f32>, s: f32) -> f32 {
    let q = abs(p - op);
    let m = q.x + q.y + q.z - s;

    var t = vec3<f32>(0.0);

    if (3.0 * q.x < m) {
        t = vec3<f32>(q.x, q.y, q.z);
    } else if (3.0 * q.y < m) {
        t = vec3<f32>(q.y, q.z, q.x);
    } else if (3.0 * q.z < m) {
        t = vec3<f32>(q.z, q.x, q.y);
    } else {
        return m * 0.57735027;
    }

    let k = clamp(0.5 * (t.z - t.y + s), 0.0, s);
    return length(vec3<f32>(t.x, t.y - s + k, t.z - k));
}

fn sdOctahedronApprox(p: vec3<f32>, op: vec3<f32>, s: f32) -> f32 {
    let q = abs(p - op);
    return (q.x+q.y+q.z-s)*0.57735027;
}

fn sdSphere(p: vec3<f32>, sp: vec3<f32>, r: f32) -> f32 {
    return length(p - sp) - r;
}

fn sdBox(p: vec3<f32>, bp: vec3<f32>, bs: vec3<f32>) -> f32 {
    let q = abs(p - bp) - bs;
    let udst = length(max(q, vec3<f32>(0.0)));
    let idst = max_comp3(min(q, vec3<f32>(0.0)));
    return udst + idst;
}

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

fn sdPostSmoothUnion(sd1: f32, sd2: f32, k: f32) -> f32 {
    return smooth_min(sd1, sd2, k);
}

fn sdPostIntersect(sd1: f32, sd2: f32) -> f32 {
    return max(sd1, sd2);
}

fn sdPostInverse(sd1: f32) -> f32 {
    return -sd1;
}

fn sdPostDifference(sd1: f32, sd2: f32) -> f32 {
    return max(sd1, -sd2);
}
