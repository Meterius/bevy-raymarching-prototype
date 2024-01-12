#import "shaders/compiled/utils.wgsl"::{max_comp3, smooth_min, euclid_mod, max3, max4, max5, min3, min4, min5}

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

fn sdVertexPlane(p: vec3<f32>, n: vec3<f32>, d: f32) -> f32 {
    return dot(p, n) - d;
}

fn sdVertexPlaneB(p: vec3<f32>, n: vec3<f32>, b: vec3<f32>) -> f32 {
    return dot(p, n) - dot(b, n);
}

// SD Complexes

const REC_TETR_ITER: i32 = 10;
const REC_TETR_SCALE: f32 = 1.0;
const REC_TETR_OFFSET: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);

fn sdTetrahedron(p: vec3<f32>) -> f32 {
    var q = p;

    var i = 1;
    for (; i <= REC_TETR_ITER; i += 1) {
        q = q * vec3<f32>(2.0) - vec3<f32>(1.0);
        q = sdPreMirrorB(q, normalize(vec3<f32>(1.0, 0.0, -1.0)), vec3<f32>(0.0));
        q = sdPreMirrorB(q, normalize(vec3<f32>(-1.0, 1.0, 0.0)), vec3<f32>(0.0, 0.0, 0.0));
        q = sdPreMirrorB(q, normalize(vec3<f32>(1.0, 0.0, 1.0)), vec3<f32>(-1.0, 0.0, -1.0));
    }

    let a = max4(
        sdVertexPlaneB(q, normalize(vec3<f32>(1.0, 1.0, -1.0)), vec3<f32>(1.0, 1.0, 1.0)),
        sdVertexPlaneB(q, normalize(vec3<f32>(-1.0, 1.0, 1.0)), vec3<f32>(1.0, 1.0, 1.0)),
        sdVertexPlaneB(q, normalize(vec3<f32>(1.0, -1.0, 1.0)), vec3<f32>(1.0, -1.0, -1.0)),
        sdVertexPlaneB(q, normalize(vec3<f32>(-1.0, -1.0, -1.0)), vec3<f32>(1.0, -1.0, -1.0)),
    ) * pow(2.0, -f32(REC_TETR_ITER));

    return a;

    /*let b = min5(
        sdSphere(p, vec3<f32>(0.0, 0.0, 0.0), 0.2),
        sdSphere(p, vec3<f32>(1.0, 1.0, 1.0), 0.1),
        sdSphere(p, vec3<f32>(-1.0, -1.0, 1.0), 0.1),
        sdSphere(p, vec3<f32>(1.0, -1.0, -1.0), 0.1),
        sdSphere(p, vec3<f32>(-1.0, 1.0, -1.0), 0.1),
    );

    return min(a, b);*/
}

fn sdPreMirrorB(p: vec3<f32>, n: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let dist = sdVertexPlaneB(p, n, b);
    if (dist <= 0.0) {
        return p - 2.0 * dist * n;
    } else {
        return p;
    }
}

fn sdRecursiveTetrahedron(p: vec3<f32>) -> f32 {
    return sdTetrahedron(p);
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
