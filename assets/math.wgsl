// ---------------- complex ---------------- //

fn cAdd(c1: vec2<f32>, c2: vec2<f32>) -> vec2<f32> {
    return c1 + c2;
}

fn cSub(c1: vec2<f32>, c2: vec2<f32>) -> vec2<f32> {
    return c1 - c2;
}

fn cMul(c1: vec2<f32>, c2: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        c1.x * c2.x - c1.y * c2.y,
        c1.y * c2.x + c1.x * c2.y
    );
}

fn cConj(c: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(c.x, -c.y);
}

fn cNorm(c: vec2<f32>) -> f32 {
    return length(c);
}

fn cInv(c: vec2<f32>) -> vec2<f32> {
    return cConj(c) / (cNorm(c) * cNorm(c));
}

fn cDiv(c1: vec2<f32>, c2: vec2<f32>) -> vec2<f32> {
    return cMul(c1, cInv(c2));
}

fn cPow(c: vec2<f32>, n: i32) -> vec2<f32> {
    var p: vec2<f32> = vec2<f32>(1.0, 0.0);
    for (var i: i32 = 0; i < n; i = i + 1) {
        p = cMul(p, c);
    }
    return p;
}

// ---------------- quaternion ---------------- //

fn qAdd(q1: vec4<f32>, q2: vec4<f32>) -> vec4<f32> {
    return q1 + q2;
}

fn qSub(q1: vec4<f32>, q2: vec4<f32>) -> vec4<f32> {
    return q1 - q2;
}

fn qMul(q1: vec4<f32>, q2: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        q1.x * q2.x - dot(q1.yzw, q2.yzw),
        q2.x * q1.yzw + q1.x * q2.yzw + cross(q1.yzw, q2.yzw)
    );
}

fn qConj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(q.x, -q.yzw);
}

fn qNorm(q: vec4<f32>) -> f32 {
    return length(q);
}

fn qInv(q: vec4<f32>) -> vec4<f32> {
    return qConj(q) / (qNorm(q) * qNorm(q));
}

fn qDiv(q1: vec4<f32>, q2: vec4<f32>) -> vec4<f32> {
    return qMul(q1, qInv(q2));
}

fn qPow(q: vec4<f32>, n: i32) -> vec4<f32> {
    var p: vec4<f32> = vec4<f32>(1.0, vec3<f32>(0.0));
    for (var i: i32 = 0; i < n; i = i + 1) {
        p = qMul(p, q);
    }
    return p;
}

// ---------------- dual ---------------- //

struct DualQ {
    q: vec4<f32>,
    d: vec4<f32>,
};

fn dqAdd(dq1: DualQ, dq2: DualQ) -> DualQ {
    return DualQ(qAdd(dq1.q, dq2.q), qAdd(dq1.d, dq2.d));
}

fn dqSub(dq1: DualQ, dq2: DualQ) -> DualQ {
    return DualQ(qSub(dq1.q, dq2.q), qSub(dq1.d, dq2.d));
}

fn dqMul(dq1: DualQ, dq2: DualQ) -> DualQ {
    return DualQ(qMul(dq1.q, dq2.q), qAdd(qMul(dq1.d, dq2.q), qMul(dq1.q, dq2.d)));
}

fn dqDiv(dq1: DualQ, dq2: DualQ) -> DualQ {
    return DualQ(qDiv(dq1.q, dq2.q), qDiv(qSub(qMul(dq1.d, dq2.q), qMul(dq1.q, dq2.d)), qMul(dq2.q, dq2.q)));
}

fn dqPow(dq: DualQ, n: i32) -> DualQ {
    var dp: DualQ = DualQ(vec4<f32>(1.0, vec3<f32>(0.0)), vec4<f32>(0.0, vec3<f32>(0.0)));
    for (var i: i32 = 0; i < n; i = i + 1) {
        dp = dqMul(dp, dq);
    }
    return dp;
}
