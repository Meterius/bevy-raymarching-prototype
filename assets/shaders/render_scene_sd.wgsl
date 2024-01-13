#import "shaders/compiled/utils.wgsl"::{min3, min4, min5, wrap, wrap_cell, min_comp3, MAX_POSITIVE_F32}
#import "shaders/compiled/phong_reflection_model.wgsl"::{PhongReflectionMaterial}
#import "shaders/compiled/signed_distance.wgsl"::{sdSphere, sdUnion, sdPostSmoothUnion, sdRecursiveTetrahedron, sdBox, sdPreCheapBend, sdPreMirrorB}

// Runtime Scene

const SD_RUNTIME_SCENE_NODES = 8;

var<private> rs_compute_nodes: array<RsComputeNode, SD_RUNTIME_SCENE_NODES> = array<RsComputeNode, SD_RUNTIME_SCENE_NODES>();

struct RsComputeNode {
    position: vec3<f32>,
    sd: f32,
}

fn sd_runtime_scene(p: vec3<f32>) -> f32 {
    var sd = MAX_POSITIVE_F32;

    if (true) {
        return sd;
    }

    rs_compute_nodes[SD_SCENE.compound_count].position = p;

    for (var i = 0; i < SD_SCENE.compound_count; i++) {
        rs_compute_nodes[i].position = (
            rs_compute_nodes[SD_SCENE.compounds[i].parent].position
            - SD_SCENE.compounds[i].pre_translation
        ) / SD_SCENE.compounds[i].pre_scale;
    }

    for (var i = 0; i < SD_SCENE.primitive_count; i++) {
        if SD_SCENE.primitives[i].use_sphere == 0 {
            rs_compute_nodes[
                SD_SCENE.primitives[i].container
            ].sd = sdBox(
                rs_compute_nodes[
                    SD_SCENE.primitives[i].container
                ].position, vec3<f32>(0.0), SD_SCENE.primitives[i].block,
            );
        } else {
            rs_compute_nodes[
                SD_SCENE.primitives[i].container
            ].sd = sdSphere(
                rs_compute_nodes[
                    SD_SCENE.primitives[i].container
                ].position, vec3<f32>(0.0), SD_SCENE.primitives[i].sphere,
            );
        }
    }

    for (var i = SD_SCENE.compound_count - 1; i >= 0; i--) {
        rs_compute_nodes[i].sd = min(
            rs_compute_nodes[ SD_SCENE.compounds[i].children[0] ].sd,
            rs_compute_nodes[ SD_SCENE.compounds[i].children[1] ].sd
        ) * SD_SCENE.compounds[i].post_scale;
    }

    return rs_compute_nodes[0].sd;
}

// Scene

fn sd_scene_axes(p: vec3<f32>) -> f32 {
    return min3(
        sdSphere(vec3<f32>(p.x, wrap(p.y, -0.5, 0.5), p.z), vec3<f32>(0.0), 0.1),
        sdSphere(vec3<f32>(wrap(p.x, -0.5, 0.5), p.y, p.z), vec3<f32>(0.0), 0.1),
        sdSphere(vec3<f32>(p.x, p.y, wrap(p.z, -0.5, 0.5)), vec3<f32>(0.0), 0.1),
    );
}

fn sd_scene_column(p: vec3<f32>, cell: vec3<f32>) -> f32 {
    return sdPostSmoothUnion(
        min3(
            sdBox(p, vec3<f32>(0.0, 3.0, 0.0), vec3<f32>(0.5, 3.0, 0.5)),
            sdBox(p, vec3<f32>(0.0, 5.0, 0.0), vec3<f32>(0.75, 0.15, 0.75)),
            sdBox(p, vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.7, 1.0, 0.7)),
        ),
        sdSphere(p, vec3<f32>(0.0, 10.0 + (
            2.0 * (
                sin(2.0 * PI * (cell.x * 0.1 + GLOBALS.time) / 10.0)
                + sin(2.0 * PI * (cell.z * 0.1 + GLOBALS.time * 4.0) / 15.0)
            )
        ), 0.0), 0.5),
        1.1,
    );
}

fn sd_scene_column_pattern(p: vec3<f32>, grid_gap: vec2<f32>) -> f32 {
    let q = p;
    let columns_cell = vec3<f32>(
        wrap_cell(q.x, -grid_gap.x, grid_gap.x),
        q.y,
        wrap_cell(q.z, -grid_gap.y, grid_gap.y),
    );

    let columns_wrapped_pos = vec3<f32>(
        wrap(q.x, -grid_gap.x, grid_gap.x),
        q.y,
        wrap(q.z, -grid_gap.y, grid_gap.y),
    );

    return sdPostSmoothUnion(sdPostSmoothUnion(
        sd_scene_column(columns_wrapped_pos + vec3<f32>(2.0 * grid_gap.x, 0.0, 0.0), vec3<f32>(grid_gap, 1.0) * (columns_cell + vec3<f32>(1.0, 0.0, 0.0))),
        sd_scene_column(columns_wrapped_pos - vec3<f32>(2.0 * grid_gap.x, 0.0, 0.0), vec3<f32>(grid_gap, 1.0) * (columns_cell - vec3<f32>(1.0, 0.0, 0.0))),
        2.0,
    ), sd_scene_column(columns_wrapped_pos, columns_cell * vec3<f32>(grid_gap, 1.0)), 4.0);
}

struct SdSceneData {
    sd: f32,
    iterations: f32,
}

fn sd_scene(p: vec3<f32>) -> SdSceneData {
    var q = p;

    // Tetrahedron
    let tetrahedron_scale = 400.0;
    let tetrahedron_wrapped_pos = mix(vec3<f32>(
        wrap(q.x, -tetrahedron_scale*2.0, tetrahedron_scale*2.0),
        q.y,
        wrap(q.z, -tetrahedron_scale*2.0, tetrahedron_scale*2.0),
    ), q, 1.0);

    let tetrahedron_translated_pos = tetrahedron_wrapped_pos - vec3<f32>(0.0, tetrahedron_scale + 25.0, 0.0);
    let tetrahedron_scaled_pos = tetrahedron_translated_pos / tetrahedron_scale;

    let sd_tetrahedron_data = sdRecursiveTetrahedron(tetrahedron_scaled_pos);
    let sd_tetrahedron = sd_tetrahedron_data.x * tetrahedron_scale;

    // Columns

    let sd_columns = sd_scene_column_pattern(q, vec2<f32>(1.0, 1.0));
    let sd_large_columns = sd_scene_column_pattern(q / 30.0 + vec3<f32>(00.0, 0.0, 30.0), vec2<f32>(5.0 + sin(GLOBALS.time * 0.1) * 6.0, 10.0)) * 30.0;

    // Axes

    let sd_axes = MAX_POSITIVE_F32; // sd_scene_axes(q);

    var b = 0.0;
    var a = sd_scene_column(p + vec3<f32>(b), vec3<f32>(0.0));

    b += 1.0;
    a = min(a, sd_scene_column(p + vec3<f32>(b), vec3<f32>(0.0)));
        b += 1.0;
    a = min(a, sd_scene_column(p + vec3<f32>(b), vec3<f32>(0.0)));
        b += 1.0;
    a = min(a, sd_scene_column(p + vec3<f32>(b), vec3<f32>(0.0)));
        b += 1.0;
    a = min(a, sd_scene_column(p + vec3<f32>(b), vec3<f32>(0.0)));
        b += 1.0;
    a = min(a, sd_scene_column(p + vec3<f32>(b), vec3<f32>(0.0)));
        b += 1.0;
    a = min(a, sd_scene_column(p + vec3<f32>(b), vec3<f32>(0.0)));
        b += 1.0;
    a = min(a, sd_scene_column(p + vec3<f32>(b), vec3<f32>(0.0)));
        b += 1.0;
    a = min(a, sd_scene_column(p + vec3<f32>(b), vec3<f32>(0.0)));
        b += 1.0;
    a = min(a, sd_scene_column(p + vec3<f32>(b), vec3<f32>(0.0)));
        b += 1.0;
    a = min(a, sd_scene_column(p + vec3<f32>(b), vec3<f32>(0.0)));
        b += 1.0;
    a = min(a, sd_scene_column(p + vec3<f32>(b), vec3<f32>(0.0)));
        b += 1.0;
    a = min(a, sd_scene_column(p + vec3<f32>(b), vec3<f32>(0.0)));
        b += 1.0;
    a = min(a, sd_scene_column(p + vec3<f32>(b), vec3<f32>(0.0)));
        b += 1.0;
    a = min(a, sd_scene_column(p + vec3<f32>(b), vec3<f32>(0.0)));
        b += 1.0;
    a = min(a, sd_scene_column(p + vec3<f32>(b), vec3<f32>(0.0)));
        b += 1.0;
    a = min(a, sd_scene_column(p + vec3<f32>(b), vec3<f32>(0.0)));
        b += 1.0;
    a = min(a, sd_scene_column(p + vec3<f32>(b), vec3<f32>(0.0)));
        b += 1.0;
    a = min(a, sd_scene_column(p + vec3<f32>(b), vec3<f32>(0.0)));
            b += 1.0;
    a = min(a, sd_scene_column(p + vec3<f32>(b), vec3<f32>(0.0)));
        b += 1.0;
    a = min(a, sd_scene_column(p + vec3<f32>(b), vec3<f32>(0.0)));
        b += 1.0;
    a = min(a, sd_scene_column(p + vec3<f32>(b), vec3<f32>(0.0)));
        b += 1.0;
    a = min(a, sd_scene_column(p + vec3<f32>(b), vec3<f32>(0.0)));
        b += 1.0;
    a = min(a, sd_scene_column(p + vec3<f32>(b), vec3<f32>(0.0)));
        b += 1.0;
    a = min(a, sd_scene_column(p + vec3<f32>(b), vec3<f32>(0.0)));
        b += 1.0;

    // Plane

    let sd_plane = MAX_POSITIVE_F32;

    let sd_runtime_scene = sd_runtime_scene(q);

    let qt = abs(q - vec3<f32>(4.0, 5.0, 4.0)) - vec3<f32>(1.0, 1.0, 1.0);
    let rounded_box = length(max(qt, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - 0.5;

    return SdSceneData(min(min(min(a, sd_axes), rounded_box), min(sd_runtime_scene, sd_plane)), 0.0);

    // return SdSceneData(min(
    //     sdPostSmoothUnion(
    //         sd_tetrahedron,
    //         sdPostSmoothUnion(
    //             sdPostSmoothUnion(
    //                 sd_plane,
    //                 sd_large_columns,
    //                 60.0,
    //             ),
    //             sd_columns,
    //             5.0,
    //         ),
    //         20.0,
    //     ),
    //     sd_axes,
    // ), sd_tetrahedron_data.y);
}

fn sd_scene_material(p: vec3<f32>, base_color: vec3<f32>) -> PhongReflectionMaterial {
    if (sd_scene_axes(p) < 0.001 && length(p) > 0.5) {
        var color: vec3<f32>;
        if (abs(p.x) > 0.5) {
            color = vec3<f32>(0.5 + f32(p.x > 0.0) * 1.0, 0.2, 0.2);
        } else if (abs(p.y) > 0.5) {
            color = vec3<f32>(0.2, 0.5 + f32(p.y > 0.0) * 1.0, 0.2);
        } else {
            color = vec3<f32>(0.2, 0.2, 0.5 + f32(p.z > 0.0) * 1.0);
        }

        return PhongReflectionMaterial(color, 0.0, 0.0, 0.9, 1.0);
    }

    return PhongReflectionMaterial(base_color, 0.3, 0.7, 0.05, 30.0);
}

const NORMAL_EPSILON = 0.001;

fn sd_scene_normal(p: vec3<f32>) -> vec3<f32> {
    return normalize(vec3(
        sd_scene(vec3<f32>(p.x + NORMAL_EPSILON, p.y, p.z)).sd - sd_scene(vec3<f32>(p.x - NORMAL_EPSILON, p.y, p.z)).sd,
        sd_scene(vec3<f32>(p.x, p.y + NORMAL_EPSILON, p.z)).sd - sd_scene(vec3<f32>(p.x, p.y - NORMAL_EPSILON, p.z)).sd,
        sd_scene(vec3<f32>(p.x, p.y, p.z  + NORMAL_EPSILON)).sd - sd_scene(vec3<f32>(p.x, p.y, p.z - NORMAL_EPSILON)).sd
    ));
}
