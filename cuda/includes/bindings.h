#pragma once

#include <stdbool.h>

#define CUDA_DEBUG false

#define CONE_MARCH_LEVELS 2
#define MAX_SUN_LIGHT_COUNT 1
#define MAX_POINT_LIGHT_COUNT 1
#define MAX_COMPOSITION_NODE_COUNT 4096
#define BLOCK_SIZE 128
// #define DISABLE_CONE_MARCH

enum RayMarchHitOutcome {
    Collision, StepLimit, DepthLimit
};

struct RenderDataTextureValue {
    float depth;
    float steps;
    enum RayMarchHitOutcome outcome;
    float color[3];
    float light;
};

struct RenderDataTexture {
    struct RenderDataTextureValue *texture;
    unsigned int size[2];
};

struct ConeMarchTextureValue {
    float depth;
    float steps;
    enum RayMarchHitOutcome outcome;
    int padding;
};

struct ConeMarchTexture {
    struct ConeMarchTextureValue *texture;
    unsigned int size[2];
};

struct ConeMarchTextures {
    struct ConeMarchTexture textures[CONE_MARCH_LEVELS];
};

struct Texture {
    unsigned int *texture;
    unsigned int size[2];
};

struct GlobalsBuffer {
    unsigned long long tick;
    float time;
    bool use_step_glow_on_background;
    bool use_step_glow_on_foreground;
    unsigned int render_texture_size[2];
    float render_screen_size[2];
};

struct CameraBuffer {
    float position[3];
    float forward[3];
    float up[3];
    float right[3];
    float fov;
};

// CPU

struct PointLight {
    float position[3];
    float intensity;
};

struct SunLight {
    float direction[3];
};

struct MeshVertex {
    float pos[3];
    char padding[4];
};

struct MeshTriangle {
    unsigned int vertex_ids[3];
    char padding[4];
};

struct Mesh {
    unsigned int triangle_start_id;
    unsigned int triangle_end_id;
};

struct MeshBuffer {
    struct MeshVertex *vertices;
    struct MeshTriangle *triangles;
    struct Mesh *meshes;
};

enum SdPrimitiveVariant {
    None,
    Empty,
    Sphere,
    Cube,
    Mandelbulb,
    Triangle
};

enum SdCompositionVariant {
    Union,
    Difference,
    Intersect,
    Mirror
};

struct SdComposition {
    float bound_max[3];
    float bound_min[3];

    unsigned int child: 24;
    unsigned int second_child_offset: 8;
    unsigned int parent: 24;
    enum SdCompositionVariant variant: 4;
    enum SdPrimitiveVariant primitive_variant: 4;
};

struct SdCompositionMirrorAppendix {
    float translation[3];
    float direction[3];
    char padding[8];
};

struct SdCompositionPrimitiveAppendix {
    float scale[3];
    float rotation[4];
    unsigned int mesh_id; // only valid if primitive is mesh
};

struct SdCompositionPrimitiveTriangleAppendix {
    float v1[3];
    float v2[3];
    // one of the vertices aligns in at least elements with a corner of the bounding box,
    // it can be reconstructed using a bit-vector that encodes the corner,
    // the two bits indicate whether min or max plane is hit
    unsigned int bb_v0;
    float v0_z;
};

struct SdRuntimeSceneLighting {
    int sun_light_count;
    struct SunLight sun_lights[MAX_SUN_LIGHT_COUNT];
};

struct SdRuntimeSceneGeometry {
    struct SdComposition *compositions;
    struct MeshBuffer meshes;
};

struct SdRuntimeScene {
    struct SdRuntimeSceneGeometry geometry;
    struct SdRuntimeSceneLighting lighting;
};
