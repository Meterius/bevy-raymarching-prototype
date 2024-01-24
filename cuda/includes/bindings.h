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

// par0 stores inverted euler rotation
enum SdPrimitiveVariant {
    None,
    Empty,
    Sphere,
    Cube,
    Mandelbulb
};

enum SdCompositionVariant {
    Union,
    Difference,
    Intersect,
    Mirror // par0 stores direction
};

struct SdComposition {
    float bound_max[3];
    float bound_min[3];

    float par0[4];
    float par1[3];
    float par2[3];

    unsigned int child;
    unsigned int parent: 24;
    enum SdCompositionVariant variant: 4;
    enum SdPrimitiveVariant primitive_variant: 4;
};

struct SdRuntimeSceneLighting {
    int sun_light_count;
    struct SunLight sun_lights[MAX_SUN_LIGHT_COUNT];
};

struct SdRuntimeSceneGeometry {
    struct SdComposition *compositions;
};

struct SdRuntimeScene {
    struct SdRuntimeSceneGeometry geometry;
    struct SdRuntimeSceneLighting lighting;
};
