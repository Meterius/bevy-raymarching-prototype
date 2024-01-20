#pragma once

#define CONE_MARCH_LEVELS 4
#define MAX_SUN_LIGHT_COUNT 10
#define MAX_POINT_LIGHT_COUNT 10
#define MAX_SPHERE_COUNT 256
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

struct SdSphere {
    float translation[3];
    float radius;
};

struct SdRuntimeSceneLighting {
    int sun_light_count;
    struct SunLight sun_lights[MAX_SUN_LIGHT_COUNT];
};

struct SdRuntimeScene {
    int sphere_count;
    struct SdSphere spheres[MAX_SPHERE_COUNT];

    struct SdRuntimeSceneLighting lighting;
};
