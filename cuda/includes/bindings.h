#pragma once

#define CONE_MARCH_LEVELS 4
// #define DISABLE_CONE_MARCH

enum RayMarchHitOutcome {
    Collision, StepLimit, DepthLimit
};

struct ConeMarchTextureValue {
    float depth;
    int steps;
    enum RayMarchHitOutcome outcome;
};

struct ConeMarchTexture {
    struct ConeMarchTextureValue* texture;
    unsigned int size[2];
};

struct ConeMarchTextures {
    struct ConeMarchTexture textures[CONE_MARCH_LEVELS];
};

struct Texture {
    unsigned int* texture;
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

struct SdSphere {
    float translation[3];
    float rotation[3];
    float scale[3];
};