#pragma once

#include <stdbool.h>

#define CUDA_DEBUG false

#define BLOCK_SIZE 128

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
