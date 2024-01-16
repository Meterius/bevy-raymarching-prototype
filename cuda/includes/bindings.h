#pragma once

struct DepthTextureEntry {
    float depth;
    int steps;
};

struct DepthTexture {
    struct DepthTextureEntry* texture;
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