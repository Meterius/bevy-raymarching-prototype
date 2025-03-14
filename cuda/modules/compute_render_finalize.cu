#include "./common.cu"

extern "C" __global__ void compute_render_finalize(
    const Texture render_texture,
    const RenderDataTexture render_data_texture,
    const GlobalsBuffer globals
) {
    u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    uvec2 texture_coord = render_texture_coord({ render_data_texture.size[0], render_data_texture.size[1] });
    u32 texture_index = id;

    RenderDataTextureValue texture_value = render_data_texture.texture[texture_index];

    vec3 color;
    if (texture_value.outcome == Collision) {
        color = from_array(texture_value.color) * texture_value.light + vec3(texture_value.depth * 0.00001f);
    } else if (texture_value.outcome == DepthLimit) {
        color = vec3(0.0f, 1.0f, 0.0f);
    } else {
        color = vec3(0.0f, 0.0f, 1.0f);
    }

    color = hdr_map_aces_tone(max(color, 0.0f));

    unsigned int rgba = ((unsigned int) (255.0f * color.x) & 0xff) |
                        (((unsigned int) (255.0f * color.y) & 0xff) << 8) |
                        (((unsigned int) (255.0f * color.z) & 0xff) << 16) |
                        ((unsigned int) 255 << 24);

    render_texture.texture[index_2d(texture_coord, render_texture)] = rgba;
}
