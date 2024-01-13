#import "shaders/compiled/data.generated.wgsl"::{RenderGlobals, RenderCamera, RenderScene}

@group(0) @binding(0) var TEXTURE: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(1) var<uniform> GLOBALS: RenderGlobals;
@group(0) @binding(2) var<uniform> CAMERA: RenderCamera;
@group(0) @binding(3) var<uniform> SCENE: RenderScene;