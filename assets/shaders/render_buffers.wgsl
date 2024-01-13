#import "shaders/compiled/data.generated.wgsl"::{RenderGlobals, RenderCamera, RenderScene, RenderSDObject, RenderSDSphere, RenderSDBox, RenderSDUnion}

@group(0) @binding(0) var TEXTURE: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(1) var<uniform> GLOBALS: RenderGlobals;
@group(0) @binding(3) var<uniform> SCENE: RenderScene;
@group(0) @binding(2) var<uniform> CAMERA: RenderCamera;

@group(1) @binding(0) var<storage, read> SD_OBJECTS: array<RenderSDObject>;
@group(1) @binding(1) var<storage, read> SD_SPHERES: array<RenderSDSphere>;
@group(1) @binding(2) var<storage, read> SD_BOXES: array<RenderSDBox>;
@group(1) @binding(3) var<storage, read> SD_UNIONS: array<RenderSDUnion>;
