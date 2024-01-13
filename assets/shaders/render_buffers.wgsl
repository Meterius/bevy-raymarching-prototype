#import "shaders/compiled/data.generated.wgsl"::{RenderGlobals, RenderCamera, RenderScene, RenderSDTransform, RenderSDReference, RenderSDSphere, RenderSDBox, RenderSDUnion}

@group(0) @binding(0) var TEXTURE: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(1) var<uniform> GLOBALS: RenderGlobals;
@group(0) @binding(3) var<uniform> SCENE: RenderScene;
@group(0) @binding(2) var<uniform> CAMERA: RenderCamera;

@group(1) @binding(0) var<storage, read> SD_ROOTS: array<RenderSDReference, 20>;
@group(1) @binding(1) var<storage, read> SD_SPHERES: array<RenderSDSphere, 20>;
@group(1) @binding(2) var<storage, read> SD_BOXES: array<RenderSDBox, 20>;
@group(1) @binding(3) var<storage, read> SD_TRANSFORMS: array<RenderSDTransform, 20>;
@group(1) @binding(4) var<storage, read> SD_UNIONS: array<RenderSDUnion, 20>;
