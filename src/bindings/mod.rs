use cudarc::driver::DeviceRepr;

pub mod cuda;

unsafe impl DeviceRepr for cuda::GlobalsBuffer {}
unsafe impl DeviceRepr for cuda::CameraBuffer {}
unsafe impl DeviceRepr for cuda::ConeMarchTexture {}
unsafe impl DeviceRepr for cuda::ConeMarchTextures {}

unsafe impl DeviceRepr for cuda::SdComposition {}

unsafe impl DeviceRepr for cuda::SdSpherePrimitive {}

unsafe impl DeviceRepr for cuda::SdCubePrimitive {}
unsafe impl DeviceRepr for cuda::SdRuntimeScene {}
unsafe impl DeviceRepr for cuda::Texture {}
unsafe impl DeviceRepr for cuda::ConeMarchTextureValue {}
unsafe impl DeviceRepr for cuda::RenderDataTexture {}
unsafe impl DeviceRepr for cuda::RenderDataTextureValue {}

impl Default for cuda::SdSpherePrimitive {
    fn default() -> Self {
        Self {
            translation: [0.0; 3],
            scale: [0.0; 3],
        }
    }
}

impl Default for cuda::SdCubePrimitive {
    fn default() -> Self {
        Self {
            translation: [0.0; 3],
            scale: [0.0; 3],
        }
    }
}

impl Default for cuda::SdComposition {
    fn default() -> Self {
        Self {
            primitive: 0,
            parent: 0,
            primitive_variant: 0,
            child_leftmost: 0,
            child_rightmost: 0,
            variant: 0,
        }
    }
}
