use cudarc::driver::DeviceRepr;

pub mod cuda;

unsafe impl DeviceRepr for cuda::GlobalsBuffer {}
unsafe impl DeviceRepr for cuda::CameraBuffer {}
unsafe impl DeviceRepr for cuda::ConeMarchTexture {}
unsafe impl DeviceRepr for cuda::ConeMarchTextures {}

unsafe impl DeviceRepr for cuda::SdComposition {}

unsafe impl DeviceRepr for cuda::SdRuntimeScene {}
unsafe impl DeviceRepr for cuda::Texture {}

unsafe impl DeviceRepr for cuda::MeshVertex {}

unsafe impl DeviceRepr for cuda::MeshTriangle {}

unsafe impl DeviceRepr for cuda::Mesh {}

unsafe impl DeviceRepr for cuda::MeshBuffer {}
unsafe impl DeviceRepr for cuda::ConeMarchTextureValue {}
unsafe impl DeviceRepr for cuda::RenderDataTexture {}
unsafe impl DeviceRepr for cuda::RenderDataTextureValue {}
