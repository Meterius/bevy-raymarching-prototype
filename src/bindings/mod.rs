use cudarc::driver::DeviceRepr;

pub mod cuda;

unsafe impl DeviceRepr for cuda::GlobalsBuffer {}
unsafe impl DeviceRepr for cuda::CameraBuffer {}
unsafe impl DeviceRepr for cuda::Texture {}
unsafe impl DeviceRepr for cuda::RenderDataTexture {}
unsafe impl DeviceRepr for cuda::RenderDataTextureValue {}
