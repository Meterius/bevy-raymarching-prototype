use cudarc::driver::DeviceRepr;

pub mod cuda;

unsafe impl DeviceRepr for cuda::GlobalsBuffer {}
unsafe impl DeviceRepr for cuda::CameraBuffer {}
unsafe impl DeviceRepr for cuda::DepthTexture {}
unsafe impl DeviceRepr for cuda::DepthTextureEntry {}
