use cudarc::driver::{
    result, sys, CudaDevice, CudaFunction, CudaStream, DeviceRepr, LaunchAsync, LaunchConfig,
};

use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum CustomCudaFunctionInner {
    CudaFunction(CudaFunction),
    SysCudaFunction(sys::CUfunction),
}

#[derive(Debug, Clone)]
pub struct CustomCudaFunction {
    cu_function: CustomCudaFunctionInner,
    device: Arc<CudaDevice>,
}

impl CustomCudaFunction {
    pub fn from_safe(cu_function: CudaFunction, device: Arc<CudaDevice>) -> Self {
        Self {
            cu_function: CustomCudaFunctionInner::CudaFunction(cu_function),
            device,
        }
    }

    pub fn from_sys(cu_function: sys::CUfunction, device: Arc<CudaDevice>) -> Self {
        Self {
            cu_function: CustomCudaFunctionInner::SysCudaFunction(cu_function),
            device,
        }
    }
}

macro_rules! impl_launch {
    ([$($Vars:tt),*], [$($Idx:tt),*]) => {
        unsafe impl<$($Vars: DeviceRepr),*> LaunchAsync<($($Vars, )*)> for CustomCudaFunction {
            #[inline(always)]
            unsafe fn launch(
                self,
                _cfg: LaunchConfig,
                _args: ($($Vars, )*)
            ) -> Result<(), result::DriverError> {
                unimplemented!();
            }

            #[inline(always)]
            unsafe fn launch_on_stream(
                self,
                stream: &CudaStream,
                cfg: LaunchConfig,
                args: ($($Vars, )*)
            ) -> Result<(), result::DriverError> {
                match self.cu_function {
                    CustomCudaFunctionInner::SysCudaFunction(cu_function) => {
                        let params = &mut [$(args.$Idx.as_kernel_param(), )*];
                        self.device.bind_to_thread()?;
                        result::launch_kernel(
                            cu_function,
                            cfg.grid_dim,
                            cfg.block_dim,
                            cfg.shared_mem_bytes,
                            stream.stream,
                            params,
                        )
                    }
                    CustomCudaFunctionInner::CudaFunction(function) => {
                        function.clone().launch_on_stream(stream, cfg, args)
                    }
                }
            }
        }
    };
}

impl_launch!([A], [0]);
impl_launch!([A, B], [0, 1]);
impl_launch!([A, B, C], [0, 1, 2]);
impl_launch!([A, B, C, D], [0, 1, 2, 3]);
impl_launch!([A, B, C, D, E], [0, 1, 2, 3, 4]);
impl_launch!([A, B, C, D, E, F], [0, 1, 2, 3, 4, 5]);
impl_launch!([A, B, C, D, E, F, G], [0, 1, 2, 3, 4, 5, 6]);
impl_launch!([A, B, C, D, E, F, G, H], [0, 1, 2, 3, 4, 5, 6, 7]);
impl_launch!([A, B, C, D, E, F, G, H, I], [0, 1, 2, 3, 4, 5, 6, 7, 8]);
impl_launch!(
    [A, B, C, D, E, F, G, H, I, J],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
);
impl_launch!(
    [A, B, C, D, E, F, G, H, I, J, K],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
);
impl_launch!(
    [A, B, C, D, E, F, G, H, I, J, K, L],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
);

unsafe impl Send for CustomCudaFunction {}

unsafe impl Sync for CustomCudaFunction {}
