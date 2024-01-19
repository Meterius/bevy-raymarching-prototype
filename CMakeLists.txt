cmake_minimum_required(VERSION 3.16)
project(bevy_raymarching_prototype CUDA)

set(CMAKE_CUDA_STANDARD 14)

include_directories(cuda/includes)
include_directories(cuda/kernels)

add_executable(
    bevy_raymarching_prototype
    cuda/includes/signed_distance.cu
    cuda/includes/rendering.cu
    cuda/includes/ray_marching.cu
    cuda/modules/main.cu
        cuda/includes/types.cu
)

set_target_properties(bevy_raymarching_prototype PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(bevy_raymarching_prototype PROPERTIES CUDA_ARCHITECTURES OFF)
