cmake_minimum_required(VERSION 3.16)
project(bevy_raymarching_prototype CUDA)

set(CMAKE_CUDA_STANDARD 14)

include_directories(cuda/includes)
include_directories(cuda/kernels)

add_executable(bevy_raymarching_prototype cuda/kernels/renderer.cu cuda/kernels/signed_distance.cu cuda/kernels/ray_marching.cu
        cuda/includes/utils.h)

set_target_properties(bevy_raymarching_prototype PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(bevy_raymarching_prototype PROPERTIES CUDA_ARCHITECTURES OFF)
