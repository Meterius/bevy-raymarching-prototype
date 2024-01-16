use regex::Regex;
use std::ops::Deref;
use std::path::PathBuf;

// CUDA

extern crate cc;

use std::process::Command;

fn compile_cuda() {
    // Tell cargo to invalidate the built crate whenever fils of interest changes.
    println!("cargo:rerun-if-changed={}", "cuda");

    // Specify the desired architecture version.
    let arch = "compute_86"; // For example, using SM 8.6 (Ampere architecture).
    let code = "sm_86"; // For the same SM 8.6 (Ampere architecture).

    // build the cuda kernels
    let cuda_src = PathBuf::from("cuda/kernels/renderer.cu");
    let ptx_file = "assets/cuda/compiled/renderer.ptx";

    let mut path = std::env::var("PATH").unwrap();
    path.push_str(";C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.38.33130\\bin\\Hostx64\\x64;");

    let nvcc_status = Command::new("nvcc")
        .env("PATH", path)
        .arg("-ptx")
        .arg("-o")
        .arg(&ptx_file)
        .arg(&cuda_src)
        .arg(format!("-arch={}", arch))
        .arg(format!("-code={}", code))
        //.arg("-G")
        //.arg("-lineinfo")
        .arg("-Xptxas")
        .arg("-O3")
        .arg("--use_fast_math")
        .status()
        .unwrap();

    assert!(
        nvcc_status.success(),
        "Failed to compile CUDA source to PTX."
    );

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("cuda/includes/bindings.h")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // we use "no_copy" and "no_debug" here because we don't know if we can safely generate them for our structs in C code (they may contain raw pointers)
        .no_copy("*")
        .no_debug("*")
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // we need to make modifications to the generated code
    let generated_bindings = bindings.to_string();

    // Regex to find raw pointers to float and replace them with CudaSlice<f32>
    // You can copy this regex to add/modify other types of pointers, for example "*mut i32"
    let modified_bindings = generated_bindings;
    let modified_bindings =
        String::from("use cudarc::driver::CudaSlice;") + modified_bindings.deref();

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    std::fs::write("src/bindings/cuda.rs", modified_bindings.as_bytes())
        .expect("Failed to write bindings");
}

// Build Script

fn main() {
    compile_cuda();
}
