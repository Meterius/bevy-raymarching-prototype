use proc_macro2::TokenStream;
use quote::quote;
use std::fs::File;
use std::path::PathBuf;
use syn::{parse_file, Attribute, Item, Meta};

// Utility

fn add_derive_extensions_to_structs(
    code: &str,
    structs_to_modify: &[&str],
    extensions: &[&str],
) -> String {
    let ast = parse_file(code).expect("Failed to parse code");

    let modified_ast = ast
        .items
        .into_iter()
        .map(|item| {
            if let Item::Struct(struct_item) = &item {
                if structs_to_modify.contains(&&*struct_item.ident.to_string()) {
                    let mut new_struct = struct_item.clone();
                    extend_derive_with(&mut new_struct.attrs, extensions);
                    return Item::Struct(new_struct);
                }
            }
            item
        })
        .collect::<Vec<_>>();

    let tokens = quote! {
        #(#modified_ast)*
    };

    tokens.to_string()
}

fn extend_derive_with(attrs: &mut Vec<Attribute>, extensions: &[&str]) {
    for attr in attrs.iter_mut() {
        match &mut attr.meta {
            Meta::List(ref mut meta_list) => {
                if meta_list.path.is_ident("derive") {
                    for ext in extensions.iter() {
                        meta_list
                            .tokens
                            .extend(TokenStream::from_str(format!(", {ext}").as_str()));
                    }
                    break;
                }
            }
            _ => {}
        };
    }
}

// CUDA

extern crate cc;

use std::process::Command;
use std::str::FromStr;

#[allow(unused_macros)]
macro_rules! warn {
    ($($tokens: tt)*) => {
        println!("cargo:warning={}", format!($($tokens)*))
    }
}

fn compile_cuda() {
    // Tell cargo to invalidate the built crate whenever fils of interest changes.
    println!("cargo:rerun-if-changed={}", "cuda");

    // build the cuda modules
    let cuda_src = PathBuf::from("cuda/modules/main.cu");
    let ptx_file = "assets/cuda/compiled/main.ptx";

    let cubin_file = "assets/cuda/compiled/main.cubin";
    let reg_limited_cubin_file = "assets/cuda/compiled/main_regl.cubin";

    let mut path = std::env::var("PATH").unwrap();
    path.push_str(";C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.38.33130\\bin\\Hostx64\\x64;");

    let make_nvcc = |out_file: &str| {
        let file = File::create(format!("logs/nvcc_{}.txt", out_file.replace("/", "_"))).unwrap();

        let mut nvcc_cmd = Command::new("nvcc");

        nvcc_cmd
            .env("PATH", path.as_str())
            .arg("-o")
            .arg(out_file)
            .arg(&cuda_src)
            .stderr(file.try_clone().unwrap())
            .stdout(file.try_clone().unwrap());

        if cfg!(debug_assertions) {
            nvcc_cmd.arg("-lineinfo");
            // nvcc_cmd.arg("-G");
            nvcc_cmd.arg("--use_fast_math");
            nvcc_cmd.arg("-Xptxas=\"-v\"");
            nvcc_cmd.arg("-Xptxas=\"-o=3\"");
        } else {
            nvcc_cmd.arg("--use_fast_math");
            nvcc_cmd.arg("-Xptxas=\"-o=3\"");
        }

        println!("{nvcc_cmd:?}");

        nvcc_cmd
    };

    let nvcc_status = make_nvcc(ptx_file)
        .arg(format!("-arch={}", "compute_86"))
        .arg(format!("-code={}", "sm_86"))
        .arg("-ptx")
        .status()
        .unwrap();
    assert!(
        nvcc_status.success(),
        "Failed to compile CUDA source to PTX."
    );

    for reg_compressed in vec![false, true].into_iter() {
        let mut nvcc_status = make_nvcc(if reg_compressed {
            reg_limited_cubin_file
        } else {
            cubin_file
        });

        nvcc_status.arg(format!("-arch={}", "native")).arg("-cubin");

        if reg_compressed {
            nvcc_status.arg("--maxrregcount=56");
        }

        let nvcc_status = nvcc_status.status().unwrap();

        assert!(
            nvcc_status.success(),
            "Failed to compile CUDA source to CUBIN."
        );
    }

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

    // Regex to find raw pointers to float and replace them with CudaSlice<f32>
    // You can copy this regex to add/modify other types of pointers, for example "*mut i32"
    let mut modified_bindings = String::from("#![allow(warnings)]\n\n");
    modified_bindings.push_str(
        add_derive_extensions_to_structs(
            bindings.to_string().as_str(),
            &["SdSpherePrimitive", "SdCubePrimitive", "SdComposition"],
            &["Default"],
        )
        .as_str(),
    );

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    std::fs::write("src/bindings/cuda.rs", modified_bindings.as_bytes())
        .expect("Failed to write bindings");
}

// Build Script

fn main() {
    compile_cuda();
}
