use regex::Regex;
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;

// WGSL Preprocessor

#[allow(unused_macros)]
macro_rules! warn {
    ($($tokens: tt)*) => {
        println!("cargo:warning={}", format!($($tokens)*))
    }
}

fn ensure_dir_exists(dir: &Path) -> std::io::Result<()> {
    if !dir.exists() {
        fs::create_dir_all(dir)?;
    }
    Ok(())
}

fn write_to_file(file_path: &Path, content: &str) -> std::io::Result<()> {
    if let Some(dir) = file_path.parent() {
        ensure_dir_exists(dir)?;
    }

    let mut file = File::create(file_path)?;
    file.write_all(content.as_bytes())?;
    Ok(())
}

fn insert_subdirectory(original_path: &Path, sub_dir: &str) -> PathBuf {
    let mut new_path = PathBuf::new();

    if let Some(parent) = original_path.parent() {
        new_path.push(parent);
    }

    new_path.push(sub_dir);

    if let Some(filename) = original_path.file_name() {
        new_path.push(filename);
    }

    new_path
}

fn visit_dirs(dir: &Path, cb: &mut dyn FnMut(&Path, &str)) {
    if dir.is_dir() {
        match fs::read_dir(dir) {
            Ok(read_dir) => {
                for entry in read_dir.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        visit_dirs(&path, cb);
                    } else {
                        match fs::File::open(&path) {
                            Ok(mut file) => {
                                let mut contents = String::new();
                                match file.read_to_string(&mut contents) {
                                    Ok(_) => cb(&path, contents.as_str()),
                                    Err(e) => println!("Error reading file: {}", e),
                                }
                            }
                            Err(e) => println!("Error opening file: {}", e),
                        }
                    }
                }
            }
            Err(e) => println!("Error reading directory: {}", e),
        }
    }
}

#[derive(Debug)]
struct MacroFunc {
    name: String,
    args: Vec<String>,
}

impl FromStr for MacroFunc {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let re = Regex::new(r"^#macro_func (\w+)\((.*)\)\s+$").map_err(|_| "Invalid regex")?;
        let caps = re.captures(s).ok_or("Invalid macro function format")?;

        let name = caps
            .get(1)
            .ok_or("Macro function name not found")?
            .as_str()
            .to_string();
        let args_str = caps
            .get(2)
            .ok_or("Macro function arguments not found")?
            .as_str();
        let args = args_str
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();

        Ok(MacroFunc { name, args })
    }
}

#[derive(Debug)]
struct MacroFuncCall {
    name: String,
    args: Vec<String>,
}

impl FromStr for MacroFuncCall {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let re =
            Regex::new(r"^#macro_func_call(\w+)\(([\w\s,]*)\)\s+$").map_err(|_| "Invalid regex")?;
        let caps = re.captures(s).ok_or("Invalid macro function format")?;

        let name = caps
            .get(1)
            .ok_or("Macro function name not found")?
            .as_str()
            .to_string();
        let args_str = caps
            .get(2)
            .ok_or("Macro function arguments not found")?
            .as_str();
        let args = args_str
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();

        Ok(MacroFuncCall { name, args })
    }
}

fn compile_shaders() {
    let root = Path::new("assets/shaders");

    let mut macro_functions = HashMap::<String, (MacroFunc, String)>::new();

    let is_valid_file = |path: &Path| {
        path.extension() == Some(OsStr::new("wgsl"))
            && path.iter().all(|component| component != "compiled")
    };

    // collect macro functions
    visit_dirs(root, &mut |_path: &Path, contents: &str| {
        let mut index = 0;

        while let Some(start_offset) = contents[index..].find("#macro_func") {
            let macro_start_index = index + start_offset;
            let end_offset = contents[macro_start_index..]
                .find("#macro_func_end")
                .expect("macro_func definition not closed");
            let macro_end_index = macro_start_index + end_offset;

            let def_end_offset = contents[macro_start_index..].find("\n").unwrap();
            let def_end_index = macro_start_index + def_end_offset;

            let macro_func =
                MacroFunc::from_str(&contents[macro_start_index..def_end_index]).unwrap();

            macro_functions.insert(
                macro_func.name.clone(),
                (
                    macro_func,
                    String::from(&contents[def_end_index + 1..macro_end_index]),
                ),
            );

            index = macro_end_index;
        }
    });

    // replace macro functions
    visit_dirs(root, &mut |path: &Path, contents: &str| {
        if !is_valid_file(path) {
            return;
        }

        let mut index = 0;

        let mut result = String::from(contents);

        while let Some(start_offset) = contents[index..].find("#macro_func_call") {
            let start_index = index + start_offset;
            let call_end_offset = contents[start_index..].find("\n").unwrap();
            let call_end_index = start_index + call_end_offset;

            let macro_func_call =
                MacroFuncCall::from_str(&contents[start_index..call_end_index]).unwrap();

            let (macro_func, body) = macro_functions
                .get(macro_func_call.name.as_str())
                .expect("Unknown macro func");

            let mut content = String::from(body);

            for i in 0..macro_func.args.len() {
                content = content.replace(
                    macro_func.args[i].as_str(),
                    macro_func_call.args[i].as_str(),
                );
            }

            result.replace_range(start_index..=call_end_index, content.as_str());

            index = call_end_index + 1;
        }

        write_to_file(
            insert_subdirectory(path, "compiled").as_path(),
            result.as_str(),
        )
        .unwrap();
    });
}

// Wgsl Struct Generator

fn is_shader_struct(item: &syn::ItemStruct) -> bool {
    item.attrs.iter().any(|attr| {
        match &attr.meta {
            syn::Meta::List(value) => {
                if let Some(ident) = value.path.get_ident() {
                    if ident != &syn::Ident::new("derive", proc_macro2::Span::call_site()) {
                        return false;
                    }

                    return value.tokens.clone().into_iter().any(
                        |tree| match tree {
                            proc_macro2::TokenTree::Ident(ident) => ident == syn::Ident::new("ShaderType", proc_macro2::Span::call_site()),
                            _ => false
                        }
                    );
                } else {
                    return false;
                }
            },
            _ => { return false; },
        };
    })
}

use quote::ToTokens;

fn compile_shader_structs() {
    let ast = syn::parse_file(include_str!("src/renderer/types.rs")).unwrap();

    let mut result = String::new();

    let mut type_translation = std::collections::HashMap::<String, String>::new();
    type_translation.insert(String::from("f32"), String::from("f32"));
    type_translation.insert(String::from("Vec2"), String::from("vec2<f32>"));
    type_translation.insert(String::from("Vec3"), String::from("vec3<f32>"));

    ast.items.iter().for_each(|item| {
        match item {
            syn::Item::Struct(def) => {
                if !is_shader_struct(def) { return; }

                result.push_str(format!("struct {} {{\n", def.ident).as_str());

                def.fields.iter().for_each(|field| {
                    let type_name = format!("{}", field.ty.to_token_stream());

                    let translated_type_name = {
                        let type_name = if type_name.starts_with("Vec < ") && type_name.ends_with(" >") {
                            format!("array<{}>", &type_name.as_str()[6..type_name.len() - 2])
                        } else { type_name };
                        
                        type_translation
                            .get(type_name.as_str())
                            .unwrap_or(&type_name)
                            .clone()
                    };

                    result.push_str(format!("\t{}: {},\n", field.ident.as_ref().unwrap(), translated_type_name).as_str());
                });

                result.push_str("}\n\n");
            },
            _ => {},
        };
    });

    let mut file = File::create(Path::new("assets/shaders/data.generated.wgsl")).unwrap();
    file.write_all(result.as_bytes()).unwrap();
}

// Build Script

fn main() {
    compile_shader_structs();
    compile_shaders();
}
