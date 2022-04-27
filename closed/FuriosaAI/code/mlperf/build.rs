extern crate cbindgen;
extern crate cpp_build;

use std::env;
use std::path::Path;

fn main() {
    cbindgen::Builder::new()
        .with_crate(env::var("CARGO_MANIFEST_DIR").unwrap())
        .with_config(cbindgen::Config {
            include_guard: Some("bindings_h".to_string()),
            language: cbindgen::Language::Cxx,
            ..cbindgen::Config::default()
        })
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(Path::new(&env::var("OUT_DIR").unwrap()).join("bindings.h"));

    cpp_build::Config::new()
        .flag("-march=native")
        .flag("-std=c++17")
        .flag("-fopenmp")
        .include(env::var("OUT_DIR").unwrap())
        .opt_level(3)
        .build("src/lib.rs");
}
