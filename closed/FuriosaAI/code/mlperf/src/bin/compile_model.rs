use std::env;
use std::path::PathBuf;

use structopt::StructOpt;

use mlperf::utils::{compile, compile_with_config_from_device};

#[derive(StructOpt)]
struct CompilerOptions {
    /// Sets the model file path
    #[structopt(long = "model", parse(from_os_str))]
    model_path: PathBuf,

    /// Sets the batch size [default: 1]
    #[structopt(long = "batch-size")]
    batch_size: Option<usize>,

    /// Sets `remove_unlower` compiler config [default: false]
    #[structopt(long = "remove-unlower")]
    remove_unlower: bool,

    /// Do not save the result to cache [default: false]
    #[structopt(long = "no-save")]
    no_save: bool,

    /// Compile with config from the device.
    /// To use this, set both `NPU_DEVNAME` and `NPU_GLOBAL_CONFIG_PATH` [default: false]
    #[structopt(long = "dev-conf")]
    conf_from_dev: bool,
}

fn main() {
    let matches = CompilerOptions::clap().get_matches();
    let options = CompilerOptions::from_clap(&matches);

    if options.no_save {
        env::set_var("MLCOMMONS_USE_CACHED_GRAPH", "false");
    } else {
        env::set_var("MLCOMMONS_USE_CACHED_GRAPH", "true");
    }

    let batch_size = options.batch_size.unwrap_or(1);
    let (_, _, binary) = if options.conf_from_dev {
        compile_with_config_from_device(&options.model_path, batch_size, options.remove_unlower)
            .unwrap()
    } else {
        compile(&options.model_path, batch_size, options.remove_unlower).unwrap()
    };

    println!(
        "built binary from {:?} with batch size {}, remove_unlower ({}): {} bytes",
        options.model_path,
        batch_size,
        options.remove_unlower,
        binary.len()
    );
}
