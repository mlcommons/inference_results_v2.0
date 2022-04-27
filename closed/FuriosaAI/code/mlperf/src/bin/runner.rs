use std::path::PathBuf;

use structopt::StructOpt;

use loadgen::mlperf::{TestMode, TestScenario};
use mlperf::utils::{init_logger, init_nux_profiler, PostProcessor, RunOptions};
use mlperf::{resnet50, resnet50_block, ssd_large, ssd_large_block, ssd_small, ssd_small_block};
use nux::SessionOptions;

#[derive(StructOpt)]
enum Models {
    Resnet50,
    Resnet50Block,
    SSDSmall,
    SSDSmallBlock,
    SSDLarge,
    SSDLargeBlock,
}

#[allow(clippy::match_str_case_mismatch)] // Clippy doesn't handle "RESNET50" properly.
fn parse_model(src: &str) -> Models {
    match src.to_uppercase().as_str() {
        "RESNET50" => Models::Resnet50,
        "RESNET50BLOCK" | "RESNET50_BLOCK" => Models::Resnet50Block,
        "SSDSMALL" | "SSD_SMALL" => Models::SSDSmall,
        "SSDSMALLBLOCK" | "SSD_SMALL_BLOCK" => Models::SSDSmallBlock,
        "SSDLARGE" | "SSD_LARGE" => Models::SSDLarge,
        "SSDLARGEBLOCK" | "SSD_LARGE_BLOCK" => Models::SSDLargeBlock,
        _ => panic!("model should be one of resnet50, ssd_small, and ssd_large (or resnet50_block, ssd_small_block, ssd_large_block)"),
    }
}

fn parse_scenario(src: &str) -> TestScenario {
    match src.to_uppercase().as_str() {
        "OFFLINE" => TestScenario::Offline,
        "MULTISTREAM" => TestScenario::MultiStream,
        _ => TestScenario::SingleStream,
    }
}

fn parse_mode(src: &str) -> TestMode {
    match src.to_uppercase().as_str() {
        "ACCURACYONLY" => TestMode::AccuracyOnly,
        _ => TestMode::PerformanceOnly,
    }
}

fn parse_post_processor(src: &str) -> PostProcessor {
    match src.to_uppercase().as_str() {
        "CPP" | "CPP_PAR" => PostProcessor::CppPar,
        "RUST_PAR" => PostProcessor::RustPar,
        "RUST" => PostProcessor::Rust,
        _ => panic!("post-processor should be one of cpp_par, cpp, rust_par, and rust"),
    }
}

#[derive(StructOpt)]
struct RunnerOptions {
    /// Sets test model (RESNET50, SSD_SMALL, and SSD_LARGE)
    #[structopt(short, long, parse(from_str = parse_model))]
    model: Models,

    /// Sets the mlperf.conf file path
    #[structopt(short, long, parse(from_os_str))]
    config: PathBuf,

    /// Sets test scenario (SingleStream and Offline) [default: SingleStream]
    #[structopt(short, long, parse(from_str = parse_scenario))]
    scenario: Option<TestScenario>,

    /// Sets test model (PerformanceOnly and AccuracyOnly) [default: PerformanceOnly]
    #[structopt(long, parse(from_str = parse_mode))]
    mode: Option<TestMode>,

    /// Sets the input data path [default: one of the following directories (imagenet-golden, coco-300-golden, and coco-1200-golden) according to the selected model]
    #[structopt(short, long, parse(from_os_str))]
    input: Option<PathBuf>,

    /// Sets the number of workers [default: 1]
    #[structopt(long = "workers")]
    worker_num: Option<usize>,

    /// Sets Input queue size [default: None]
    #[structopt(long = "input-queue-size")]
    input_queue_size: Option<usize>,

    /// Sets Output queue size [default: None]
    #[structopt(long = "output-queue-size")]
    output_queue_size: Option<usize>,

    /// Selects optimal PE fuse for each model and scenario [default: false]
    #[structopt(long = "optimal-pe-fuse")]
    optimal_pe_fuse: bool,

    /// Sets the batch size [default: 1]
    #[structopt(long = "batch-size")]
    batch_size: Option<usize>,

    /// Enables Tokio Console, which automatically disables the profiler [default: false]
    #[structopt(long = "enable-tokio-console")]
    enable_tokio_console: bool,

    #[structopt(long = "post-processor", parse(from_str = parse_post_processor))]
    post_processor: Option<PostProcessor>,
}

fn main() {
    let matches = RunnerOptions::clap().get_matches();
    let options = RunnerOptions::from_clap(&matches);
    let scenario =
        if let Some(scenario) = options.scenario { scenario } else { TestScenario::SingleStream };
    let mode = if let Some(mode) = options.mode { mode } else { TestMode::PerformanceOnly };

    let mut session_options = SessionOptions::default();
    if let Some(worker_num) = options.worker_num {
        session_options.worker_num(worker_num);
    }
    if let Some(input_queue_size) = options.input_queue_size {
        session_options.input_queue_size(input_queue_size);
    }
    if let Some(output_queue_size) = options.output_queue_size {
        session_options.output_queue_size(output_queue_size);
    }

    let mut run_options = RunOptions::default();
    run_options.optimal_pe_fuse(options.optimal_pe_fuse);
    run_options.batch_size(options.batch_size.unwrap_or(1));
    run_options.enable_tokio_console = options.enable_tokio_console;
    run_options.post_processor =
        options.post_processor.unwrap_or(if scenario == TestScenario::Offline {
            PostProcessor::Rust
        } else {
            PostProcessor::CppPar
        });
    run_options.worker_num(options.worker_num.unwrap_or(1));

    let _guard = init_nux_profiler();
    if _guard.is_none() {
        init_logger();
    }

    match (options.model, scenario) {
        (Models::Resnet50Block, _) | (Models::Resnet50, TestScenario::SingleStream) => {
            resnet50_block::run(
                options.input,
                scenario,
                mode,
                options.config,
                session_options,
                run_options,
            )
        }
        (Models::Resnet50, _) => resnet50::run(
            options.input,
            scenario,
            mode,
            options.config,
            session_options,
            run_options,
        ),
        (Models::SSDSmall, _) => ssd_small::run(
            options.input,
            scenario,
            mode,
            options.config,
            session_options,
            run_options,
        ),
        (Models::SSDSmallBlock, _) => ssd_small_block::run(
            options.input,
            scenario,
            mode,
            options.config,
            session_options,
            run_options,
        ),
        (Models::SSDLargeBlock, _) | (Models::SSDLarge, TestScenario::SingleStream) => {
            ssd_large_block::run(
                options.input,
                scenario,
                mode,
                options.config,
                session_options,
                run_options,
            )
        }
        (Models::SSDLarge, _) => ssd_large::run(
            options.input,
            scenario,
            mode,
            options.config,
            session_options,
            run_options,
        ),
    }
}
