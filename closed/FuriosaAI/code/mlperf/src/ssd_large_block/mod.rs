#![allow(unused_extern_crates)]
extern crate openmp_sys;

use std::convert::TryInto;
use std::ffi::CString;
use std::fs::File;
use std::path::{Path, PathBuf};

use loadgen::mlperf::{
    self, QuerySample, QuerySampleIndex, QuerySampleLatency, QuerySampleLibrary, SystemUnderTest,
    TestMode, TestScenario,
};
use npu_executor::GraphExecutor;
use npu_ir::Buffer;
use nux::session::blocking::single_threaded::BlockingSession;
use nux::{HasModel, SessionOptions, TensorArray};

use super::ssd_large::{select_devices, CppPostprocessor};
use crate::utils::{
    compile, compile_with_config_from_device, prepare_log_settings, prepare_test_settings,
    ImageStore, Postprocess, RunOptions,
};

const DATA_PATH: &str = "coco-1200-golden/raw/";
const META_PATH: &str = "models/annotations/instances_val2017.json";
const TOTAL_SAMPLES: usize = 5000;

const WARMUP_RUNS: usize = 10;

/// Furiosa System Under Test
struct FuriosaSUT {
    name: CString,
    images: ImageStore,
    image_map: Vec<usize>,
    total_sample_count: usize,
    image_path: Option<PathBuf>,

    session: BlockingSession,
    inputs: TensorArray,
    outputs: TensorArray,
    postprocessor: CppPostprocessor,
}

impl SystemUnderTest for FuriosaSUT {
    fn name(&self) -> &std::ffi::CStr {
        self.name.as_c_str()
    }

    #[tracing::instrument(
        target = "chrome_layer",
        fields(name = "IssueQuery", cat = "Mlperf"),
        skip(self)
    )]
    fn issue_query(&mut self, samples: &[QuerySample]) {
        let sample = samples[0];
        self.inputs[0].buffer = self.images.buffer(sample.index);

        self.session.run2(&self.inputs, &mut self.outputs).unwrap();

        let mut result_data = self.postprocessor.postprocess2(sample.index as f32, &self.outputs);

        if sample.id != 0 {
            mlperf::query_samples_complete(&mut [result_data.to_query_response(sample.id)]);
        }
    }

    fn flush_queries(&mut self) {}

    fn report_latency_results(&mut self, _latencies_ns: &[QuerySampleLatency]) {}
}

impl FuriosaSUT {
    fn new(
        name: CString,
        total_sample_count: usize,
        scenario: TestScenario,
        image_path: Option<PathBuf>,
        mut session_options: SessionOptions,
        run_options: RunOptions,
    ) -> eyre::Result<Self> {
        // BlockingSession implementation is only for SingleStream
        assert!(scenario == TestScenario::SingleStream);

        let reader = File::open(META_PATH)?;
        let meta_data: serde_json::Value =
            serde_json::from_reader(reader).expect("JSON is not well-formatted");
        assert!(meta_data.is_object());
        let mut image_map: Vec<usize> = Vec::with_capacity(TOTAL_SAMPLES);
        let images = meta_data.get("images").unwrap().as_array().unwrap();
        for image in images {
            image_map.push(image.get("id").unwrap().as_u64().unwrap().try_into()?);
        }

        let devices = if run_options.optimal_pe_fuse { select_devices(scenario)? } else { vec![] };
        let (preprocessor, binary, postprocessor) = compile_ssd_large(&devices)?;
        if !devices.is_empty() {
            session_options.device(devices.join(","));
        }
        let session = BlockingSession::with(&binary, &session_options)?;
        let inputs = session.allocate_input_tensors().unwrap();
        let outputs = session.allocate_output_tensors().unwrap();

        Ok(Self {
            name,
            total_sample_count,
            image_map,
            images: ImageStore::new(preprocessor),
            image_path,
            session,
            inputs,
            outputs,
            postprocessor,
        })
    }

    fn warmup(&mut self, samples: &[QuerySampleIndex]) {
        // select a random sample from loaded samples.
        let index = samples[samples.len() / 2];
        // QuerySample.id is actually pointer in C++, so 0 can be used as a fake query.
        for _ in 0..WARMUP_RUNS {
            let samples = vec![QuerySample { id: 0, index }];
            self.issue_query(&samples);
        }
    }
}

fn compile_ssd_large(
    devices: &[String],
) -> eyre::Result<(GraphExecutor, Vec<u8>, CppPostprocessor)> {
    let (preprocessor, main, binary) = if devices.is_empty() {
        compile("models/ssd_resnet34_int8.onnx", 1, true)?
    } else {
        compile_with_config_from_device("models/ssd_resnet34_int8.onnx", 1, true)?
    };
    Ok((preprocessor, binary, CppPostprocessor::new(&main)))
}

#[tracing::instrument(
    target = "chrome_layer",
    fields(name = "CompleteASample", cat = "Mlperf"),
    skip(outputs, pp)
)]
fn complete_a_sample(query_sample: QuerySample, outputs: &[Buffer], pp: &impl Postprocess) {
    let data = outputs.iter().map(|b| b.as_slice::<u8>()).collect::<Vec<_>>();
    let mut result_data = pp.postprocess(query_sample.index as f32, &data);
    if query_sample.id != 0 {
        mlperf::query_samples_complete(&mut [result_data.to_query_response(query_sample.id)]);
    }
}

/// The interface a client implements to coordinate with the loadgen
/// which samples should be loaded
impl QuerySampleLibrary for FuriosaSUT {
    /// A human readable name for the model
    fn name(&self) -> &std::ffi::CStr {
        self.name.as_c_str()
    }

    /// Total number of samples in library
    fn total_sample_count(&self) -> usize {
        TOTAL_SAMPLES
    }

    /// The number of samples that are guaranteed to fit in RAM
    fn performance_sample_count(&self) -> usize {
        // This value is overridden by the value in TestSettings.performance_sample_count_override if non-zero.
        // https://github.com/mlcommons/inference/blob/56c0d7a816c7f67077fa7651a3584cac116e0633/loadgen/test_settings_internal.cc#L115
        // Value brought from https://github.com/mlcommons/inference/blob/master/mlperf.conf
        64
    }

    /// Loads the requested query samples into memory
    /// Paired with calls to UnloadSamplesFromRam
    /// A previously loaded sample will not be loaded again
    #[tracing::instrument(
        target = "chrome_layer",
        fields(name = "LoadSamplesToRam", cat = "Mlperf"),
        skip(self, samples)
    )]
    fn load_samples_to_ram(&mut self, samples: &[QuerySampleIndex]) {
        for &index in samples {
            // In test environment, we may have only small number of samples with low index numbers.
            // To load samples up to performance sample count and to measure the memory and cache performance,
            // reload same files to another buffer even if self.total_sample_count < TOTAL_SAMPLES.
            // Accuracy test will fail in that case.
            let wrapped_index = index % self.total_sample_count;
            let data_path =
                if let Some(path) = &self.image_path { path.to_str().unwrap() } else { DATA_PATH };
            let filename = format!("{}{:012}.jpg.raw", data_path, self.image_map[wrapped_index]);
            self.images.load_image(index, filename, 1);
        }
        self.warmup(samples);
    }

    /// Unloads the requested query samples from memory
    /// A previously unloaded sample will not be unloaded
    #[tracing::instrument(
        target = "chrome_layer",
        fields(name = "UnloadSamplesFromRam", cat = "Mlperf"),
        skip(self, samples)
    )]
    fn unload_samples_from_ram(&mut self, samples: &[QuerySampleIndex]) {
        for &index in samples {
            self.images.unload_image(index);
        }
    }
}

pub fn run(
    image_path: Option<PathBuf>,
    scenario: TestScenario,
    mode: TestMode,
    config_path: impl AsRef<Path>,
    session_options: SessionOptions,
    run_options: RunOptions,
) {
    let total_sample_count = if image_path.is_none() { 10 } else { TOTAL_SAMPLES };
    let model = CString::new("ssd-resnet34").unwrap();
    let test_settings = prepare_test_settings(config_path, &model, scenario, mode);
    let log_settings = prepare_log_settings("mlperf_ssd_large_out");
    let enable_tokio_console = run_options.enable_tokio_console;

    let mut sut = FuriosaSUT::new(
        model,
        total_sample_count,
        scenario,
        image_path,
        session_options,
        run_options,
    )
    .unwrap();

    eprintln!("start test");
    if enable_tokio_console {
        console_subscriber::init();
        mlperf::start_test_ex(&mut sut, &test_settings, &log_settings);
        drop(sut);
    } else {
        mlperf::start_test_ex(&mut sut, &test_settings, &log_settings);
        drop(sut);
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader};

    use super::*;

    #[test]
    fn unittest_mlperf_ssd_large() {
        let config_path = "models/mlperf_small.conf";
        run(
            None,
            TestScenario::SingleStream,
            TestMode::PerformanceOnly,
            config_path,
            SessionOptions::default(),
            RunOptions {
                optimal_pe_fuse: true,
                batch_size: 1,
                enable_tokio_console: false,
                ..Default::default()
            },
        );
        let file = File::open("mlperf_ssd_large_out/mlperf_log_summary.txt").unwrap();
        let reader = BufReader::new(file);
        assert!(reader.lines().any(|line| line.unwrap().contains("Result is : VALID")));
    }
}
