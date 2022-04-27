use std::ffi::CString;
use std::path::PathBuf;

use loadgen::mlperf::{
    self, QuerySample, QuerySampleIndex, QuerySampleLatency, QuerySampleLibrary, SystemUnderTest,
    TestMode, TestScenario,
};
use nux::session::slim::Session;
use nux::SessionOptions;

use super::resnet50::{compile_resnet50, complete_a_sample, select_devices};
use crate::utils::{
    prepare_log_settings, prepare_test_settings, ImageStore, LoweredShape, RunOptions,
};

const DATA_PATH: &str = "imagenet-golden/raw/";
const TOTAL_SAMPLES: usize = 50000;

const WARMUP_RUNS: usize = 10;

/// Furiosa System Under Test
struct FuriosaSUT<'a> {
    name: CString,
    images: ImageStore,
    total_sample_count: usize,
    batch_size: usize,
    image_path: Option<PathBuf>,
    session: Session<'a>,
    lowered_output_shape: LoweredShape,
}

impl<'a> SystemUnderTest for FuriosaSUT<'a> {
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
        let outputs = self.session.run(&[self.images.slice(sample.index)]).unwrap();

        complete_a_sample_with_trace(&sample, outputs[0], &self.lowered_output_shape);
    }

    fn flush_queries(&mut self) {}

    fn report_latency_results(&mut self, _latencies_ns: &[QuerySampleLatency]) {}
}

impl<'a> FuriosaSUT<'a> {
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

        let devices = if run_options.optimal_pe_fuse { select_devices(scenario)? } else { vec![] };
        let (preprocessor, binary, lowered_output_shape) =
            compile_resnet50(&devices, run_options.batch_size)?;

        if !devices.is_empty() {
            session_options.device(devices.join(","));
        }

        let session = Session::with(&binary, &session_options)?;

        Ok(Self {
            name,
            total_sample_count,
            batch_size: run_options.batch_size,
            images: ImageStore::new(preprocessor),
            image_path,
            session,

            lowered_output_shape,
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

#[tracing::instrument(
    target = "chrome_layer",
    fields(name = "CompleteASample", cat = "Mlperf"),
    skip(output)
)]
fn complete_a_sample_with_trace(
    sample: &QuerySample,
    output: &[u8],
    lowered_output_shape: &LoweredShape,
) {
    complete_a_sample(sample, output, lowered_output_shape);
}

/// The interface a client implements to coordinate with the loadgen
/// which samples should be loaded
impl<'a> QuerySampleLibrary for FuriosaSUT<'a> {
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
        1024
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
            let filename = format!("{}ILSVRC2012_val_{:08}.JPEG.raw", data_path, wrapped_index + 1);
            self.images.load_image(index, filename, self.batch_size);
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
    config_path: PathBuf,
    session_options: SessionOptions,
    run_options: RunOptions,
) {
    let total_sample_count = if image_path.is_none() { 10 } else { TOTAL_SAMPLES };
    let model = CString::new("resnet50").unwrap();
    let test_settings = prepare_test_settings(config_path, &model, scenario, mode);
    let log_settings = prepare_log_settings("mlperf_resnet50_out");
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
    use super::*;

    use std::fs::File;
    use std::io::{BufRead, BufReader};

    #[test]
    fn unittest_mlperf_resnet50() {
        let config_path = PathBuf::from("models/mlperf_small.conf");
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
        let file = File::open("mlperf_resnet50_out/mlperf_log_summary.txt").unwrap();
        let reader = BufReader::new(file);
        assert!(reader.lines().any(|line| line.unwrap().contains("Result is : VALID")));
    }
}
