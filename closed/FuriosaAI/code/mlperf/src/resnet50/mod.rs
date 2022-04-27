use std::ffi::CString;
use std::path::PathBuf;
use std::sync::Arc;

use loadgen::mlperf::{
    self, QuerySample, QuerySampleIndex, QuerySampleLatency, QuerySampleLibrary,
    QuerySampleResponse, SystemUnderTest, TestMode, TestScenario,
};
use npu_executor::GraphExecutor;
use nux::async_nux::session::{self, Input, RawSession};
use nux::SessionOptions;

use crate::utils::{
    compile, compile_with_config_from_device, prepare_log_settings, prepare_test_settings,
    run_warmup, select_fused_device, select_non_fused_devices, ImageStore, LoweredShape,
    RunOptions,
};

const DATA_PATH: &str = "imagenet-golden/raw/";
const TOTAL_SAMPLES: usize = 50000;

const WARMUP_RUNS: usize = 10;

/// Furiosa System Under Test
struct FuriosaSUT {
    runtime: Arc<tokio::runtime::Runtime>,
    name: CString,
    images: ImageStore,
    total_sample_count: usize,
    batch_size: usize,
    image_path: Option<PathBuf>,
    scenario: TestScenario,
    lowered_output_shape: LoweredShape,
    raw_session: Option<RawSession>,
    workload_tx: Option<crossbeam::channel::Sender<(Vec<Input>, Vec<QuerySample>)>>,
    worker_threads: Vec<Option<std::thread::JoinHandle<()>>>,
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
        assert!(self.scenario != TestScenario::Server);
        let runtime = self.runtime.clone();
        for chunked_samples in samples.chunks(self.batch_size) {
            let input = self.images.prepare_inputs(self.batch_size, chunked_samples);
            if let Some(raw_session) = self.raw_session.as_mut() {
                runtime.block_on(raw_session.run(input)).unwrap();
                let output = raw_session.outputs().next().unwrap().as_slice::<u8>();
                complete_batch(
                    chunked_samples,
                    output,
                    self.batch_size,
                    &self.lowered_output_shape,
                );
            } else {
                self.workload_tx.as_ref().unwrap().send((input, chunked_samples.to_vec())).unwrap();
            }
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
        let runtime = Arc::new(tokio::runtime::Builder::new_current_thread().enable_all().build()?);
        let devices = if run_options.optimal_pe_fuse { select_devices(scenario)? } else { vec![] };
        let (preprocessor, binary, lowered_output_shape) =
            compile_resnet50(&devices, run_options.batch_size)?;

        if !devices.is_empty() {
            session_options.device(devices.join(","));
        }
        eprintln!("create channel session with options: {:?}", session_options);
        let mut raw_sessions =
            runtime.block_on(session::create_raw_session(&binary, &session_options))?;

        let (raw_session, worker_threads, workload_tx) = if raw_sessions.len() > 1 {
            let (workload_tx, workload_rx) =
                crossbeam::channel::bounded(run_options.worker_num * 5);
            let worker_threads = raw_sessions
                .into_iter()
                .map(|raw_session| {
                    let cloned_workload_rx = workload_rx.clone();
                    Some(std::thread::spawn(move || {
                        work_on_thread(
                            raw_session,
                            run_options.batch_size,
                            cloned_workload_rx,
                            lowered_output_shape,
                        )
                    }))
                })
                .collect::<Vec<_>>();
            (None, worker_threads, Some(workload_tx))
        } else {
            (raw_sessions.pop(), vec![], None)
        };

        Ok(Self {
            runtime,
            name,
            total_sample_count,
            batch_size: run_options.batch_size,
            images: ImageStore::new(preprocessor),
            image_path,
            scenario,
            lowered_output_shape,
            raw_session,
            workload_tx,
            worker_threads,
        })
    }

    fn warmup(&mut self, samples: &[QuerySampleIndex]) {
        // select a random sample from loaded samples.
        let index = samples[samples.len() / 2];
        // QuerySample.id is actually pointer in C++, so 0 can be used as a fake query.
        let samples = (0..(WARMUP_RUNS * self.batch_size))
            .map(|_| QuerySample { id: 0, index })
            .collect::<Vec<_>>();
        run_warmup(self, &samples, self.batch_size, self.scenario);
    }
}

impl Drop for FuriosaSUT {
    fn drop(&mut self) {
        if let Some(raw_session) = self.raw_session.take() {
            drop(raw_session);
        }
        if let Some(workload_tx) = self.workload_tx.take() {
            drop(workload_tx);
        }
        for thread in &mut self.worker_threads {
            if let Some(handle) = thread.take() {
                handle.join().unwrap();
            }
        }
    }
}

pub fn select_devices(scenario: TestScenario) -> eyre::Result<Vec<String>> {
    // Override NPU_DEVNAME environment variable according to the scenario for the optimal performance.
    if scenario == TestScenario::SingleStream {
        return Ok(vec![select_fused_device()?]);
    } else if scenario == TestScenario::Offline {
        return select_non_fused_devices();
    } else if scenario == TestScenario::MultiStream {
        return Ok(vec![select_fused_device()?]);
    }
    eyre::bail!("no optimal device selection supported for the scenario")
}

pub fn compile_resnet50(
    devices: &[String],
    batch_size: usize,
) -> eyre::Result<(GraphExecutor, Vec<u8>, LoweredShape)> {
    let (preprocessor, main, binary) = if devices.is_empty() {
        compile("models/resnet50_int8.onnx", batch_size, true)?
    } else {
        compile_with_config_from_device("models/resnet50_int8.onnx", batch_size, true)?
    };
    assert_eq!(main.outputs.len(), 1, "number of output tensors should be 1");
    let lowered_output_shape = LoweredShape::from(&main[main.outputs[0]].shape);
    Ok((preprocessor, binary, lowered_output_shape))
}

fn work_on_thread(
    mut raw_session: RawSession,
    batch_size: usize,
    workload_rx: crossbeam::channel::Receiver<(Vec<Input>, Vec<QuerySample>)>,
    lowered_output_shape: LoweredShape,
) {
    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    while let Ok((inputs, samples)) = workload_rx.recv() {
        if let Err(e) = runtime.block_on(raw_session.run(inputs)) {
            panic!("Failed to inference, {:?}", e);
        }
        let output = raw_session.outputs().next().unwrap().as_slice::<u8>();
        complete_batch(&samples, output, batch_size, &lowered_output_shape);
    }
}

#[tracing::instrument(
    target = "chrome_layer",
    fields(name = "CompleteBatch", cat = "Mlperf"),
    skip(output, lowered_output_shape)
)]
fn complete_batch(
    query_samples: &[QuerySample],
    output: &[u8],
    batch_size: usize,
    lowered_output_shape: &LoweredShape,
) {
    for (sample, output) in
        itertools::izip!(query_samples, output.chunks(output.len() / batch_size))
    {
        complete_a_sample(sample, output, lowered_output_shape)
    }
}

pub fn complete_a_sample(sample: &QuerySample, output: &[u8], lowered_output_shape: &LoweredShape) {
    let max_index =
        (0..1001).max_by_key(|&c| output[lowered_output_shape.index(c, 0, 0)] as i8).unwrap();
    let result_data = max_index as f32;
    let data = &result_data as *const f32 as usize;
    if sample.id != 0 {
        mlperf::query_samples_complete(&mut [QuerySampleResponse {
            id: sample.id,
            data,
            size: std::mem::size_of::<f32>(),
        }]);
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
    }
    mlperf::start_test_ex(&mut sut, &test_settings, &log_settings);
    drop(sut);
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
