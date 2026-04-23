/// ONNX Runtime session management. Loads models with CUDA GPU support,
/// falling back to CPU automatically. Only one model is loaded at a time
/// to minimize VRAM usage.

use std::path::Path;
use std::sync::Once;
use std::time::Instant;

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Tensor;

static INIT_ORT: Once = Once::new();

/// Add the onnxruntime DLL directory to the process PATH so ORT can find
/// CUDA/cuDNN DLLs at runtime. Called once before the first session load.
fn ensure_ort_path() {
    INIT_ORT.call_once(|| {
        let ort_dir = std::path::PathBuf::from("data/onnxruntime");
        if ort_dir.exists() {
            if let Ok(current) = std::env::var("PATH") {
                let abs = std::fs::canonicalize(&ort_dir)
                    .unwrap_or_else(|_| ort_dir.clone());
                std::env::set_var("PATH", format!("{};{}", abs.display(), current));
                log_to_file(&format!("[session] added {} to PATH", abs.display()));
            }
        }
    });
}

pub(super) struct ModelSession {
    session: Session,
    name: String,
    run_count: u32,
}

impl ModelSession {
    pub fn load(path: &Path) -> Result<Self> {
        ensure_ort_path();
        let t0 = Instant::now();
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default().build(),
            ])
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(path)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("failed to load ONNX model: {}", path.display()))?;

        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        // Detect which execution provider is active
        let provider = detect_provider(&session);
        log_to_file(&format!(
            "[session] loaded {} in {:.2}s — provider: {}",
            name, t0.elapsed().as_secs_f64(), provider,
        ));
        Ok(Self { session, name, run_count: 0 })
    }

    pub fn run(
        &mut self,
        inputs: Vec<(&str, Vec<usize>, Vec<f32>)>,
    ) -> Result<Vec<f32>> {
        let t0 = Instant::now();
        let mut owned_tensors: Vec<Tensor<f32>> = Vec::with_capacity(inputs.len());
        let mut names: Vec<String> = Vec::with_capacity(inputs.len());

        for (name, shape, data) in inputs {
            let shape_i64: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
            let tensor = Tensor::from_array((shape_i64, data.into_boxed_slice()))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            names.push(name.to_string());
            owned_tensors.push(tensor);
        }

        let feed: Vec<(String, &Tensor<f32>)> = names
            .into_iter()
            .zip(owned_tensors.iter())
            .collect();

        let outputs = self
            .session
            .run(feed)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let first = outputs
            .iter()
            .next()
            .context("model produced no outputs")?;

        let (_shape, data) = first
            .1
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        self.run_count += 1;
        let elapsed = t0.elapsed().as_secs_f64();
        if self.run_count <= 3 || self.run_count % 10 == 0 {
            log_to_file(&format!(
                "[inference] {} run #{} in {:.3}s",
                self.name, self.run_count, elapsed
            ));
        }

        Ok(data.to_vec())
    }
}

impl Drop for ModelSession {
    fn drop(&mut self) {
        log_to_file(&format!(
            "[session] dropping {} after {} runs",
            self.name, self.run_count
        ));
    }
}

pub(super) fn load_model(dir: &Path, name: &str) -> Result<ModelSession> {
    let path = dir.join(format!("{name}.onnx"));
    ModelSession::load(&path)
}

fn detect_provider(session: &Session) -> &'static str {
    // ORT doesn't expose which EP is active directly, but we can check
    // metadata — if CUDA registered successfully, inputs will be on GPU.
    // Simplest heuristic: check if CUDAExecutionProvider appears in the
    // session's provider list via the C API. Since ort doesn't expose this
    // cleanly, we infer from load time + available DLLs.
    let has_cudnn = std::path::Path::new("data/onnxruntime/cudnn64_9.dll").exists()
        || which_in_path("cudnn64_9.dll");
    let has_cuda_rt = which_in_path("cudart64_12.dll")
        || std::path::Path::new("data/onnxruntime/cudart64_12.dll").exists();

    if has_cudnn && has_cuda_rt {
        "CUDA (cudnn found)"
    } else if has_cuda_rt {
        "CUDA (WARNING: cudnn NOT found — will fall back to CPU)"
    } else {
        "CPU (no CUDA runtime found)"
    }
}

fn which_in_path(dll: &str) -> bool {
    if let Ok(path) = std::env::var("PATH") {
        for dir in path.split(';') {
            if std::path::Path::new(dir).join(dll).exists() {
                return true;
            }
        }
    }
    false
}

pub(super) fn log_to_file(msg: &str) {
    use std::io::Write;
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();
    let line = format!("{:.3} {}\n", timestamp, msg);
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("terrain_amplification.log")
    {
        let _ = f.write_all(line.as_bytes());
    }
}
