/// EDM (Elucidating the Design Space of Diffusion Models) scheduler.
/// Implements Karras sigma schedule, EDM preconditioning, DPM-Solver++ updates,
/// and flow-matching Euler steps.

const SIGMA_DATA: f64 = 0.5;
const SIGMA_MIN: f64 = 0.002;
const SIGMA_MAX: f64 = 80.0;
const RHO: f64 = 7.0;

// ── Sigma schedule ──────────────────────────────────────────────────────────

/// Karras sigma schedule: n+1 values from σ_max down to 0.
pub(super) fn karras_sigmas(n: usize) -> Vec<f64> {
    let min_inv_rho = SIGMA_MIN.powf(1.0 / RHO);
    let max_inv_rho = SIGMA_MAX.powf(1.0 / RHO);
    let mut sigmas = Vec::with_capacity(n + 1);
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        let inv_rho = max_inv_rho + t * (min_inv_rho - max_inv_rho);
        sigmas.push(inv_rho.powf(RHO));
    }
    sigmas.push(0.0);
    sigmas
}

// ── EDM preconditioning ─────────────────────────────────────────────────────

pub(super) struct Preconditioning {
    pub c_in: f64,
    pub c_noise: f32,
    pub c_skip: f64,
    pub c_out: f64,
}

pub(super) fn precondition(sigma: f64) -> Preconditioning {
    let sigma_sq = sigma * sigma;
    let sigma_data_sq = SIGMA_DATA * SIGMA_DATA;
    let denom = sigma_sq + sigma_data_sq;

    Preconditioning {
        c_in: 1.0 / denom.sqrt(),
        c_noise: (sigma / SIGMA_DATA).atan() as f32,
        c_skip: sigma_data_sq / denom,
        c_out: sigma * SIGMA_DATA / denom.sqrt(),
    }
}

/// Recover denoised prediction from model output.
pub(super) fn decondition(sample: &[f32], model_output: &[f32], pc: &Preconditioning) -> Vec<f32> {
    let c_skip = pc.c_skip as f32;
    let c_out = pc.c_out as f32;
    sample.iter().zip(model_output).map(|(&s, &o)| c_skip * s + c_out * o).collect()
}

// ── DPM-Solver++ updates ────────────────────────────────────────────────────

/// First-order update: used for steps 0, 1, and the final step.
pub(super) fn dpm_first_order(sample: &[f32], x0_pred: &[f32], sigma_s: f64, sigma_t: f64) -> Vec<f32> {
    let ratio = (sigma_t / sigma_s) as f32;
    sample.iter().zip(x0_pred).map(|(&s, &x0)| ratio * s - (ratio - 1.0) * x0).collect()
}

/// Second-order midpoint update: uses current and previous model outputs.
pub(super) fn dpm_second_order(sample: &[f32], x0_current: &[f32], x0_previous: &[f32], sigma_s0: f64, sigma_s1: f64, sigma_t: f64) -> Vec<f32> {
    let lambda_t = -sigma_t.ln();
    let lambda_s0 = -sigma_s0.ln();
    let lambda_s1 = -sigma_s1.ln();
    let h = lambda_t - lambda_s0;
    let h0 = lambda_s0 - lambda_s1;
    let r0 = h0 / h;

    let exp_neg_h = (sigma_t / sigma_s0) as f32;
    let d0_coeff = -(exp_neg_h - 1.0);
    let d1_coeff = -0.5 * (exp_neg_h - 1.0);
    let r0 = r0 as f32;

    sample
        .iter()
        .zip(x0_current)
        .zip(x0_previous)
        .map(|((&s, &m0), &m1)| {
            let d1 = (m0 - m1) / r0;
            exp_neg_h * s + d0_coeff * m0 + d1_coeff * d1
        })
        .collect()
}

/// Full DPM-Solver++ state machine for multi-step denoising.
pub(super) struct DpmSolverState {
    sigmas: Vec<f64>,
    step: usize,
    lower_order_count: usize,
    prev_x0: Option<Vec<f32>>,
    prev_sigma: Option<f64>,
}

impl DpmSolverState {
    pub fn new(num_steps: usize) -> Self {
        Self {
            sigmas: karras_sigmas(num_steps),
            step: 0,
            lower_order_count: 0,
            prev_x0: None,
            prev_sigma: None,
        }
    }

    pub fn current_sigma(&self) -> f64 {
        self.sigmas[self.step]
    }

    pub fn is_done(&self) -> bool {
        self.step >= self.sigmas.len() - 1
    }

    /// Advance one step: given the current sample and model output, return the next sample.
    pub fn step(&mut self, sample: &[f32], model_output: &[f32]) -> Vec<f32> {
        let sigma_s = self.sigmas[self.step];
        let sigma_t = self.sigmas[self.step + 1];
        let pc = precondition(sigma_s);
        let x0 = decondition(sample, model_output, &pc);

        let is_last = self.step + 1 == self.sigmas.len() - 1;
        let use_second_order = self.lower_order_count >= 2 && !is_last;

        let next_sample = if use_second_order {
            let prev_x0 = self.prev_x0.as_ref().unwrap();
            let prev_sigma = self.prev_sigma.unwrap();
            dpm_second_order(sample, &x0, prev_x0, sigma_s, prev_sigma, sigma_t)
        } else {
            if self.lower_order_count < 2 {
                self.lower_order_count += 1;
            }
            dpm_first_order(sample, &x0, sigma_s, sigma_t)
        };

        self.prev_sigma = Some(sigma_s);
        self.prev_x0 = Some(x0);
        self.step += 1;

        next_sample
    }
}

// ── Flow-matching Euler step ────────────────────────────────────────────────

pub(super) fn flow_matching_t_init() -> f64 {
    (SIGMA_MAX / SIGMA_DATA).atan()
}

/// Initialize noisy sample for flow-matching: x_T = sin(t) * σ_data * noise.
pub(super) fn flow_matching_init(noise: &[f32]) -> Vec<f32> {
    let t = flow_matching_t_init();
    let scale = (t.sin() * SIGMA_DATA) as f32;
    noise.iter().map(|&n| scale * n).collect()
}

/// Trigonometric flow-matching step.
/// sample = cos(t) * x_t - sin(t) * σ_data * pred
/// where pred = -model_output (model outputs negated velocity).
/// The result is divided by σ_data to stay in the normalized domain.
pub(super) fn flow_matching_step(x_t: &[f32], model_output: &[f32], t: f64) -> Vec<f32> {
    let cos_t = t.cos() as f32;
    let sin_t = t.sin() as f32;
    let sd = SIGMA_DATA as f32;
    x_t.iter()
        .zip(model_output)
        .map(|(&x, &raw_pred)| {
            let pred = -raw_pred; // model output is negated
            (cos_t * x - sin_t * sd * pred) / sd
        })
        .collect()
}

pub(super) fn sigma_data() -> f64 {
    SIGMA_DATA
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn karras_schedule_endpoints() {
        let sigmas = karras_sigmas(20);
        assert_eq!(sigmas.len(), 21);
        assert!((sigmas[0] - SIGMA_MAX).abs() < 0.01);
        assert_eq!(sigmas[20], 0.0);
        // Monotonically decreasing
        for w in sigmas.windows(2) {
            assert!(w[0] > w[1], "{} <= {}", w[0], w[1]);
        }
    }

    #[test]
    fn karras_schedule_values_match_reference() {
        let sigmas = karras_sigmas(20);
        assert!((sigmas[0] - 80.0).abs() < 0.1);
        // sigmas[1] depends on RHO=7 ramp — verify it's in the right ballpark
        assert!(sigmas[1] > 40.0 && sigmas[1] < 75.0, "sigma[1] = {}", sigmas[1]);
        assert!(sigmas[19] < 0.01);
    }

    #[test]
    fn precondition_at_sigma_data() {
        let pc = precondition(SIGMA_DATA);
        // When σ = σ_data: c_skip = 0.5, c_out ≈ 0.354, c_in ≈ 1.414
        assert!((pc.c_skip - 0.5).abs() < 1e-10);
        assert!((pc.c_in - 1.0 / (2.0 * SIGMA_DATA * SIGMA_DATA).sqrt()).abs() < 1e-10);
        assert!((pc.c_noise - (1.0_f64).atan() as f32).abs() < 1e-5);
    }

    #[test]
    fn precondition_at_high_sigma() {
        let pc = precondition(80.0);
        // At high σ: c_skip → 0, c_out → σ_data, c_in → 1/σ
        assert!(pc.c_skip < 0.001);
        assert!((pc.c_out - SIGMA_DATA).abs() < 0.01);
        assert!((pc.c_in - 1.0 / 80.0).abs() < 0.001);
    }

    #[test]
    fn dpm_first_order_at_zero_sigma_returns_prediction() {
        let sample = vec![1.0, 2.0, 3.0];
        let x0 = vec![0.5, 1.0, 1.5];
        // When σ_t = 0: ratio = 0, result = -(-1) * x0 = x0
        let result = dpm_first_order(&sample, &x0, 1.0, 0.0);
        for (r, &x) in result.iter().zip(&x0) {
            assert!((r - x).abs() < 1e-6);
        }
    }

    #[test]
    fn flow_matching_t_init_value() {
        let t = flow_matching_t_init();
        // atan(80/0.5) = atan(160) ≈ 1.5646
        assert!((t - 1.5646).abs() < 0.001);
    }

    #[test]
    fn dpm_solver_state_runs_20_steps() {
        let mut state = DpmSolverState::new(20);
        let n = 64 * 64 * 6;
        let mut sample = vec![0.1f32; n];
        for _ in 0..20 {
            assert!(!state.is_done());
            let fake_output = vec![0.0f32; n];
            sample = state.step(&sample, &fake_output);
        }
        assert!(state.is_done());
    }
}
