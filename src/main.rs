// main.rs

use nalgebra::{Cholesky, Matrix3, Matrix6, SMatrix, Vector3, Vector6};
use plotters::prelude::*;
use plotters::prelude::full_palette::{BLUE, CYAN, GREEN, GREY, ORANGE, PURPLE, RED};
use rand::prelude::*;
use std::f64::consts::PI;
use std::fs::create_dir_all;
use std::process::Command;

// --- 共通の関数とモデル ---
const STATE_DIM: usize = 6;
const OBS_DIM: usize = 3;

fn state_transition_function(x_prev: &Vector6<f64>, dt: f64) -> Vector6<f64> {
    let mut f = Matrix6::identity();
    f[(0, 3)] = dt;
    f[(1, 4)] = dt;
    f[(2, 5)] = dt;
    f * x_prev
}

fn observation_function(x_curr: &Vector6<f64>) -> Vector3<f64> {
    let px = x_curr[0];
    let py = x_curr[1];
    let pz = x_curr[2];
    
    let range = (px.powi(2) + py.powi(2) + pz.powi(2)).sqrt();
    let azimuth = py.atan2(px);
    let elevation = pz.atan2((px.powi(2) + py.powi(2)).sqrt());
    
    Vector3::new(range, azimuth, elevation)
}

// --- 各フィルタの実装 ---
struct ExtendedKalmanFilter {
    x: Vector6<f64>,
    p: Matrix6<f64>,
    q: Matrix6<f64>,
    r: Matrix3<f64>,
    dt: f64,
}
impl ExtendedKalmanFilter {
    pub fn new(initial_x: Vector6<f64>, initial_p: Matrix6<f64>, q: Matrix6<f64>, r: Matrix3<f64>, dt: f64) -> Self {
        Self { x: initial_x, p: initial_p, q, r, dt }
    }

    fn calculate_jacobian_h(&self, x_pred: &Vector6<f64>) -> SMatrix<f64, OBS_DIM, STATE_DIM> {
        let px = x_pred[0];
        let py = x_pred[1];
        let pz = x_pred[2];

        let d_sq = px.powi(2) + py.powi(2);
        let d = d_sq.sqrt();
        let r_sq = d_sq + pz.powi(2);
        let r = r_sq.sqrt();

        let mut h = SMatrix::<f64, OBS_DIM, STATE_DIM>::zeros();

        if r.abs() < 1e-6 { return h; }

        h[(0, 0)] = px / r;
        h[(0, 1)] = py / r;
        h[(0, 2)] = pz / r;
        
        if d_sq.abs() < 1e-6 { return h; }
        
        h[(1, 0)] = -py / d_sq;
        h[(1, 1)] = px / d_sq;
        
        h[(2, 0)] = -px * pz / (d * r_sq);
        h[(2, 1)] = -py * pz / (d * r_sq);
        h[(2, 2)] = d / r_sq;

        h
    }
    
    pub fn predict(&mut self) {
        let mut f = Matrix6::identity();
        f[(0, 3)] = self.dt;
        f[(1, 4)] = self.dt;
        f[(2, 5)] = self.dt;

        self.x = f * self.x;
        self.p = f * self.p * f.transpose() + self.q;
    }

    pub fn update(&mut self, z: Vector3<f64>) {
        let h_jacobian = self.calculate_jacobian_h(&self.x);
        let y = z - observation_function(&self.x);
        let s = h_jacobian * self.p * h_jacobian.transpose() + self.r;

        if let Some(s_inv) = s.try_inverse() {
            let k = self.p * h_jacobian.transpose() * s_inv;
            self.x += k * y;
            self.p = (Matrix6::identity() - k * h_jacobian) * self.p;
        }
    }
}

struct UnscentedKalmanFilter {
    x: Vector6<f64>,
    p: Matrix6<f64>,
    q: Matrix6<f64>,
    r: Matrix3<f64>,
    dt: f64,
    weights_m: SMatrix<f64, 1, { 2 * STATE_DIM + 1 }>,
    weights_c: SMatrix<f64, 1, { 2 * STATE_DIM + 1 }>,
    lambda: f64,
    last_innovation: Vector3<f64>,
    last_innovation_covariance: Matrix3<f64>,
}
impl UnscentedKalmanFilter {
    pub fn new(initial_x: Vector6<f64>, initial_p: Matrix6<f64>, q: Matrix6<f64>, r: Matrix3<f64>, dt: f64) -> Self {
        let n = STATE_DIM as f64;
        let alpha: f64 = 1e-3;
        let kappa = 0.0;
        let beta = 2.0;
        let lambda = alpha.powi(2) * (n + kappa) - n;
        let mut weights_m = SMatrix::<f64, 1, { 2 * STATE_DIM + 1 }>::zeros();
        let mut weights_c = SMatrix::<f64, 1, { 2 * STATE_DIM + 1 }>::zeros();
        weights_m[(0, 0)] = lambda / (n + lambda);
        weights_c[(0, 0)] = lambda / (n + lambda) + (1.0 - alpha.powi(2) + beta);
        for i in 1..(2 * STATE_DIM + 1) {
            weights_m[(0, i)] = 0.5 / (n + lambda);
            weights_c[(0, i)] = 0.5 / (n + lambda);
        }
        Self { x: initial_x, p: initial_p, q, r, dt, weights_m, weights_c, lambda, last_innovation: Vector3::zeros(), last_innovation_covariance: Matrix3::zeros() }
    }
    fn generate_sigma_points(&self) -> Option<SMatrix<f64, STATE_DIM, { 2 * STATE_DIM + 1 }>> {
        Cholesky::new((STATE_DIM as f64 + self.lambda) * self.p).map(|chol| {
            let p_sqrt = chol.l();
            let mut sigma_points = SMatrix::<f64, STATE_DIM, { 2 * STATE_DIM + 1 }>::zeros();
            sigma_points.set_column(0, &self.x);
            for i in 0..STATE_DIM {
                sigma_points.set_column(i + 1, &(self.x + p_sqrt.column(i)));
                sigma_points.set_column(i + 1 + STATE_DIM, &(self.x - p_sqrt.column(i)));
            }
            sigma_points
        })
    }
    pub fn predict(&mut self) {
        if let Some(sigma_points) = self.generate_sigma_points() {
            let mut predicted_sigma_points = SMatrix::<f64, STATE_DIM, { 2 * STATE_DIM + 1 }>::zeros();
            for i in 0..(2 * STATE_DIM + 1) {
                predicted_sigma_points.set_column(i, &state_transition_function(&sigma_points.column(i).into(), self.dt));
            }
            self.x = predicted_sigma_points * self.weights_m.transpose();
            let mut p_pred = Matrix6::zeros();
            for i in 0..(2 * STATE_DIM + 1) {
                let diff = predicted_sigma_points.column(i) - self.x;
                p_pred += self.weights_c[(0, i)] * (diff * diff.transpose());
            }
            self.p = p_pred + self.q;
        }
    }
    pub fn update(&mut self, z: Vector3<f64>) {
        if let Some(sigma_points_pred) = self.generate_sigma_points() {
            let mut observed_sigma_points = SMatrix::<f64, OBS_DIM, { 2 * STATE_DIM + 1 }>::zeros();
            for i in 0..(2 * STATE_DIM + 1) {
                observed_sigma_points.set_column(i, &observation_function(&sigma_points_pred.column(i).into()));
            }
            let z_pred = observed_sigma_points * self.weights_m.transpose();
            let mut s = Matrix3::zeros();
            let mut t = SMatrix::<f64, STATE_DIM, OBS_DIM>::zeros();
            for i in 0..(2 * STATE_DIM + 1) {
                let z_diff = observed_sigma_points.column(i) - z_pred;
                let x_diff = sigma_points_pred.column(i) - self.x;
                s += self.weights_c[(0, i)] * (z_diff * z_diff.transpose());
                t += self.weights_c[(0, i)] * (x_diff * z_diff.transpose());
            }
            s += self.r;
            let innovation = z - z_pred;
            if let Some(s_inv) = s.try_inverse() {
                let k = t * s_inv;
                self.x += k * innovation;
                self.p -= k * s * k.transpose();
            }
            self.last_innovation = innovation;
            self.last_innovation_covariance = s;
        }
    }
}

struct CubatureKalmanFilter {
    x: Vector6<f64>,
    p: Matrix6<f64>,
    q: Matrix6<f64>,
    r: Matrix3<f64>,
    dt: f64,
}
impl CubatureKalmanFilter {
    pub fn new(initial_x: Vector6<f64>, initial_p: Matrix6<f64>, q: Matrix6<f64>, r: Matrix3<f64>, dt: f64) -> Self {
        Self { x: initial_x, p: initial_p, q, r, dt }
    }
    fn generate_cubature_points(&self, x: &Vector6<f64>, p: &Matrix6<f64>) -> Option<SMatrix<f64, STATE_DIM, { 2 * STATE_DIM }>> {
        Cholesky::new(*p).map(|chol| {
            let s = chol.l();
            let mut points = SMatrix::<f64, STATE_DIM, { 2 * STATE_DIM }>::zeros();
            let factor = (STATE_DIM as f64).sqrt();
            for i in 0..STATE_DIM {
                let s_col = s.column(i);
                points.set_column(i, &(x + factor * s_col));
                points.set_column(i + STATE_DIM, &(x - factor * s_col));
            }
            points
        })
    }
    pub fn predict(&mut self) {
        if let Some(points) = self.generate_cubature_points(&self.x, &self.p) {
            let mut propagated_points = SMatrix::<f64, STATE_DIM, { 2 * STATE_DIM }>::zeros();
            for i in 0..(2 * STATE_DIM) {
                propagated_points.set_column(i, &state_transition_function(&points.column(i).into(), self.dt));
            }
            self.x = propagated_points.column_mean();
            let mut p_pred = Matrix6::zeros();
            for i in 0..(2 * STATE_DIM) {
                let diff = propagated_points.column(i) - self.x;
                p_pred += diff * diff.transpose();
            }
            self.p = p_pred / (2.0 * STATE_DIM as f64) + self.q;
        }
    }
    pub fn update(&mut self, z: Vector3<f64>) {
        if let Some(points) = self.generate_cubature_points(&self.x, &self.p) {
            let mut observed_points = SMatrix::<f64, OBS_DIM, { 2 * STATE_DIM }>::zeros();
            for i in 0..(2 * STATE_DIM) {
                observed_points.set_column(i, &observation_function(&points.column(i).into()));
            }
            let z_pred: Vector3<f64> = observed_points.column_mean();
            let mut p_zz_innovation = Matrix3::zeros();
            let mut p_xz = SMatrix::<f64, STATE_DIM, OBS_DIM>::zeros();
            for i in 0..(2 * STATE_DIM) {
                let z_diff = observed_points.column(i) - z_pred;
                let x_diff = points.column(i) - self.x;
                p_zz_innovation += z_diff * z_diff.transpose();
                p_xz += x_diff * z_diff.transpose();
            }
            let weight = 1.0 / (2.0 * STATE_DIM as f64);
            let p_zz_innovation = weight * p_zz_innovation;
            let p_xz = weight * p_xz;
            let p_zz = p_zz_innovation + self.r;
            if let Some(p_zz_inv) = p_zz.try_inverse() {
                let k = p_xz * p_zz_inv;
                self.x += k * (z - z_pred);
                self.p -= k * p_zz * k.transpose();
            }
        }
    }
}

struct RobustCubatureKalmanFilter {
    x: Vector6<f64>,
    p: Matrix6<f64>,
    q: Matrix6<f64>,
    r: Matrix3<f64>,
    dt: f64,
    mahalanobis_threshold: f64,
}
impl RobustCubatureKalmanFilter {
    pub fn new(initial_x: Vector6<f64>, initial_p: Matrix6<f64>, q: Matrix6<f64>, r: Matrix3<f64>, dt: f64) -> Self {
        Self { x: initial_x, p: initial_p, q, r, dt, mahalanobis_threshold: 7.815 }
    }
    fn generate_cubature_points(&self, x: &Vector6<f64>, p: &Matrix6<f64>) -> Option<SMatrix<f64, STATE_DIM, { 2 * STATE_DIM }>> {
        Cholesky::new(*p).map(|chol| {
            let s = chol.l();
            let mut points = SMatrix::<f64, STATE_DIM, { 2 * STATE_DIM }>::zeros();
            let factor = (STATE_DIM as f64).sqrt();
            for i in 0..STATE_DIM {
                points.set_column(i, &(x + factor * s.column(i)));
                points.set_column(i + STATE_DIM, &(x - factor * s.column(i)));
            }
            points
        })
    }
    pub fn predict(&mut self) {
        if let Some(points) = self.generate_cubature_points(&self.x, &self.p) {
            let mut propagated_points = SMatrix::<f64, STATE_DIM, { 2 * STATE_DIM }>::zeros();
            for i in 0..(2 * STATE_DIM) {
                propagated_points.set_column(i, &state_transition_function(&points.column(i).into(), self.dt));
            }
            self.x = propagated_points.column_mean();
            let mut p_pred = Matrix6::zeros();
            for i in 0..(2 * STATE_DIM) {
                let diff = propagated_points.column(i) - self.x;
                p_pred += diff * diff.transpose();
            }
            self.p = p_pred / (2.0 * STATE_DIM as f64) + self.q;
        }
    }
    pub fn update(&mut self, z: Vector3<f64>) {
        if let Some(points) = self.generate_cubature_points(&self.x, &self.p) {
            let mut observed_points = SMatrix::<f64, OBS_DIM, { 2 * STATE_DIM }>::zeros();
            for i in 0..(2 * STATE_DIM) {
                observed_points.set_column(i, &observation_function(&points.column(i).into()));
            }
            let z_pred: Vector3<f64> = observed_points.column_mean();
            let mut p_zz_innovation = Matrix3::zeros();
            let mut p_xz = SMatrix::<f64, STATE_DIM, OBS_DIM>::zeros();
            let weight = 1.0 / (2.0 * STATE_DIM as f64);
            for i in 0..(2 * STATE_DIM) {
                let z_diff = observed_points.column(i) - z_pred;
                let x_diff = points.column(i) - self.x;
                p_zz_innovation += z_diff * z_diff.transpose();
                p_xz += x_diff * z_diff.transpose();
            }
            let p_zz_innovation = weight * p_zz_innovation;
            let p_xz = weight * p_xz;
            let innovation = z - z_pred;

            let r_eff = if let Some(p_zz_inv) = (p_zz_innovation + self.r).try_inverse() {
                let mahalanobis_sq = innovation.transpose() * p_zz_inv * innovation;
                if mahalanobis_sq[(0, 0)] > self.mahalanobis_threshold {
                    self.r * 100.0
                } else {
                    self.r
                }
            } else {
                self.r
            };

            let p_zz_eff = p_zz_innovation + r_eff;
            if let Some(p_zz_eff_inv) = p_zz_eff.try_inverse() {
                let k = p_xz * p_zz_eff_inv;
                self.x += k * innovation;
                self.p -= k * p_zz_eff * k.transpose();
            }
        }
    }
}

struct InteractingMultipleModelFilter {
    x: Vector6<f64>,
    p: Matrix6<f64>,
    models: Vec<UnscentedKalmanFilter>,
    model_probabilities: Vec<f64>,
    transition_matrix: SMatrix<f64, 2, 2>,
}
impl InteractingMultipleModelFilter {
    pub fn new(initial_x: Vector6<f64>, initial_p: Matrix6<f64>, model_definitions: Vec<(Matrix6<f64>, Matrix3<f64>)>, transition_matrix: SMatrix<f64, 2, 2>, dt: f64) -> Self {
        let num_models = model_definitions.len();
        let models = model_definitions
            .into_iter()
            .map(|(q, r)| UnscentedKalmanFilter::new(initial_x, initial_p, q, r, dt))
            .collect();
        Self { x: initial_x, p: initial_p, models, model_probabilities: vec![1.0 / num_models as f64; num_models], transition_matrix }
    }
    pub fn predict(&mut self) {
        let num_models = self.models.len();
        let mut mixed_states = vec![Vector6::zeros(); num_models];
        let mut mixed_covariances = vec![Matrix6::zeros(); num_models];
        let mut mixing_probabilities = SMatrix::<f64, 2, 2>::zeros();
        let mut c_bar = vec![0.0; num_models];

        for j in 0..num_models {
            c_bar[j] = (0..num_models)
                .map(|i| self.transition_matrix[(i, j)] * self.model_probabilities[i])
                .sum();
        }
        for j in 0..num_models {
            for i in 0..num_models {
                if c_bar[j] > 1e-9 {
                    mixing_probabilities[(i, j)] = self.transition_matrix[(i, j)] * self.model_probabilities[i] / c_bar[j];
                }
            }
        }
        
        for j in 0..num_models {
            let mut mixed_x = Vector6::zeros();
            for i in 0..num_models {
                mixed_x += self.models[i].x * mixing_probabilities[(i, j)];
            }
            mixed_states[j] = mixed_x;

            let mut mixed_p = Matrix6::zeros();
            for i in 0..num_models {
                let diff = self.models[i].x - mixed_states[j];
                mixed_p += mixing_probabilities[(i, j)] * (self.models[i].p + diff * diff.transpose());
            }
            mixed_covariances[j] = mixed_p;
        }

        for j in 0..num_models {
            self.models[j].x = mixed_states[j];
            self.models[j].p = mixed_covariances[j];
            self.models[j].predict();
        }
    }
    pub fn update(&mut self, z: Vector3<f64>) {
        let num_models = self.models.len();
        let mut likelihoods = vec![0.0; num_models];
        let mut c_bar = vec![0.0; num_models];
        for j in 0..num_models {
            c_bar[j] = (0..num_models)
                .map(|i| self.transition_matrix[(i, j)] * self.model_probabilities[i])
                .sum();
        }

        for j in 0..num_models {
            self.models[j].update(z);
            let s = self.models[j].last_innovation_covariance;
            let y = self.models[j].last_innovation;
            if let Some(s_inv) = s.try_inverse() {
                let det_s = s.determinant();
                if det_s.abs() > 1e-9 {
                    let norm_factor = (2.0 * PI).powf(-(OBS_DIM as f64) / 2.0) * det_s.abs().powf(-0.5);
                    let exp_factor = (-0.5 * (y.transpose() * s_inv * y)[(0, 0)]).exp();
                    likelihoods[j] = norm_factor * exp_factor;
                }
            }
        }

        let mut total_likelihood = 0.0;
        for j in 0..num_models {
            self.model_probabilities[j] = likelihoods[j] * c_bar[j];
            total_likelihood += self.model_probabilities[j];
        }
        if total_likelihood > 1e-9 {
            for prob in self.model_probabilities.iter_mut() {
                *prob /= total_likelihood;
            }
        }

        let mut combined_x = Vector6::zeros();
        for j in 0..num_models {
            combined_x += self.models[j].x * self.model_probabilities[j];
        }
        self.x = combined_x;
        
        let mut combined_p = Matrix6::zeros();
        for j in 0..num_models {
            let diff = self.models[j].x - self.x;
            combined_p += self.model_probabilities[j] * (self.models[j].p + diff * diff.transpose());
        }
        self.p = combined_p;
    }
}

trait KalmanLike {
    fn predict(&mut self);
    fn update(&mut self, z: Vector3<f64>);
    fn get_state(&self) -> &Vector6<f64>;
    fn get_name(&self) -> &str;
}

impl KalmanLike for ExtendedKalmanFilter {
    fn predict(&mut self) { self.predict() }
    fn update(&mut self, z: Vector3<f64>) { self.update(z) }
    fn get_state(&self) -> &Vector6<f64> { &self.x }
    fn get_name(&self) -> &str { "EKF" }
}
impl KalmanLike for UnscentedKalmanFilter {
    fn predict(&mut self) { self.predict() }
    fn update(&mut self, z: Vector3<f64>) { self.update(z) }
    fn get_state(&self) -> &Vector6<f64> { &self.x }
    fn get_name(&self) -> &str { "UKF" }
}
impl KalmanLike for CubatureKalmanFilter {
    fn predict(&mut self) { self.predict() }
    fn update(&mut self, z: Vector3<f64>) { self.update(z) }
    fn get_state(&self) -> &Vector6<f64> { &self.x }
    fn get_name(&self) -> &str { "CKF" }
}
impl KalmanLike for RobustCubatureKalmanFilter {
    fn predict(&mut self) { self.predict() }
    fn update(&mut self, z: Vector3<f64>) { self.update(z) }
    fn get_state(&self) -> &Vector6<f64> { &self.x }
    fn get_name(&self) -> &str { "RCKF" }
}
impl KalmanLike for InteractingMultipleModelFilter {
    fn predict(&mut self) { self.predict() }
    fn update(&mut self, z: Vector3<f64>) { self.update(z) }
    fn get_state(&self) -> &Vector6<f64> { &self.x }
    fn get_name(&self) -> &str { "IMM-UKF" }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- Simulation Setup ---
    let dt = 0.1;
    let num_steps = 2000; // ステップ数を2000に変更
    let output_dir = "frames";
    let video_filename = "nonlinear_comparison_large.mp4";
    create_dir_all(output_dir)?;

    // --- True Path Parameters (Helix) ---
    let radius = 50.0;
    let omega = 0.05;
    let vz = 4.0; 

    // --- Initial State ---
    let initial_true_state = Vector6::new(radius, 0.0, 0.0, 0.0, radius * omega, vz);
    let initial_x = initial_true_state.clone(); 
    let initial_p = Matrix6::from_diagonal(&Vector6::new(10.0, 10.0, 10.0, 1.0, 1.0, 1.0));

    // --- Noise Models ---
    let r_mat = Matrix3::from_diagonal(&Vector3::new(1.0, 0.01, 0.01));
    let q_cv = Matrix6::from_diagonal(&Vector6::new(0.01, 0.01, 0.01, 0.1, 0.1, 0.1));
    let q_maneuver = Matrix6::from_diagonal(&Vector6::new(0.5, 0.5, 0.5, 2.0, 2.0, 1.0));
    
    let imm_models = vec![(q_cv, r_mat), (q_maneuver, r_mat)];
    let transition_matrix = SMatrix::<f64, 2, 2>::new(0.95, 0.05, 0.05, 0.95);

    let mut filters: Vec<Box<dyn KalmanLike>> = vec![
        Box::new(ExtendedKalmanFilter::new(initial_x, initial_p, q_cv, r_mat, dt)),
        Box::new(UnscentedKalmanFilter::new(initial_x, initial_p, q_cv, r_mat, dt)),
        Box::new(CubatureKalmanFilter::new(initial_x, initial_p, q_cv, r_mat, dt)),
        Box::new(RobustCubatureKalmanFilter::new(initial_x, initial_p, q_cv, r_mat, dt)),
        Box::new(InteractingMultipleModelFilter::new(initial_x, initial_p, imm_models, transition_matrix, dt)),
    ];
    
    let mut true_history: Vec<Vector6<f64>> = Vec::with_capacity(num_steps);
    let mut obs_history_raw: Vec<Vector3<f64>> = Vec::with_capacity(num_steps);
    let mut estimates_history: Vec<Vec<Vector6<f64>>> = vec![Vec::with_capacity(num_steps); filters.len()];
    let mut outlier_steps = Vec::new();
    let mut rng = rand::thread_rng();
    
    println!("Rendering {} frames...", num_steps);
    println!("This will take a while...");

    for i in 0..num_steps {
        let time = i as f64 * dt;
        
        let px = radius * (omega * time).cos();
        let py = radius * (omega * time).sin();
        let pz = vz * time;
        let vx = -radius * omega * (omega * time).sin();
        let vy = radius * omega * (omega * time).cos();
        let current_true_state = Vector6::new(px, py, pz, vx, vy, vz);
        true_history.push(current_true_state);

        let mut obs_noise = Vector3::new(
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-0.02..0.02),
            rng.gen_range(-0.02..0.02),
        );
        if i == num_steps / 4 || i == num_steps * 3 / 4 {
            obs_noise += Vector3::new(30.0, 0.5, 0.5);
            outlier_steps.push(i);
        }
        let observation = observation_function(&current_true_state) + obs_noise;
        obs_history_raw.push(observation);

        for (j, filter) in filters.iter_mut().enumerate() {
            filter.predict();
            filter.update(observation);
            estimates_history[j].push(filter.get_state().clone());
        }

        // --- Plotting Current Frame ---
        let frame_path = format!("{}/frame_{:04}.png", output_dir, i); // 2000ステップなので%04に変更
        let root = BitMapBackend::new(&frame_path, (1920, 1080)).into_drawing_area(); // 解像度を1920x1080に変更
        root.fill(&WHITE)?;
        
        // Z軸の描画範囲をシミュレーションの長さに合わせて調整
        let z_max = vz * (num_steps as f64 * dt) + 50.0;
        let (x_range, y_range, z_range) = ((-60.0..60.0), (-60.0..60.0), (0.0..z_max));
        
        let mut chart = ChartBuilder::on(&root)
            .caption(format!("Non-Linear Kalman Filters - Step {}", i + 1), ("sans-serif", 50).into_font())
            .margin(20)
            .build_cartesian_3d(x_range.clone(), z_range.clone(), y_range.clone())?;
        
        chart.with_projection(|mut pb| {
            pb.pitch = 0.6;
            pb.yaw = 1.9 + (i as f64 / num_steps as f64) * PI / 2.0;
            pb.scale = 0.7;
            pb.into_matrix()
        });

        chart.configure_axes()
            .label_style(("sans-serif", 20).into_font())
            .draw()?;
        
        let current_true_path = true_history.iter().map(|s| (s[0], s[2], s[1]));
        
        let current_obs_path = obs_history_raw.iter().map(|z| {
            let (r, az, el) = (z[0], z[1], z[2]);
            let x = r * el.cos() * az.cos();
            let y = r * el.cos() * az.sin();
            let z_coord = r * el.sin();
            (x, z_coord, y)
        });

        chart.draw_series(PointSeries::of_element(current_obs_path, 2, &GREY.mix(0.5), &|c, s, st| {
            EmptyElement::at(c) + Circle::new((0, 0), s, st.filled())
        }))?;
        
        if i > 0 {
            chart.draw_series(LineSeries::new(current_true_path, BLACK.stroke_width(4)))?;
            let colors = [ORANGE, GREEN, RED, PURPLE, BLUE];
            for (j, filter) in filters.iter().enumerate() {
                chart.draw_series(LineSeries::new(
                    estimates_history[j].iter().map(|s| (s[0], s[2], s[1])),
                    colors[j].stroke_width(2),
                ))?.label(filter.get_name())
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], colors[j].filled()));
            }
        }
        
        chart.configure_series_labels().border_style(BLACK).background_style(WHITE.mix(0.8)).draw()?;
        root.present()?;
        
        print!("\rRendering frame {}/{}...", i + 1, num_steps);
        use std::io::{stdout, Write};
        stdout().flush()?;
    }
    
    println!("\nAnimation frames rendered to '{}' directory.", output_dir);
    
    // --- Video Conversion using ffmpeg ---
    println!("Attempting to convert frames to MP4 using ffmpeg...");
    let ffmpeg_input = format!("{}/frame_%04d.png", output_dir); // %04に変更
    let output = Command::new("ffmpeg")
        .arg("-framerate").arg("60") // フレームレートを60に変更
        .arg("-i").arg(&ffmpeg_input)
        .arg("-c:v").arg("libx264")
        .arg("-pix_fmt").arg("yuv420p")
        .arg("-y").arg(video_filename)
        .output()?;

    if output.status.success() {
        println!("Successfully created video: {}", video_filename);
    } else {
        println!("ffmpeg command failed.");
        eprintln!("ffmpeg stderr: {}", String::from_utf8_lossy(&output.stderr));
        println!("You can try running the command manually:");
        println!("ffmpeg -framerate 60 -i {} -c:v libx264 -pix_fmt yuv420p {}", ffmpeg_input, video_filename);
    }

    Ok(())
}