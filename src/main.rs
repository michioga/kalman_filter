// main.rs

use nalgebra::{Cholesky, Matrix2, Matrix4, SMatrix, Vector2, Vector4};
use plotters::prelude::full_palette::CYAN;
use plotters::prelude::full_palette::GREY;
use plotters::prelude::full_palette::ORANGE;
use plotters::prelude::full_palette::PURPLE;
use plotters::prelude::*;
use plotters::style::Color;
use rand::prelude::*;
use std::f64::consts::PI;

// --------------------------------------------------------------------------------
// --- 共通の関数とモデル ---
// --------------------------------------------------------------------------------
fn state_transition_function(x_prev: &Vector4<f64>, dt: f64) -> Vector4<f64> {
    let x = x_prev[0];
    let y = x_prev[1];
    let v = x_prev[2];
    let theta = x_prev[3];
    Vector4::new(
        x + v * theta.cos() * dt,
        y + v * theta.sin() * dt,
        v,
        theta,
    )
}

fn observation_function(x_curr: &Vector4<f64>) -> Vector2<f64> {
    Vector2::new(x_curr[0], x_curr[1])
}

// --------------------------------------------------------------------------------
// --- 各フィルタの実装 ---
// --------------------------------------------------------------------------------
struct KalmanFilter {
    x: Vector4<f64>,
    p: Matrix4<f64>,
    q: Matrix4<f64>,
    r: Matrix2<f64>,
    dt: f64,
}
impl KalmanFilter {
    pub fn new(initial_x: Vector4<f64>, initial_p: Matrix4<f64>, dt: f64) -> Self {
        let q = Matrix4::from_diagonal(&Vector4::new(0.01, 0.01, 0.001, 0.001));
        let r = Matrix2::from_diagonal(&Vector2::new(0.1, 0.1));
        Self { x: initial_x, p: initial_p, q, r, dt }
    }
    fn calculate_state_transition_matrix(&self, x_prev: &Vector4<f64>) -> Matrix4<f64> {
        let v = x_prev[2];
        let theta = x_prev[3];
        let dt = self.dt;
        Matrix4::new(1.0, 0.0, dt * theta.cos(), -v * dt * theta.sin(), 0.0, 1.0, dt * theta.sin(), v * dt * theta.cos(), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    }
    fn get_observation_matrix(&self) -> SMatrix<f64, 2, 4> {
        SMatrix::<f64, 2, 4>::new(1., 0., 0., 0., 0., 1., 0., 0.)
    }
    pub fn predict(&mut self) {
        let f = self.calculate_state_transition_matrix(&self.x);
        self.x = f * self.x;
        self.p = f * self.p * f.transpose() + self.q;
    }
    pub fn update(&mut self, z: Vector2<f64>) {
        let h = self.get_observation_matrix();
        let y = z - h * self.x;
        let s = h * self.p * h.transpose() + self.r;
        if let Some(s_inv) = s.try_inverse() {
            let k = self.p * h.transpose() * s_inv;
            self.x += k * y;
            self.p = (Matrix4::identity() - k * h) * self.p;
        }
    }
    pub fn get_state(&self) -> &Vector4<f64> { &self.x }
}

struct ExtendedKalmanFilter {
    x: Vector4<f64>,
    p: Matrix4<f64>,
    q: Matrix4<f64>,
    r: Matrix2<f64>,
    dt: f64,
}
impl ExtendedKalmanFilter {
    pub fn new(initial_x: Vector4<f64>, initial_p: Matrix4<f64>, dt: f64) -> Self {
        let q = Matrix4::from_diagonal(&Vector4::new(0.01, 0.01, 0.001, 0.001));
        let r = Matrix2::from_diagonal(&Vector2::new(0.1, 0.1));
        Self { x: initial_x, p: initial_p, q, r, dt }
    }
    fn calculate_jacobian_f(&self, x_prev: &Vector4<f64>) -> Matrix4<f64> {
        let v = x_prev[2];
        let theta = x_prev[3];
        let dt = self.dt;
        Matrix4::new(1., 0., dt * theta.cos(), -v * dt * theta.sin(), 0., 1., dt * theta.sin(), v * dt * theta.cos(), 0., 0., 1., 0., 0., 0., 0., 1.)
    }
    pub fn predict(&mut self) {
        let f_jacobian = self.calculate_jacobian_f(&self.x);
        self.x = state_transition_function(&self.x, self.dt);
        self.p = f_jacobian * self.p * f_jacobian.transpose() + self.q;
    }
    pub fn update(&mut self, z: Vector2<f64>) {
        let h_jacobian = SMatrix::<f64, 2, 4>::new(1., 0., 0., 0., 0., 1., 0., 0.);
        let y = z - observation_function(&self.x);
        let s = h_jacobian * self.p * h_jacobian.transpose() + self.r;
        if let Some(s_inv) = s.try_inverse() {
            let k = self.p * h_jacobian.transpose() * s_inv;
            self.x += k * y;
            self.p = (Matrix4::identity() - k * h_jacobian) * self.p;
        }
    }
    pub fn get_state(&self) -> &Vector4<f64> { &self.x }
}

struct UnscentedKalmanFilter {
    x: Vector4<f64>,
    p: Matrix4<f64>,
    q: Matrix4<f64>,
    r: Matrix2<f64>,
    dt: f64,
    weights_m: SMatrix<f64, 1, 9>,
    weights_c: SMatrix<f64, 1, 9>,
    lambda: f64,
    n: usize,
    last_innovation: Vector2<f64>,
    last_innovation_covariance: Matrix2<f64>,
}
impl UnscentedKalmanFilter {
    pub fn new(initial_x: Vector4<f64>, initial_p: Matrix4<f64>, q: Matrix4<f64>, dt: f64) -> Self {
        let n = 4;
        let alpha: f64 = 1e-3;
        let kappa: f64 = 0.0;
        let beta: f64 = 2.0;
        let lambda = alpha.powi(2) * (n as f64 + kappa) - n as f64;
        let mut weights_m = SMatrix::<f64, 1, 9>::zeros();
        let mut weights_c = SMatrix::<f64, 1, 9>::zeros();
        weights_m[(0, 0)] = lambda / (n as f64 + lambda);
        weights_c[(0, 0)] = lambda / (n as f64 + lambda) + (1.0 - alpha.powi(2) + beta);
        for i in 1..(2 * n + 1) {
            weights_m[(0, i)] = 0.5 / (n as f64 + lambda);
            weights_c[(0, i)] = 0.5 / (n as f64 + lambda);
        }
        let r = Matrix2::from_diagonal(&Vector2::new(0.1, 0.1));
        Self { x: initial_x, p: initial_p, q, r, dt, weights_m, weights_c, lambda, n, last_innovation: Vector2::zeros(), last_innovation_covariance: Matrix2::zeros() }
    }
    fn generate_sigma_points(&self) -> Option<SMatrix<f64, 4, 9>> {
        Cholesky::new((self.n as f64 + self.lambda) * self.p).map(|chol| {
            let p_sqrt = chol.l();
            let mut sigma_points = SMatrix::<f64, 4, 9>::zeros();
            sigma_points.set_column(0, &self.x);
            for i in 0..self.n {
                sigma_points.set_column(i + 1, &(self.x + p_sqrt.column(i)));
                sigma_points.set_column(i + 1 + self.n, &(self.x - p_sqrt.column(i)));
            }
            sigma_points
        })
    }
    pub fn predict(&mut self) {
        if let Some(sigma_points) = self.generate_sigma_points() {
            let mut predicted_sigma_points = SMatrix::<f64, 4, 9>::zeros();
            for i in 0..(2 * self.n + 1) {
                predicted_sigma_points.set_column(i, &state_transition_function(&sigma_points.column(i).into(), self.dt));
            }
            self.x = predicted_sigma_points * self.weights_m.transpose();
            let mut p_pred = Matrix4::zeros();
            for i in 0..(2 * self.n + 1) {
                let diff = predicted_sigma_points.column(i) - self.x;
                p_pred += self.weights_c[(0, i)] * (diff * diff.transpose());
            }
            self.p = p_pred + self.q;
        }
    }
    pub fn update(&mut self, z: Vector2<f64>) {
        if let Some(sigma_points_pred) = self.generate_sigma_points() {
            let mut observed_sigma_points = SMatrix::<f64, 2, 9>::zeros();
            for i in 0..(2 * self.n + 1) {
                observed_sigma_points.set_column(i, &observation_function(&sigma_points_pred.column(i).into()));
            }
            let z_pred = observed_sigma_points * self.weights_m.transpose();
            let mut s = Matrix2::zeros();
            let mut t = SMatrix::<f64, 4, 2>::zeros();
            for i in 0..(2 * self.n + 1) {
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
    pub fn get_state(&self) -> &Vector4<f64> { &self.x }
}

struct CubatureKalmanFilter {
    x: Vector4<f64>,
    p: Matrix4<f64>,
    q: Matrix4<f64>,
    r: Matrix2<f64>,
    dt: f64,
    n: usize,
}
impl CubatureKalmanFilter {
    pub fn new(initial_x: Vector4<f64>, initial_p: Matrix4<f64>, dt: f64) -> Self {
        let q = Matrix4::from_diagonal(&Vector4::new(0.01, 0.01, 0.001, 0.001));
        let r = Matrix2::from_diagonal(&Vector2::new(0.1, 0.1));
        Self { x: initial_x, p: initial_p, q, r, dt, n: 4 }
    }
    fn generate_cubature_points(&self, x: &Vector4<f64>, p: &Matrix4<f64>) -> Option<SMatrix<f64, 4, 8>> {
        Cholesky::new(*p).map(|chol| {
            let s = chol.l();
            let mut points = SMatrix::<f64, 4, 8>::zeros();
            let factor = (self.n as f64).sqrt();
            for i in 0..self.n {
                let s_col = s.column(i);
                points.set_column(i, &(x + factor * s_col));
                points.set_column(i + self.n, &(x - factor * s_col));
            }
            points
        })
    }
    pub fn predict(&mut self) {
        if let Some(points) = self.generate_cubature_points(&self.x, &self.p) {
            let mut propagated_points = SMatrix::<f64, 4, 8>::zeros();
            for i in 0..(2 * self.n) {
                propagated_points.set_column(i, &state_transition_function(&points.column(i).into(), self.dt));
            }
            self.x = propagated_points.column_mean();
            let mut p_pred = Matrix4::zeros();
            for i in 0..(2 * self.n) {
                let diff = propagated_points.column(i) - self.x;
                p_pred += diff * diff.transpose();
            }
            self.p = p_pred / (2.0 * self.n as f64) + self.q;
        }
    }
    pub fn update(&mut self, z: Vector2<f64>) {
        if let Some(points) = self.generate_cubature_points(&self.x, &self.p) {
            let mut observed_points = SMatrix::<f64, 2, 8>::zeros();
            for i in 0..(2 * self.n) {
                observed_points.set_column(i, &observation_function(&points.column(i).into()));
            }
            let z_pred: Vector2<f64> = observed_points.column_mean();
            let mut p_zz_innovation = Matrix2::zeros();
            let mut p_xz = SMatrix::<f64, 4, 2>::zeros();
            for i in 0..(2 * self.n) {
                let z_diff = observed_points.column(i) - z_pred;
                let x_diff = points.column(i) - self.x;
                p_zz_innovation += z_diff * z_diff.transpose();
                p_xz += x_diff * z_diff.transpose();
            }
            let weight = 1.0 / (2.0 * self.n as f64);
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
    pub fn get_state(&self) -> &Vector4<f64> { &self.x }
}

struct RobustCubatureKalmanFilter {
    x: Vector4<f64>,
    p: Matrix4<f64>,
    q: Matrix4<f64>,
    r: Matrix2<f64>,
    dt: f64,
    n: usize,
    mahalanobis_threshold: f64,
}
impl RobustCubatureKalmanFilter {
    pub fn new(initial_x: Vector4<f64>, initial_p: Matrix4<f64>, dt: f64) -> Self {
        let q = Matrix4::from_diagonal(&Vector4::new(0.01, 0.01, 0.001, 0.001));
        let r = Matrix2::from_diagonal(&Vector2::new(0.1, 0.1));
        Self { x: initial_x, p: initial_p, q, r, dt, n: 4, mahalanobis_threshold: 5.991, }
    }
    fn generate_cubature_points(&self, x: &Vector4<f64>, p: &Matrix4<f64>) -> Option<SMatrix<f64, 4, 8>> {
        Cholesky::new(*p).map(|chol| {
            let s = chol.l();
            let mut points = SMatrix::<f64, 4, 8>::zeros();
            let factor = (self.n as f64).sqrt();
            for i in 0..self.n {
                points.set_column(i, &(x + factor * s.column(i)));
                points.set_column(i + self.n, &(x - factor * s.column(i)));
            }
            points
        })
    }
    pub fn predict(&mut self) {
        if let Some(points) = self.generate_cubature_points(&self.x, &self.p) {
            let mut propagated_points = SMatrix::<f64, 4, 8>::zeros();
            for i in 0..(2 * self.n) {
                propagated_points.set_column(i, &state_transition_function(&points.column(i).into(), self.dt));
            }
            self.x = propagated_points.column_mean();
            let mut p_pred = Matrix4::zeros();
            for i in 0..(2 * self.n) {
                let diff = propagated_points.column(i) - self.x;
                p_pred += diff * diff.transpose();
            }
            self.p = p_pred / (2.0 * self.n as f64) + self.q;
        }
    }
    pub fn update(&mut self, z: Vector2<f64>) {
        if let Some(points) = self.generate_cubature_points(&self.x, &self.p) {
            let mut observed_points = SMatrix::<f64, 2, 8>::zeros();
            for i in 0..(2 * self.n) {
                observed_points.set_column(i, &observation_function(&points.column(i).into()));
            }
            let z_pred: Vector2<f64> = observed_points.column_mean();
            let mut p_zz_innovation = Matrix2::zeros();
            let mut p_xz = SMatrix::<f64, 4, 2>::zeros();
            let weight = 1.0 / (2.0 * self.n as f64);
            for i in 0..(2 * self.n) {
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
    pub fn get_state(&self) -> &Vector4<f64> { &self.x }
}

struct InteractingMultipleModelFilter {
    x: Vector4<f64>,
    p: Matrix4<f64>,
    models: Vec<UnscentedKalmanFilter>,
    model_probabilities: Vec<f64>,
    transition_matrix: SMatrix<f64, 2, 2>,
}
impl InteractingMultipleModelFilter {
    pub fn new(initial_x: Vector4<f64>, initial_p: Matrix4<f64>, model_definitions: Vec<Matrix4<f64>>, transition_matrix: SMatrix<f64, 2, 2>, dt: f64) -> Self {
        let num_models = model_definitions.len();
        let models = model_definitions.into_iter().map(|q| UnscentedKalmanFilter::new(initial_x, initial_p, q, dt)).collect();
        Self { x: initial_x, p: initial_p, models, model_probabilities: vec![1.0 / num_models as f64; num_models], transition_matrix }
    }
    pub fn predict(&mut self) {
        let num_models = self.models.len();
        let mut mixed_states = vec![Vector4::zeros(); num_models];
        let mut mixed_covariances = vec![Matrix4::zeros(); num_models];
        let mut mixing_probabilities = SMatrix::<f64, 2, 2>::zeros();
        let mut c_bar = vec![0.0; num_models];
        for j in 0..num_models {
            for i in 0..num_models {
                c_bar[j] += self.transition_matrix[(i, j)] * self.model_probabilities[i];
            }
        }
        for j in 0..num_models {
            for i in 0..num_models {
                if c_bar[j] > 1e-9 {
                    mixing_probabilities[(i, j)] = self.transition_matrix[(i, j)] * self.model_probabilities[i] / c_bar[j];
                }
            }
        }
        for j in 0..num_models {
            let mut mixed_x = Vector4::zeros();
            for i in 0..num_models {
                mixed_x += self.models[i].x * mixing_probabilities[(i, j)];
            }
            mixed_states[j] = mixed_x;
            let mut mixed_p = Matrix4::zeros();
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
        self.x = state_transition_function(&self.x, self.models[0].dt);
    }
    pub fn update(&mut self, z: Vector2<f64>) {
        let num_models = self.models.len();
        let mut likelihoods = vec![0.0; num_models];
        let mut c_bar = vec![0.0; num_models];
        for j in 0..num_models {
            for i in 0..num_models {
                c_bar[j] += self.transition_matrix[(i, j)] * self.model_probabilities[i];
            }
        }
        for j in 0..num_models {
            self.models[j].update(z);
            let s = self.models[j].last_innovation_covariance;
            let y = self.models[j].last_innovation;
            if let Some(s_inv) = s.try_inverse() {
                let det_s = s.determinant();
                if det_s.abs() > 1e-9 { // Use abs() for determinant
                    let norm_factor = 1.0 / (2.0 * PI * det_s.sqrt());
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
        let mut combined_x = Vector4::zeros();
        for j in 0..num_models {
            combined_x += self.models[j].x * self.model_probabilities[j];
        }
        self.x = combined_x;
        let mut combined_p = Matrix4::zeros();
        for j in 0..num_models {
            let diff = self.models[j].x - self.x;
            combined_p += self.model_probabilities[j] * (self.models[j].p + diff * diff.transpose());
        }
        self.p = combined_p;
    }
}


// --------------------------------------------------------------------------------
// --- メインのシミュレーションと描画 ---
// --------------------------------------------------------------------------------
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dt = 0.1;
    let initial_x = Vector4::new(0.0, 0.0, 1.0, 0.0);
    let initial_p = Matrix4::from_diagonal(&Vector4::new(100.0, 100.0, 10.0, 10.0));
    let q_nc = Matrix4::from_diagonal(&Vector4::new(0.01, 0.01, 0.01, 0.001));
    let q_ct = Matrix4::from_diagonal(&Vector4::new(0.01, 0.01, 0.1, 0.1));
    let imm_models = vec![q_nc, q_ct];
    let transition_matrix = SMatrix::<f64, 2, 2>::new(0.95, 0.05, 0.05, 0.95);
    let mut filters: Vec<(&str, Box<dyn KalmanLike>)> = vec![
        ("KF", Box::new(KalmanFilter::new(initial_x, initial_p, dt))),
        ("EKF", Box::new(ExtendedKalmanFilter::new(initial_x, initial_p, dt))),
        ("UKF", Box::new(UnscentedKalmanFilter::new(initial_x, initial_p, q_nc, dt))),
        ("CKF", Box::new(CubatureKalmanFilter::new(initial_x, initial_p, dt))),
        ("RCKF", Box::new(RobustCubatureKalmanFilter::new(initial_x, initial_p, dt))),
        ("IMM-UKF", Box::new(InteractingMultipleModelFilter::new(initial_x, initial_p, imm_models, transition_matrix, dt))),
    ];
    let mut true_history: Vec<(f64, f64)> = Vec::new();
    let mut obs_history = Vec::new();
    let mut estimates_history: Vec<Vec<(f64, f64)>> = vec![Vec::new(); filters.len()];
    let mut outlier_indices = Vec::new();
    let mut rng = rand::rngs::ThreadRng::default();
    
    // --- シミュレーションループ ---
    for i in 0..100 {
        let true_v = 1.0;
        let true_angular_velocity = if i < 50 { 0.01 } else { 0.1 };
        let _true_theta = true_angular_velocity * (i as f64 + 1.0) * dt;
        let prev_true = if i > 0 {
            Vector4::new(true_history[i - 1].0, true_history[i - 1].1, true_v, true_angular_velocity * (i as f64) * dt)
        } else {
            initial_x
        };
        let true_state = state_transition_function(&prev_true, dt);
        true_history.push((true_state[0], true_state[1]));
        let mut observation_noise_x = (rng.random::<f64>() - 0.5) * 0.5;
        let mut observation_noise_y = (rng.random::<f64>() - 0.5) * 0.5;
        if i == 30 || i == 60 {
            observation_noise_x += 10.0;
            observation_noise_y -= 10.0;
            outlier_indices.push(obs_history.len());
        }
        let observation = observation_function(&true_state) + Vector2::new(observation_noise_x, observation_noise_y);
        obs_history.push((observation[0], observation[1]));

        for (idx, (_name, filter)) in filters.iter_mut().enumerate() {
            filter.predict();
            filter.update(observation);
            let state = filter.get_state();
            estimates_history[idx].push((state[0], state[1]));
        }
    }

    // --- グラフ描画 ---
    let root_area = SVGBackend::new("rust_plotters_comparison.svg", (1024, 768)).into_drawing_area();
    root_area.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root_area)
        .caption("Kalman Filter Comparison", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-5f64..15f64, -5f64..25f64)?;
    chart.configure_mesh().draw()?;
    chart.draw_series(obs_history.iter().map(|(x, y)| Cross::new((*x, *y), 5, &GREY)))?.label("Observations").legend(|(x, y)| Cross::new((x, y), 5, &GREY));
    chart.draw_series(outlier_indices.iter().map(|&i| Circle::new(obs_history[i], 8, RED.stroke_width(2))))?.label("Outlier").legend(|(x, y)| Circle::new((x, y), 8, RED.stroke_width(2)));
    chart.draw_series(LineSeries::new(true_history.clone(), &BLACK))?.label("True Path").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));
    let colors = [ORANGE, GREEN, RED, PURPLE, BLUE, CYAN];
    for (i, (name, _)) in filters.iter().enumerate() {
        chart.draw_series(LineSeries::new(estimates_history[i].clone(), &colors[i]))?.label(*name).legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &colors[i]));
    }
    chart.configure_series_labels().background_style(&WHITE.mix(0.8)).border_style(&BLACK).draw()?;
    root_area.present()?;
    println!("Graph saved to rust_plotters_comparison.svg");
    Ok(())
}

// --------------------------------------------------------------------------------
// --- フィルタを動的に扱うためのトレイト ---
// --------------------------------------------------------------------------------
trait KalmanLike {
    fn predict(&mut self);
    fn update(&mut self, z: Vector2<f64>);
    fn get_state(&self) -> &Vector4<f64>;
}

impl KalmanLike for KalmanFilter {
    fn predict(&mut self) { self.predict() }
    fn update(&mut self, z: Vector2<f64>) { self.update(z) }
    fn get_state(&self) -> &Vector4<f64> { self.get_state() }
}
impl KalmanLike for ExtendedKalmanFilter {
    fn predict(&mut self) { self.predict() }
    fn update(&mut self, z: Vector2<f64>) { self.update(z) }
    fn get_state(&self) -> &Vector4<f64> { self.get_state() }
}
impl KalmanLike for UnscentedKalmanFilter {
    fn predict(&mut self) { self.predict() }
    fn update(&mut self, z: Vector2<f64>) { self.update(z) }
    fn get_state(&self) -> &Vector4<f64> { self.get_state() }
}
impl KalmanLike for CubatureKalmanFilter {
    fn predict(&mut self) { self.predict() }
    fn update(&mut self, z: Vector2<f64>) { self.update(z) }
    fn get_state(&self) -> &Vector4<f64> { self.get_state() }
}
impl KalmanLike for RobustCubatureKalmanFilter {
    fn predict(&mut self) { self.predict() }
    fn update(&mut self, z: Vector2<f64>) { self.update(z) }
    fn get_state(&self) -> &Vector4<f64> { self.get_state() }
}
impl KalmanLike for InteractingMultipleModelFilter {
    fn predict(&mut self) { self.predict() }
    fn update(&mut self, z: Vector2<f64>) { self.update(z) }
    fn get_state(&self) -> &Vector4<f64> { &self.x }
}
