// main.rs

use nalgebra::{Cholesky, Matrix2, Matrix4, SMatrix, Vector2, Vector4};
use plotters::prelude::full_palette::GREY;
use plotters::prelude::full_palette::ORANGE;
use plotters::prelude::full_palette::PURPLE;
use plotters::prelude::*;
use plotters::style::Color;
use rand::prelude::*;

// --------------------------------------------------------------------------------
// --- 共通の関数とモデル (変更なし) ---
// --------------------------------------------------------------------------------
fn state_transition_function(x_prev: &Vector4<f64>, dt: f64) -> Vector4<f64> {
    let x = x_prev[0];
    let y = x_prev[1];
    let v = x_prev[2];
    let theta = x_prev[3];
    Vector4::new(x + v * theta.cos() * dt, y + v * theta.sin() * dt, v, theta)
}

fn observation_function(x_curr: &Vector4<f64>) -> Vector2<f64> {
    Vector2::new(x_curr[0], x_curr[1])
}

// --------------------------------------------------------------------------------
// --- 各フィルタの実装 (変更なし) ---
// --------------------------------------------------------------------------------
// (ここに、以前のKF, EKF, UKF, CKF, RCKFの`struct`と`impl`をそのまま貼り付けます)
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
        Self {
            x: initial_x,
            p: initial_p,
            q,
            r,
            dt,
        }
    }
    fn calculate_state_transition_matrix(&self, x_prev: &Vector4<f64>) -> Matrix4<f64> {
        let v = x_prev[2];
        let theta = x_prev[3];
        let dt = self.dt;
        Matrix4::new(
            1.0,
            0.0,
            dt * theta.cos(),
            -v * dt * theta.sin(),
            0.0,
            1.0,
            dt * theta.sin(),
            v * dt * theta.cos(),
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
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
        let s_inv = s.try_inverse().unwrap();
        let k = self.p * h.transpose() * s_inv;
        self.x += k * y;
        self.p = (Matrix4::identity() - k * h) * self.p;
    }
    pub fn get_state(&self) -> &Vector4<f64> {
        &self.x
    }
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
        Self {
            x: initial_x,
            p: initial_p,
            q,
            r,
            dt,
        }
    }
    fn calculate_jacobian_f(&self, x_prev: &Vector4<f64>) -> Matrix4<f64> {
        let v = x_prev[2];
        let theta = x_prev[3];
        let dt = self.dt;
        Matrix4::new(
            1.,
            0.,
            dt * theta.cos(),
            -v * dt * theta.sin(),
            0.,
            1.,
            dt * theta.sin(),
            v * dt * theta.cos(),
            0.,
            0.,
            1.,
            0.,
            0.,
            0.,
            0.,
            1.,
        )
    }
    pub fn predict(&mut self) {
        let f_jacobian = self.calculate_jacobian_f(&self.x);
        self.x = state_transition_function(&self.x, self.dt);
        self.p = f_jacobian * self.p * f_jacobian.transpose() + self.q;
    }
    pub fn update(&mut self, z: Vector2<f64>) {
        let h_jacobian = SMatrix::<f64, 2, 4>::new(1., 0., 0., 0., 0., 1., 0., 0.);
        let y = z - observation_function(&self.x);
        let s_inv = (h_jacobian * self.p * h_jacobian.transpose() + self.r)
            .try_inverse()
            .unwrap();
        let k = self.p * h_jacobian.transpose() * s_inv;
        self.x += k * y;
        self.p = (Matrix4::identity() - k * h_jacobian) * self.p;
    }
    pub fn get_state(&self) -> &Vector4<f64> {
        &self.x
    }
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
}
impl UnscentedKalmanFilter {
    pub fn new(initial_x: Vector4<f64>, initial_p: Matrix4<f64>, dt: f64) -> Self {
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
        let q = Matrix4::from_diagonal(&Vector4::new(0.01, 0.01, 0.001, 0.001));
        let r = Matrix2::from_diagonal(&Vector2::new(0.1, 0.1));
        Self {
            x: initial_x,
            p: initial_p,
            q,
            r,
            dt,
            weights_m,
            weights_c,
            lambda,
            n,
        }
    }
    fn generate_sigma_points(&self) -> SMatrix<f64, 4, 9> {
        let mut sigma_points = SMatrix::<f64, 4, 9>::zeros();
        let p_sqrt = Cholesky::new((self.n as f64 + self.lambda) * self.p)
            .unwrap()
            .l();
        sigma_points.set_column(0, &self.x);
        for i in 0..self.n {
            sigma_points.set_column(i + 1, &(self.x + p_sqrt.column(i)));
            sigma_points.set_column(i + 1 + self.n, &(self.x - p_sqrt.column(i)));
        }
        sigma_points
    }
    pub fn predict(&mut self) {
        let sigma_points = self.generate_sigma_points();
        let mut predicted_sigma_points = SMatrix::<f64, 4, 9>::zeros();
        for i in 0..(2 * self.n + 1) {
            predicted_sigma_points.set_column(
                i,
                &state_transition_function(&sigma_points.column(i).into(), self.dt),
            );
        }
        self.x = predicted_sigma_points * self.weights_m.transpose();
        let mut p_pred = Matrix4::zeros();
        for i in 0..(2 * self.n + 1) {
            let diff = predicted_sigma_points.column(i) - self.x;
            p_pred += self.weights_c[(0, i)] * (diff * diff.transpose());
        }
        self.p = p_pred + self.q;
    }
    pub fn update(&mut self, z: Vector2<f64>) {
        let sigma_points = self.generate_sigma_points();
        let mut observed_sigma_points = SMatrix::<f64, 2, 9>::zeros();
        for i in 0..(2 * self.n + 1) {
            observed_sigma_points
                .set_column(i, &observation_function(&sigma_points.column(i).into()));
        }
        let z_pred = observed_sigma_points * self.weights_m.transpose();
        let mut s = Matrix2::zeros();
        let mut t = SMatrix::<f64, 4, 2>::zeros();
        for i in 0..(2 * self.n + 1) {
            let z_diff = observed_sigma_points.column(i) - z_pred;
            let x_diff = sigma_points.column(i) - self.x;
            s += self.weights_c[(0, i)] * (z_diff * z_diff.transpose());
            t += self.weights_c[(0, i)] * (x_diff * z_diff.transpose());
        }
        s += self.r;
        let k = t * s.try_inverse().unwrap();
        self.x += k * (z - z_pred);
        self.p -= k * s * k.transpose();
    }
    pub fn get_state(&self) -> &Vector4<f64> {
        &self.x
    }
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
        Self {
            x: initial_x,
            p: initial_p,
            q,
            r,
            dt,
            n: 4,
        }
    }
    fn generate_cubature_points(&self, x: &Vector4<f64>, p: &Matrix4<f64>) -> SMatrix<f64, 4, 8> {
        let mut points = SMatrix::<f64, 4, 8>::zeros();
        let s = Cholesky::new(*p).unwrap().l();
        let factor = (self.n as f64).sqrt();
        for i in 0..self.n {
            let s_col = s.column(i);
            points.set_column(i, &(x + factor * s_col));
            points.set_column(i + self.n, &(x - factor * s_col));
        }
        points
    }
    pub fn predict(&mut self) {
        let points = self.generate_cubature_points(&self.x, &self.p);
        let mut propagated_points = SMatrix::<f64, 4, 8>::zeros();
        for i in 0..(2 * self.n) {
            propagated_points.set_column(
                i,
                &state_transition_function(&points.column(i).into(), self.dt),
            );
        }
        self.x = propagated_points.column_mean();
        let mut p_pred = Matrix4::zeros();
        for i in 0..(2 * self.n) {
            let diff = propagated_points.column(i) - self.x;
            p_pred += diff * diff.transpose();
        }
        self.p = p_pred / (2.0 * self.n as f64) + self.q;
    }
    pub fn update(&mut self, z: Vector2<f64>) {
        let points = self.generate_cubature_points(&self.x, &self.p);
        let mut observed_points = SMatrix::<f64, 2, 8>::zeros();
        for i in 0..(2 * self.n) {
            observed_points.set_column(i, &observation_function(&points.column(i).into()));
        }
        let z_pred: Vector2<f64> = observed_points.column_mean();
        let mut p_zz = Matrix2::zeros();
        let mut p_xz = SMatrix::<f64, 4, 2>::zeros();
        for i in 0..(2 * self.n) {
            let z_diff = observed_points.column(i) - z_pred;
            let x_diff = points.column(i) - self.x;
            p_zz += z_diff * z_diff.transpose();
            p_xz += x_diff * z_diff.transpose();
        }
        let weight = 1.0 / (2.0 * self.n as f64);
        let p_zz = weight * p_zz + self.r;
        let p_xz = weight * p_xz;
        let k = p_xz * p_zz.try_inverse().unwrap();
        self.x += k * (z - z_pred);
        self.p -= k * p_zz * k.transpose();
    }
    pub fn get_state(&self) -> &Vector4<f64> {
        &self.x
    }
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
        Self {
            x: initial_x,
            p: initial_p,
            q,
            r,
            dt,
            n: 4,
            mahalanobis_threshold: 5.991,
        }
    }
    fn generate_cubature_points(&self, x: &Vector4<f64>, p: &Matrix4<f64>) -> SMatrix<f64, 4, 8> {
        let mut points = SMatrix::<f64, 4, 8>::zeros();
        let s = Cholesky::new(*p).unwrap().l();
        let factor = (self.n as f64).sqrt();
        for i in 0..self.n {
            points.set_column(i, &(x + factor * s.column(i)));
            points.set_column(i + self.n, &(x - factor * s.column(i)));
        }
        points
    }
    pub fn predict(&mut self) {
        let points = self.generate_cubature_points(&self.x, &self.p);
        let mut propagated_points = SMatrix::<f64, 4, 8>::zeros();
        for i in 0..(2 * self.n) {
            propagated_points.set_column(
                i,
                &state_transition_function(&points.column(i).into(), self.dt),
            );
        }
        self.x = propagated_points.column_mean();
        let mut p_pred = Matrix4::zeros();
        for i in 0..(2 * self.n) {
            let diff = propagated_points.column(i) - self.x;
            p_pred += diff * diff.transpose();
        }
        self.p = p_pred / (2.0 * self.n as f64) + self.q;
    }
    pub fn update(&mut self, z: Vector2<f64>) {
        let points = self.generate_cubature_points(&self.x, &self.p);
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
        let p_zz = p_zz_innovation + self.r;
        let mahalanobis_sq = innovation.transpose() * p_zz.try_inverse().unwrap() * innovation;
        let r_eff = if mahalanobis_sq[(0, 0)] > self.mahalanobis_threshold {
            self.r * 100.0
        } else {
            self.r
        };
        let p_zz_eff = p_zz_innovation + r_eff;
        let k = p_xz * p_zz_eff.try_inverse().unwrap();
        self.x += k * innovation;
        self.p -= k * p_zz_eff * k.transpose();
    }
    pub fn get_state(&self) -> &Vector4<f64> {
        &self.x
    }
}

// --------------------------------------------------------------------------------
// --- メインのシミュレーションと描画 ---
// --------------------------------------------------------------------------------
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dt = 0.1;
    let initial_x = Vector4::new(0.0, 0.0, 1.0, 0.0);
    let initial_p = Matrix4::from_diagonal(&Vector4::new(100.0, 100.0, 10.0, 10.0));

    // --- 各フィルタのインスタンス化 ---
    let mut filters: Vec<(&str, Box<dyn KalmanLike>)> = vec![
        ("KF", Box::new(KalmanFilter::new(initial_x, initial_p, dt))),
        (
            "EKF",
            Box::new(ExtendedKalmanFilter::new(initial_x, initial_p, dt)),
        ),
        (
            "UKF",
            Box::new(UnscentedKalmanFilter::new(initial_x, initial_p, dt)),
        ),
        (
            "CKF",
            Box::new(CubatureKalmanFilter::new(initial_x, initial_p, dt)),
        ),
        (
            "RCKF",
            Box::new(RobustCubatureKalmanFilter::new(initial_x, initial_p, dt)),
        ),
    ];

    // --- シミュレーション履歴の保存用 ---
    let mut true_history = Vec::new();
    let mut obs_history = Vec::new();
    let mut estimates_history: Vec<Vec<(f64, f64)>> = vec![Vec::new(); filters.len()];
    let mut outlier_indices = Vec::new();

    let mut rng = rand::rng();

    // --- シミュレーションループ ---
    for i in 0..100 {
        let true_v = 1.0;
        let true_angular_velocity = 0.05;
        let true_theta = true_angular_velocity * (i as f64 + 1.0) * dt;
        let radius = true_v / true_angular_velocity;
        let true_x_curr = initial_x[0] + radius * true_theta.sin();
        let true_y_curr = initial_x[1] + radius * (1.0 - true_theta.cos());
        let true_state = Vector4::new(true_x_curr, true_y_curr, true_v, true_theta);
        true_history.push((true_state[0], true_state[1]));

        let mut observation_noise_x = (rng.random::<f64>() - 0.5) * 0.5;
        let mut observation_noise_y = (rng.random::<f64>() - 0.5) * 0.5;

        if i == 30 || i == 60 {
            observation_noise_x += 10.0;
            observation_noise_y -= 10.0;
            outlier_indices.push(obs_history.len());
        }

        let observation = observation_function(&true_state)
            + Vector2::new(observation_noise_x, observation_noise_y);
        obs_history.push((observation[0], observation[1]));

        for (idx, (_, f)) in filters.iter_mut().enumerate() {
            f.predict();
            f.update(observation);
            let state = f.get_state();
            estimates_history[idx].push((state[0], state[1]));
        }
    }

    // --- グラフ描画 ---
    let root_area =
        SVGBackend::new("rust_plotters_comparison.svg", (1024, 768)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root_area)
        .caption("Kalman Filter Comparison", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-5f64..25f64, -5f64..25f64)?;

    chart.configure_mesh().draw()?;

    // 観測点
    chart
        .draw_series(
            obs_history
                .iter()
                .map(|(x, y)| Cross::new((*x, *y), 5, &GREY)),
        )?
        .label("Observations")
        .legend(|(x, y)| Cross::new((x, y), 5, &GREY));

    // 外れ値
    chart
        .draw_series(
            outlier_indices
                .iter()
                .map(|&i| Circle::new(obs_history[i], 8, RED.stroke_width(2))),
        )?
        .label("Outlier")
        .legend(|(x, y)| Circle::new((x, y), 8, RED.stroke_width(2)));

    // 真の軌跡
    chart
        .draw_series(LineSeries::new(true_history, &BLACK))?
        .label("True Path")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));

    // 各フィルタの推定軌跡
    let colors = [ORANGE, GREEN, RED, PURPLE, BLUE];
    for (i, (name, _)) in filters.iter().enumerate() {
        chart
            .draw_series(LineSeries::new(estimates_history[i].clone(), &colors[i]))?
            .label(*name)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &colors[i]));
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root_area.present()?;
    println!("Graph saved to rust_plotters_comparison.svg");

    Ok(())
}

// --- フィルタを動的に扱うためのトレイト ---
trait KalmanLike {
    fn predict(&mut self);
    fn update(&mut self, z: Vector2<f64>);
    fn get_state(&self) -> &Vector4<f64>;
}

impl KalmanLike for KalmanFilter {
    fn predict(&mut self) {
        self.predict()
    }
    fn update(&mut self, z: Vector2<f64>) {
        self.update(z)
    }
    fn get_state(&self) -> &Vector4<f64> {
        self.get_state()
    }
}
impl KalmanLike for ExtendedKalmanFilter {
    fn predict(&mut self) {
        self.predict()
    }
    fn update(&mut self, z: Vector2<f64>) {
        self.update(z)
    }
    fn get_state(&self) -> &Vector4<f64> {
        self.get_state()
    }
}
impl KalmanLike for UnscentedKalmanFilter {
    fn predict(&mut self) {
        self.predict()
    }
    fn update(&mut self, z: Vector2<f64>) {
        self.update(z)
    }
    fn get_state(&self) -> &Vector4<f64> {
        self.get_state()
    }
}
impl KalmanLike for CubatureKalmanFilter {
    fn predict(&mut self) {
        self.predict()
    }
    fn update(&mut self, z: Vector2<f64>) {
        self.update(z)
    }
    fn get_state(&self) -> &Vector4<f64> {
        self.get_state()
    }
}
impl KalmanLike for RobustCubatureKalmanFilter {
    fn predict(&mut self) {
        self.predict()
    }
    fn update(&mut self, z: Vector2<f64>) {
        self.update(z)
    }
    fn get_state(&self) -> &Vector4<f64> {
        self.get_state()
    }
}
