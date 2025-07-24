// main.rs

use nalgebra::{Cholesky, Matrix2, Matrix4, SMatrix, Vector2, Vector4};
use rand::Rng;
use rand::rngs::ThreadRng;

// --------------------------------------------------------------------------------
// --- 共通の関数とモデル ---
// --------------------------------------------------------------------------------

/// 状態遷移関数 f(x)
fn state_transition_function(x_prev: &Vector4<f64>, dt: f64) -> Vector4<f64> {
    let x = x_prev[0];
    let y = x_prev[1];
    let v = x_prev[2];
    let theta = x_prev[3];
    Vector4::new(x + v * theta.cos() * dt, y + v * theta.sin() * dt, v, theta)
}

/// 観測関数 h(x)
fn observation_function(x_curr: &Vector4<f64>) -> Vector2<f64> {
    let x = x_curr[0];
    let y = x_curr[1];
    Vector2::new(x, y)
}

// --------------------------------------------------------------------------------
// --- 拡張カルマンフィルタ(EKF)の実装 ---
// --------------------------------------------------------------------------------
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
        ExtendedKalmanFilter {
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

    fn calculate_jacobian_h(&self) -> SMatrix<f64, 2, 4> {
        SMatrix::<f64, 2, 4>::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    }

    pub fn predict(&mut self) {
        let f_jacobian = self.calculate_jacobian_f(&self.x);
        self.x = state_transition_function(&self.x, self.dt);
        self.p = f_jacobian * self.p * f_jacobian.transpose() + self.q;
    }

    pub fn update(&mut self, z: Vector2<f64>) {
        let h_jacobian = self.calculate_jacobian_h();
        let y = z - observation_function(&self.x);
        let s_inv = (h_jacobian * self.p * h_jacobian.transpose() + self.r)
            .try_inverse()
            .expect("EKF: S行列が逆行列を持たない");
        let k = self.p * h_jacobian.transpose() * s_inv;
        self.x += k * y;
        self.p = (Matrix4::identity() - k * h_jacobian) * self.p;
    }

    pub fn get_state(&self) -> Vector4<f64> {
        self.x
    }
}

// --------------------------------------------------------------------------------
// --- アンセンテッドカルマンフィルタ(UKF)の実装 ---
// --------------------------------------------------------------------------------
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

        UnscentedKalmanFilter {
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
            .expect("UKF: Pのコレスキー分解に失敗")
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
        let k = t * s.try_inverse().expect("UKF: S行列が逆行列を持たない");
        self.x += k * (z - z_pred);
        self.p -= k * s * k.transpose();
    }

    pub fn get_state(&self) -> Vector4<f64> {
        self.x
    }
}

// --------------------------------------------------------------------------------
// --- キューバチャーカルマンフィルタ(CKF)の実装 ---
// --------------------------------------------------------------------------------
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
        CubatureKalmanFilter {
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
        let s = Cholesky::new(*p).expect("CKF: Pのコレスキー分解に失敗").l();
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

        // 予測状態の計算
        self.x = propagated_points.column_mean();

        // 予測共分散の計算
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

        let k = p_xz * p_zz.try_inverse().expect("CKF: S行列が逆行列を持たない");
        self.x += k * (z - z_pred);
        self.p -= k * p_zz * k.transpose();
    }

    pub fn get_state(&self) -> Vector4<f64> {
        self.x
    }
}

// --------------------------------------------------------------------------------
// --- メインのシミュレーション ---
// --------------------------------------------------------------------------------
fn main() {
    let mut rng: ThreadRng = rand::rng();
    let dt = 0.1;

    let initial_x = Vector4::new(0.0, 0.0, 1.0, 0.0);
    let initial_p = Matrix4::from_diagonal(&Vector4::new(100.0, 100.0, 10.0, 10.0));

    // --- 4つのフィルタをインスタンス化 ---
    let mut ekf = ExtendedKalmanFilter::new(initial_x, initial_p, dt);
    let mut ukf = UnscentedKalmanFilter::new(initial_x, initial_p, dt);
    let mut ckf = CubatureKalmanFilter::new(initial_x, initial_p, dt);

    println!("EKF, UKF, CKF 比較シミュレーションを開始します。");
    println!("{:-<180}", "");
    println!(
        "{:<10} | {:<40} | {:<25} | {:<40} | {:<40}",
        "ステップ", "真の状態", "観測値", "EKF 推定値", "UKF 推定値"
    );
    println!(
        "{:<10} | {:<40} | {:<25} | {:<40} |",
        "", "", "", "CKF 推定値"
    );
    println!("{:-<180}", "");

    // シミュレーションループ
    for i in 0..100 {
        // 真の状態 (円運動)
        let true_v = 1.0;
        let true_angular_velocity = 0.05;
        let true_theta = true_angular_velocity * (i as f64 + 1.0) * dt;
        let radius = true_v / true_angular_velocity;
        let true_x_curr = initial_x[0] + radius * true_theta.sin();
        let true_y_curr = initial_x[1] + radius * (1.0 - true_theta.cos());
        let true_state = Vector4::new(true_x_curr, true_y_curr, true_v, true_theta);

        // 観測値の生成 (真の位置にノイズを加える)
        let observation_noise_x = (rng.random::<f64>() - 0.5) * 0.5;
        let observation_noise_y = (rng.random::<f64>() - 0.5) * 0.5;
        let observation = Vector2::new(
            true_state[0] + observation_noise_x,
            true_state[1] + observation_noise_y,
        );

        // --- 各フィルタの予測と更新 ---
        ekf.predict();
        ekf.update(observation);

        ukf.predict();
        ukf.update(observation);

        ckf.predict();
        ckf.update(observation);

        // --- 結果表示 ---
        if i % 10 == 0 || i == 99 {
            let true_str = format!(
                "x={:5.2}, y={:5.2}, v={:4.2}, th={:4.2}",
                true_state[0], true_state[1], true_state[2], true_state[3]
            );
            let obs_str = format!("x={:5.2}, y={:5.2}", observation[0], observation[1]);

            let format_state = |s: Vector4<f64>| {
                format!(
                    "x={:5.2}, y={:5.2}, v={:4.2}, th={:4.2}",
                    s[0], s[1], s[2], s[3]
                )
            };
            let ekf_str = format_state(ekf.get_state());
            let ukf_str = format_state(ukf.get_state());
            let ckf_str = format_state(ckf.get_state());

            println!(
                "Step {i:<5} | {true_str:<40} | {obs_str:<25} | {ekf_str:<40} | {ukf_str:<40}",
            );
            println!("{:<10} | {:<40} | {:<25} | {:<40} |", "", "", "", ckf_str);
            println!("{:-<180}", "");
        }
    }
    println!("\nシミュレーション終了。");
}
