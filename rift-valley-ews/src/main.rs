use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================
// Lake Coefficients
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LakeCoefficients {
    pub name: String,
    pub alpha: f64,   // precipitation sensitivity
    pub beta: f64,    // evaporation scaling
    pub gamma: f64,   // runoff coefficient
    pub delta: f64,   // groundwater coefficient
    pub epsilon: f64, // seepage coefficient
    pub zeta: f64,    // human extraction coefficient
    pub eta: f64,     // persistence (must be < 1)
    pub sigma_w: f64, // process noise std dev (m/month)
    pub l_crit: f64,  // critical flood level (m)
    pub l_baseline: f64, // baseline level (m, circa 2010)
}

impl LakeCoefficients {
    /// Restoring force: fraction of level lost each month to outflow/seepage
    /// not captured by individual terms. Must be positive for stability.
    pub fn restoring_force(&self) -> f64 {
        1.0 - self.eta
    }
}

// ============================================================
// Lake State
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LakeState {
    pub level: f64,        // current lake level (m)
    pub rate: f64,         // rate of change (m/month)
    pub precipitation: f64, // recent precip (m/month)
    pub evaporation: f64,   // recent evap (m/month)
    pub groundwater: f64,   // recent groundwater inflow (m/month)
}

// ============================================================
// Warning Action
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WarningAction {
    Monitor = 0,   // no action
    Alert = 1,     // issue public warning
    Evacuate = 2,  // order evacuation
    Emergency = 3, // mandatory evacuation
}

impl WarningAction {
    pub fn label(&self) -> &str {
        match self {
            WarningAction::Monitor => "MONITOR",
            WarningAction::Alert => "ALERT",
            WarningAction::Evacuate => "EVACUATE",
            WarningAction::Emergency => "EMERGENCY",
        }
    }

    pub fn color(&self) -> &str {
        match self {
            WarningAction::Monitor => "\x1b[32m",  // green
            WarningAction::Alert => "\x1b[33m",     // yellow
            WarningAction::Evacuate => "\x1b[38;5;208m", // orange
            WarningAction::Emergency => "\x1b[31m", // red
        }
    }

    /// Mitigation factor: fraction of flood damage remaining after action
    pub fn mitigation(&self) -> f64 {
        match self {
            WarningAction::Monitor => 1.0,
            WarningAction::Alert => 0.5,
            WarningAction::Evacuate => 0.2,
            WarningAction::Emergency => 0.05,
        }
    }
}

// ============================================================
// Water Balance Model
// ============================================================

pub struct WaterBalanceModel;

impl WaterBalanceModel {
    /// Predict next month's lake level.
    /// L(t+1) = eta*L(t) + alpha*(P - beta*E) + gamma*R + delta*G - epsilon*S - zeta*H + xi + w
    pub fn predict(
        coeff: &LakeCoefficients,
        state: &LakeState,
        runoff: f64,
        seepage: f64,
        extraction: f64,
        seasonal: f64,
        rng: &mut impl rand::Rng,
    ) -> f64 {
        let deterministic = coeff.eta * state.level
            + coeff.alpha * (state.precipitation - coeff.beta * state.evaporation)
            + coeff.gamma * runoff
            + coeff.delta * state.groundwater
            - coeff.epsilon * seepage
            - coeff.zeta * extraction
            + seasonal;

        let noise: f64 = rng.gen::<f64>() * 2.0 - 1.0; // uniform [-1, 1]
        let noise = noise * coeff.sigma_w;

        deterministic + noise
    }

    /// Predict next state (deterministic, for Bellman solver)
    pub fn predict_deterministic(
        coeff: &LakeCoefficients,
        state: &LakeState,
        runoff: f64,
        seepage: f64,
        extraction: f64,
        seasonal: f64,
    ) -> f64 {
        coeff.eta * state.level
            + coeff.alpha * (state.precipitation - coeff.beta * state.evaporation)
            + coeff.gamma * runoff
            + coeff.delta * state.groundwater
            - coeff.epsilon * seepage
            - coeff.zeta * extraction
            + seasonal
    }
}

// ============================================================
// Kalman Filter for State Estimation
// ============================================================

#[derive(Debug, Clone)]
pub struct KalmanFilter {
    pub estimate: f64,     // filtered level estimate
    pub variance: f64,     // filtered variance
    pub process_noise: f64,
    pub measurement_noise: f64,
}

impl KalmanFilter {
    pub fn new(initial_level: f64, initial_variance: f64, process_noise: f64, measurement_noise: f64) -> Self {
        Self {
            estimate: initial_level,
            variance: initial_variance,
            process_noise,
            measurement_noise,
        }
    }

    /// Predict step: advance state by one time step
    pub fn predict(&mut self, model_prediction: f64) {
        // State prediction
        self.estimate = model_prediction;
        // Variance prediction (process noise adds uncertainty)
        self.variance += self.process_noise;
    }

    /// Update step: incorporate a new measurement
    pub fn update(&mut self, measurement: f64) {
        // Kalman gain
        let gain = self.variance / (self.variance + self.measurement_noise);
        // State update
        self.estimate += gain * (measurement - self.estimate);
        // Variance update
        self.variance *= 1.0 - gain;
    }

    /// Combined predict-update cycle
    pub fn step(&mut self, model_prediction: f64, measurement: Option<f64>) {
        self.predict(model_prediction);
        if let Some(y) = measurement {
            self.update(y);
        }
    }

    pub fn std_dev(&self) -> f64 {
        self.variance.sqrt()
    }
}

// ============================================================
// Bellman Solver for Optimal Thresholds
// ============================================================

pub struct BellmanSolver {
    pub n_grid: usize,       // number of grid points
    pub l_min: f64,          // minimum level on grid
    pub l_max: f64,          // maximum level on grid
    pub dt: f64,             // time step (months)
    pub flood_cost: f64,     // base flood damage cost
    pub false_alarm_costs: [f64; 4], // false alarm cost per action
}

impl BellmanSolver {
    pub fn new(n_grid: usize, l_min: f64, l_max: f64) -> Self {
        Self {
            n_grid,
            l_min,
            l_max,
            dt: 1.0,
            flood_cost: 1_000_000.0, // $1M base damage
            false_alarm_costs: [0.0, 10_000.0, 50_000.0, 200_000.0],
        }
    }

    /// Grid spacing
    pub fn dl(&self) -> f64 {
        (self.l_max - self.l_min) / (self.n_grid - 1) as f64
    }

    /// Level at grid index i
    pub fn level_at(&self, i: usize) -> f64 {
        self.l_min + i as f64 * self.dl()
    }

    /// Flood cost as function of level above critical
    pub fn flood_damage(&self, level: f64, l_crit: f64) -> f64 {
        if level > l_crit {
            let excess = level - l_crit;
            self.flood_cost * (1.0 + excess * 10.0) // cost grows with excess
        } else {
            0.0
        }
    }

    /// Solve Bellman equation for a single lake.
    /// Returns optimal thresholds [theta_1, theta_2, theta_3].
    pub fn solve(&self, coeff: &LakeCoefficients, horizon: usize) -> Vec<f64> {
        let n = self.n_grid;
        let l_crit = coeff.l_crit;

        // Value function: V[t][i] = value at time t, grid point i
        let mut v_curr = vec![0.0f64; n];
        let mut v_next = vec![0.0f64; n];
        let mut policy = vec![0u8; n]; // optimal action at each grid point

        // Terminal condition
        for i in 0..n {
            let l = self.level_at(i);
            v_curr[i] = -self.flood_damage(l, l_crit);
        }

        // Backward induction
        for _t in 0..horizon {
            for i in 0..n {
                let l = self.level_at(i);

                let mut best_val = f64::NEG_INFINITY;
                let mut best_action = 0u8;

                for a in 0u8..=3 {
                    // Expected payoff
                    let mitigation = WarningAction::mitigation_from_u8(a);
                    let flood_dmg = self.flood_damage(l, l_crit) * mitigation;
                    let false_alarm = if l <= l_crit {
                        self.false_alarm_costs[a as usize]
                    } else {
                        0.0
                    };

                    let immediate = -(flood_dmg + false_alarm);

                    // Expected future value: integrate over possible next levels
                    // Use deterministic prediction + averaging over noise
                    let next_level = WaterBalanceModel::predict_deterministic(
                        coeff,
                        &LakeState {
                            level: l,
                            rate: 0.0,
                            precipitation: 50.0 / 1000.0, // 50 mm/month default
                            evaporation: 30.0 / 1000.0,
                            groundwater: 5.0 / 1000.0,
                        },
                        10.0 / 1000.0,  // runoff
                        2.0 / 1000.0,   // seepage
                        1.0 / 1000.0,   // extraction
                        0.0,            // seasonal
                    );

                    // Interpolate next_level on grid
                    let future_val = interpolate_v(&v_next, self, next_level);

                    let total = immediate + future_val;

                    if total > best_val {
                        best_val = total;
                        best_action = a;
                    }
                }

                v_next[i] = best_val;
                policy[i] = best_action;
            }

            std::mem::swap(&mut v_curr, &mut v_next);
        }

        // Extract thresholds from policy
        extract_thresholds(&policy, self)
    }
}

fn interpolate_v(v: &[f64], solver: &BellmanSolver, level: f64) -> f64 {
    let dl = solver.dl();
    let idx = (level - solver.l_min) / dl;
    let i0 = idx.floor().max(0.0).min((solver.n_grid - 2) as f64) as usize;
    let i1 = i0 + 1;
    let frac = idx - i0 as f64;
    v[i0] * (1.0 - frac) + v[i1].min(v[i0]) * frac // clamp: don't extrapolate beyond grid
}

fn extract_thresholds(policy: &[u8], solver: &BellmanSolver) -> Vec<f64> {
    let mut thresholds = Vec::new();
    for target_action in 1u8..=3 {
        // Find lowest level where policy >= target_action
        let mut found = solver.l_max;
        for i in 0..solver.n_grid {
            if policy[i] >= target_action {
                found = solver.level_at(i);
                break;
            }
        }
        thresholds.push(found);
    }
    thresholds
}

// ============================================================
// Early Warning System
// ============================================================

pub struct EarlyWarningSystem {
    pub lakes: HashMap<String, LakeState>,
    pub coefficients: HashMap<String, LakeCoefficients>,
    pub filters: HashMap<String, KalmanFilter>,
    pub thresholds: HashMap<String, Vec<f64>>,
    pub false_alarm_count: HashMap<String, u32>,
    pub total_alerts: HashMap<String, u32>,
    pub month: u32,
}

impl EarlyWarningSystem {
    pub fn new() -> Self {
        let mut coefficients = HashMap::new();
        let mut lakes = HashMap::new();
        let mut filters = HashMap::new();
        let mut thresholds = HashMap::new();

        // Calibrated coefficients from the lake paper
        let lake_data = vec![
            ("Turkana".to_string(),     0.55, 0.18, 0.20, 0.15, 0.05, 0.03, 0.90, 0.030, 13.0, 6.0),
            ("Baringo".to_string(),     0.80, 0.10, 0.15, 0.05, 0.02, 0.04, 0.95, 0.015, 12.0, 6.0),
            ("Bogoria".to_string(),     0.82, 0.22, 0.10, 0.04, 0.02, 0.02, 0.90, 0.020, 11.0, 5.0),
            ("Nakuru".to_string(),      0.85, 0.20, 0.12, 0.03, 0.03, 0.03, 0.92, 0.019, 10.0, 4.5),
            ("Elementaita".to_string(), 0.88, 0.25, 0.10, 0.02, 0.03, 0.02, 0.95, 0.021, 9.5,  3.5),
            ("Naivasha".to_string(),    0.78, 0.15, 0.18, 0.06, 0.04, 0.05, 0.93, 0.020, 10.0, 4.0),
            ("Magadi".to_string(),      0.65, 0.35, 0.03, 0.04, 0.03, 0.01, 0.85, 0.023, 8.0,  2.5),
            ("Solai".to_string(),       0.92, 0.12, 0.15, 0.03, 0.01, 0.03, 0.98, 0.017, 11.0, 4.0),
        ];

        for (name, alpha, beta, gamma, delta, epsilon, zeta, eta, sigma_w, l_crit, l_base) in lake_data {
            let coeff = LakeCoefficients {
                name: name.clone(),
                alpha, beta, gamma, delta, epsilon, zeta, eta, sigma_w, l_crit, l_baseline: l_base,
            };

            let state = LakeState {
                level: l_base,
                rate: 0.0,
                precipitation: 50.0 / 1000.0,
                evaporation: 30.0 / 1000.0,
                groundwater: 5.0 / 1000.0,
            };

            let filter = KalmanFilter::new(l_base, 0.1, coeff.sigma_w.powi(2), 0.05);

            // Compute thresholds via Bellman solver
            let solver = BellmanSolver::new(200, l_base, l_crit + 5.0);
            let thresh = solver.solve(&coeff, 60); // 60-month horizon

            lakes.insert(name.clone(), state);
            coefficients.insert(name.clone(), coeff);
            filters.insert(name.clone(), filter);
            thresholds.insert(name.clone(), thresh);
        }

        Self {
            lakes,
            coefficients,
            filters,
            thresholds,
            false_alarm_count: HashMap::new(),
            total_alerts: HashMap::new(),
            month: 0,
        }
    }

    /// Determine the optimal warning action for a lake
    pub fn decide(&self, lake_name: &str) -> WarningAction {
        let filter = self.filters.get(lake_name).unwrap();
        let thresh = self.thresholds.get(lake_name).unwrap();
        let level = filter.estimate;

        if level >= thresh[2] {
            WarningAction::Emergency
        } else if level >= thresh[1] {
            WarningAction::Evacuate
        } else if level >= thresh[0] {
            WarningAction::Alert
        } else {
            WarningAction::Monitor
        }
    }

    /// Compute hitting time (expected months to critical level)
    pub fn hitting_time(&self, lake_name: &str) -> f64 {
        let coeff = self.coefficients.get(lake_name).unwrap();
        let filter = self.filters.get(lake_name).unwrap();
        let gap = coeff.l_crit - filter.estimate;

        if gap <= 0.0 {
            return 0.0; // already at or above critical
        }

        // Expected monthly rise: alpha * (P - beta*E) component
        // Using current state values
        let state = self.lakes.get(lake_name).unwrap();
        let monthly_rise = coeff.alpha * (state.precipitation - coeff.beta * state.evaporation);

        if monthly_rise <= 0.0 {
            return f64::INFINITY; // not rising
        }

        gap / monthly_rise
    }

    /// Compute violation probability (probability of exceeding critical level in T months)
    pub fn violation_probability(&self, lake_name: &str, horizon_months: f64) -> f64 {
        let coeff = self.coefficients.get(lake_name).unwrap();
        let filter = self.filters.get(lake_name).unwrap();
        let state = self.lakes.get(lake_name).unwrap();

        let mu = coeff.alpha * (state.precipitation - coeff.beta * state.evaporation);
        let sigma = coeff.sigma_w;
        let gap = coeff.l_crit - filter.estimate;

        if sigma <= 0.0 || horizon_months <= 0.0 {
            return if gap <= 0.0 { 1.0 } else { 0.0 };
        }

        let z = (gap - mu * horizon_months) / (sigma * horizon_months.sqrt());
        // P(violation) = 1 - Phi(z)
        1.0 - normal_cdf(z)
    }

    /// Godelian self-check: compare predicted vs observed
    pub fn godelian_check(&mut self, lake_name: &str, observed_level: f64) -> String {
        let filter = self.filters.get(lake_name).unwrap();
        let error = (filter.estimate - observed_level).abs();

        if error > 0.05 {
            format!(
                "[GODELIAN] {} prediction error {:.3} m exceeds 0.05 m threshold. Recalibration recommended.",
                lake_name, error
            )
        } else {
            format!(
                "[GODELIAN] {} prediction error {:.3} m. Model within tolerance.",
                lake_name, error
            )
        }
    }

    /// Advance the system by one month
    pub fn step(
        &mut self,
        precipitation_mm: f64,  // mm/month
        evaporation_mm: f64,    // mm/month
        runoff_mm: f64,         // mm/month
        extraction_mm: f64,     // mm/month
        rng: &mut impl rand::Rng,
    ) -> Vec<(String, WarningAction, f64, f64)> {
        self.month += 1;
        let mut results = Vec::new();

        let lake_names: Vec<String> = self.lakes.keys().cloned().collect();

        for name in &lake_names {
            let coeff = self.coefficients.get(name).unwrap();
            let state = self.lakes.get(name).unwrap();

            // Convert mm to meters
            let p = precipitation_mm / 1000.0;
            let e = evaporation_mm / 1000.0;
            let r = runoff_mm / 1000.0;
            let h = extraction_mm / 1000.0;

            // Model prediction
            let predicted = WaterBalanceModel::predict(
                coeff,
                state,
                r,
                2.0 / 1000.0, // seepage (small)
                h,
                0.0,           // seasonal
                rng,
            );

            // Update Kalman filter
            let filter = self.filters.get_mut(name).unwrap();
            filter.step(predicted, Some(predicted + rng.gen::<f64>() * 0.01 - 0.005));

            // Update lake state
            let new_level = filter.estimate;
            let new_rate = new_level - state.level;

            self.lakes.insert(name.clone(), LakeState {
                level: new_level,
                rate: new_rate,
                precipitation: p,
                evaporation: e,
                groundwater: state.groundwater,
            });

            // Decide
            let action = self.decide(name);

            // Track false alarms
            if action != WarningAction::Monitor {
                *self.total_alerts.entry(name.clone()).or_insert(0) += 1;
                if new_level <= coeff.l_crit {
                    *self.false_alarm_count.entry(name.clone()).or_insert(0) += 1;
                }
            }

            let ht = self.hitting_time(name);
            results.push((name.clone(), action, new_level, ht));
        }

        results
    }

    /// Print dashboard
    pub fn print_dashboard(&self) {
        println!("\n{}", "=".repeat(80));
        println!("  RIFT VALLEY LAKE EARLY WARNING SYSTEM  |  Month: {}", self.month);
        println!("{}\n", "=".repeat(80));
        println!(
            "{:<15} {:>10} {:>10} {:>10} {:>12} {:>12} {:>10}",
            "Lake", "Level(m)", "Crit.(m)", "Status", "Hit.Time", "Viol.Prob", "F.A.Rate"
        );
        println!("{}", "-".repeat(80));

        let names = ["Turkana", "Baringo", "Bogoria", "Nakuru", "Elementaita", "Naivasha", "Magadi", "Solai"];

        for name in &names {
            if let Some(state) = self.lakes.get(*name) {
                let coeff = self.coefficients.get(*name).unwrap();
                let action = self.decide(name);
                let ht = self.hitting_time(name);
                let vp = self.violation_probability(name, 12.0);

                let alerts = self.total_alerts.get(*name).unwrap_or(&0);
                let false_alarms = self.false_alarm_count.get(*name).unwrap_or(&0);
                let far = if *alerts > 0 {
                    *false_alarms as f64 / *alerts as f64 * 100.0
                } else {
                    0.0
                };

                let ht_str = if ht.is_infinite() {
                    ">999".to_string()
                } else if ht == 0.0 {
                    "NOW".to_string()
                } else {
                    format!("{:.0} mo", ht)
                };

                println!(
                    "{:<15} {:>10.3} {:>10.1} {}{:<10}\x1b[0m {:>12} {:>11.1}% {:>9.1}%",
                    name,
                    state.level,
                    coeff.l_crit,
                    action.color(),
                    action.label(),
                    ht_str,
                    vp * 100.0,
                    far,
                );
            }
        }
        println!("{}", "=".repeat(80));
    }
}

// ============================================================
// Utility Functions
// ============================================================

fn normal_cdf(x: f64) -> f64 {
    // Approximation of the standard normal CDF
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() / 2.0_f64.sqrt();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    0.5 * (1.0 + sign * y)
}

impl WarningAction {
    fn mitigation_from_u8(a: u8) -> f64 {
        match a {
            0 => 1.0,
            1 => 0.5,
            2 => 0.2,
            3 => 0.05,
            _ => 1.0,
        }
    }
}

// ============================================================
// Main: Simulate the Early Warning System
// ============================================================

fn main() {
    use rand::Rng;
    println!("Rift Valley Lake Early Warning System");
    println!("Based on: Water Balance Model + Stochastic Control Theory\n");

    let mut ews = EarlyWarningSystem::new();
    let mut rng = rand::thread_rng();

    // Simulate 24 months of operation
    // Scenario: increased rainfall post-2018 pattern
    for month in 0..24 {
        // Simulate seasonal precipitation pattern (mm/month)
        // Higher in Mar-May and Oct-Dec (long/short rains)
        let seasonal_factor = (month as f64 * std::f64::consts::PI / 6.0).sin();
        let base_precip = 60.0 + 30.0 * seasonal_factor; // 30-90 mm/month
        let precip = base_precip + rng.gen::<f64>() * 20.0 - 10.0;
        let evap = 35.0 + rng.gen::<f64>() * 10.0 - 5.0;
        let runoff = 8.0 + rng.gen::<f64>() * 4.0 - 2.0;
        let extraction = 2.0 + rng.gen::<f64>() * 2.0;

        let results = ews.step(precip, evap, runoff, extraction, &mut rng);

        // Print dashboard every 3 months
        if month % 3 == 0 || results.iter().any(|(_, a, _, _)| *a != WarningAction::Monitor) {
            ews.print_dashboard();
        }

        // Print alerts
        for (name, action, level, ht) in &results {
            if *action != WarningAction::Monitor {
                let coeff = ews.coefficients.get(name).unwrap();
                println!(
                    "  >>> {} {:?}: level {:.3} m, crit {:.1} m, hitting time {:.0} months",
                    name, action, level, coeff.l_crit, ht
                );
            }
        }
    }

    // Final summary
    println!("\n{}", "=".repeat(80));
    println!("  SIMULATION COMPLETE");
    println!("  Total months simulated: {}", ews.month);
    println!("  Lakes monitored: 8");
    println!("{}", "=".repeat(80));

    // Export final state as JSON
    let output = serde_json::json!({
        "month": ews.month,
        "lakes": ews.lakes,
        "thresholds": ews.thresholds,
        "false_alarm_counts": ews.false_alarm_count,
        "total_alerts": ews.total_alerts,
    });

    if let Ok(json) = serde_json::to_string_pretty(&output) {
        std::fs::write("ews_output.json", &json).ok();
        println!("\nFinal state exported to ews_output.json");
    }
}
