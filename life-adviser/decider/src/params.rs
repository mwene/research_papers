use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct DecisionParams {
    pub description: Option<String>,
    pub domain: Option<String>,
    pub state_now: Option<f64>,
    pub floor_distance: Option<f64>,
    pub gain_rate: Option<f64>,
    pub wait_drift: Option<f64>,
    pub volatility: Option<f64>,
    pub horizon_months: Option<f64>,
    pub forced_in_months: Option<f64>,
    pub act_cost: Option<f64>,
    pub floor_severity: Option<f64>,
    pub data_quality: Option<f64>,
}

pub(crate) struct Resolved {
    pub state_now: f64,
    pub floor_distance: f64,
    pub gain_rate: f64,
    pub wait_drift: f64,
    pub volatility: f64,
    pub horizon: f64,
    pub forced_in: Option<f64>,
    pub act_cost: f64,
    pub floor_severity: f64,
    pub data_quality: f64,
}

impl DecisionParams {
    pub fn validate(&self) -> Vec<String> {
        let mut w = Vec::new();
        let checks: [(Option<f64>, f64, f64, &str); 8] = [
            (self.state_now, 0.0, 1.0, "state_now"),
            (self.floor_distance, 0.0, 1.0, "floor_distance"),
            (self.gain_rate, -0.5, 0.5, "gain_rate"),
            (self.wait_drift, -0.5, 0.5, "wait_drift"),
            (self.volatility, 0.0, 2.0, "volatility"),
            (self.horizon_months, 0.0, 1200.0, "horizon_months"),
            (self.act_cost, 0.0, 1.0, "act_cost"),
            (self.floor_severity, 0.0, 1.0, "floor_severity"),
        ];
        for (v, lo, hi, name) in checks {
            if let Some(x) = v {
                if !(lo..=hi).contains(&x) {
                    w.push(format!("{name}={x} outside [{lo},{hi}]; value was clamped"));
                }
            }
        }
        if let Some(tf) = self.forced_in_months {
            if tf < 0.0 || tf > 1200.0 {
                w.push(format!("forced_in_months={tf} outside [0,1200]; use a large value for 'never'"));
            }
        }
        w
    }

    pub(crate) fn resolved(&self) -> Resolved {
        let horizon = clampf(self.horizon_months.unwrap_or(12.0), 0.25, 600.0);
        let forced_raw = self.forced_in_months;
        let forced_in = match forced_raw {
            None => Some(horizon * 2.5),
            Some(tf) if tf >= 900.0 => None,
            Some(tf) => Some(clampf(tf, 0.25, 900.0)),
        };
        Resolved {
            state_now: clampf(self.state_now.unwrap_or(0.5), 0.02, 0.98),
            floor_distance: clampf(self.floor_distance.unwrap_or(0.35), 0.03, 1.0),
            gain_rate: clampf(self.gain_rate.unwrap_or(0.01), -0.2, 0.2),
            wait_drift: clampf(self.wait_drift.unwrap_or(0.0), -0.15, 0.1),
            volatility: clampf(self.volatility.unwrap_or(0.15), 0.02, 0.8),
            horizon,
            forced_in,
            act_cost: clampf(self.act_cost.unwrap_or(0.4), 0.0, 1.0),
            floor_severity: clampf(self.floor_severity.unwrap_or(0.3), 0.0, 1.0),
            data_quality: clampf(self.data_quality.unwrap_or(1.0), 0.0, 1.0),
        }
    }
}

fn clampf(x: f64, lo: f64, hi: f64) -> f64 {
    x.clamp(lo, hi)
}
