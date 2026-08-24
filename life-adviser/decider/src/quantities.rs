use crate::mathutil::norm_cdf;

pub(crate) fn hitting_prob(x0: f64, mu: f64, sigma: f64, t_months: f64, floor: f64) -> f64 {
    let sigma = sigma.clamp(0.01, 2.0);
    let t = t_months.max(1e-6);
    let dt = sigma * t.sqrt();
    let d = floor - x0;
    let term1 = norm_cdf((d - mu * t) / dt);
    let expo = (2.0 * mu * d / (sigma * sigma)).clamp(-50.0, 50.0);
    let term2 = expo.exp() * norm_cdf((d + mu * t) / dt);
    (term1 + term2).clamp(0.0, 1.0)
}

pub(crate) fn time_to_reach(distance: f64, rate: f64) -> Option<f64> {
    if rate <= 1e-6 || distance <= 0.0 {
        return None;
    }
    Some(distance / rate)
}

pub(crate) fn urgency(forced_in: Option<f64>, horizon: f64) -> f64 {
    match forced_in {
        None => 0.08,
        Some(tf) => {
            if tf <= 0.25 {
                return 1.0;
            }
            1.0 - (-horizon / tf).exp()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn risk_zero_when_far_above_floor_short_time() {
        let p = hitting_prob(0.8, 0.02, 0.05, 3.0, 0.05);
        assert!(p < 0.001, "got {p}");
    }

    #[test]
    fn risk_one_when_start_below_floor() {
        let p = hitting_prob(0.1, 0.02, 0.2, 6.0, 0.3);
        assert!(approx(p, 1.0, 1e-6));
    }

    #[test]
    fn risk_increases_with_horizon() {
        let short = hitting_prob(0.4, -0.02, 0.15, 3.0, 0.1);
        let long = hitting_prob(0.4, -0.02, 0.15, 24.0, 0.1);
        assert!(long > short);
        assert!(long <= 1.0 && short >= 0.0);
    }

    #[test]
    fn driftless_equals_reflection_principle() {
        let p = hitting_prob(0.5, 0.0, 0.2, 9.0, 0.2);
        let refl = 2.0 * norm_cdf(-0.3 / (0.2 * 9.0f64.sqrt()));
        assert!(approx(p, refl.min(1.0), 1e-9));
    }

    #[test]
    fn positive_drift_lowers_risk() {
        let up = hitting_prob(0.4, 0.03, 0.15, 12.0, 0.1);
        let down = hitting_prob(0.4, -0.03, 0.15, 12.0, 0.1);
        assert!(up < down);
    }

    #[test]
    fn hitting_time_basic() {
        assert!(approx(time_to_reach(0.5, 0.05).unwrap(), 10.0, 1e-9));
        assert!(time_to_reach(0.5, 0.0).is_none());
        assert!(time_to_reach(0.0, 0.05).is_none());
    }

    #[test]
    fn urgency_monotone_in_forced_time() {
        let near = urgency(Some(2.0), 12.0);
        let mid = urgency(Some(12.0), 12.0);
        let far = urgency(Some(120.0), 12.0);
        let unbounded = urgency(None, 12.0);
        assert!(near > mid && mid > far && far > unbounded);
        assert!(approx(urgency(Some(0.1), 12.0), 1.0, 1e-9));
    }
}
