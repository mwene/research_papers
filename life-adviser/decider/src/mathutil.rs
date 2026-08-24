pub(crate) fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t + -1.453152027) * t) + 1.421413741) * t + -0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

pub(crate) fn norm_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(std::f64::consts::FRAC_1_SQRT_2 * z))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-9);
        assert!((norm_cdf(1.96) - 0.975).abs() < 1e-4);
        assert!((norm_cdf(-1.0) - 0.15865525).abs() < 1e-4);
    }

    #[test]
    fn bounds_and_symmetry() {
        for z in [-6.0, -1.0, 0.3, 2.5, 6.0] {
            let p = norm_cdf(z);
            assert!((0.0..=1.0).contains(&p));
        }
        assert!((norm_cdf(1.3) + norm_cdf(-1.3) - 1.0).abs() < 1e-9);
    }
}
