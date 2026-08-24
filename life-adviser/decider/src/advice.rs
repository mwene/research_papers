use crate::five::FiveQuantities;
use crate::params::Resolved;

fn fmt_months(m: f64) -> String {
    if m >= 890.0 {
        "never".to_string()
    } else if m < 1.0 {
        "under a month".to_string()
    } else if m < 18.0 {
        format!("about {} month{}", m.round() as i64, if m.round() as i64 == 1 { "" } else { "s" })
    } else {
        format!("about {:.1} years", m / 12.0)
    }
}

fn pct(x: f64) -> String {
    format!("{:.0}%", (x * 100.0).round())
}

pub(crate) fn headline(action: crate::verdict::Action, _p: &Resolved, q: &FiveQuantities) -> String {
    use crate::verdict::Action::*;
    match action {
        ActNow => match (q.urgency > 0.55, q.expected_net_gain >= 0.0) {
            (true, true) => "Act now \u{2014} the window is closing while the odds are good.",
            (true, false) => "Act now \u{2014} this is damage control, and time is not on your side.",
            (false, true) => "Act now \u{2014} acting early beats being forced late.",
            (false, false) => "Act now \u{2014} mostly to protect the floor you cannot afford to lose.",
        },
        Wait => {
            if q.uncertainty_band > 0.5 {
                "Wait and watch \u{2014} the situation is uncertain and your position is not in danger yet."
            } else {
                "Wait \u{2014} nothing forces your hand yet, so gather more signal first."
            }
        }
        Drop => {
            if q.expected_net_gain < 0.0 {
                "Let it go \u{2014} the numbers say this path loses more than it wins."
            } else {
                "Let it go \u{2014} the goal is unreachable within any horizon that matters."
            }
        }
    }
    .to_string()
}

pub(crate) fn advice_lines(action: crate::verdict::Action, p: &Resolved, q: &FiveQuantities) -> Vec<String> {
    use crate::verdict::Action::*;
    let mut out = Vec::new();

    let forced_txt = match p.forced_in {
        Some(tf) => fmt_months(tf),
        None => "never".to_string(),
    };

    match action {
        ActNow => {
            if q.urgency > 0.55 {
                out.push(format!(
                    "Your voluntary window closes in {}: past that point, circumstances choose for you, and chosen actions historically succeed two-to-three times more often than forced ones.",
                    forced_txt
                ));
            }
            if q.expected_net_gain >= 0.0 {
                out.push(format!(
                    "Acting instead of drifting gains you roughly {:.2} of progress over {} (in goal-units where the whole distance is 1.0).",
                    q.expected_net_gain,
                    fmt_months(p.horizon)
                ));
            }
            if p.act_cost > 0.6 {
                out.push("The cost of moving first is real \u{2014} pay it anyway. Early cost is tuition; late cost is a fine with interest.".into());
            }
        }
        Wait => {
            out.push(format!(
                "Nothing forces your hand until {}. Use the time deliberately: define what new information would change your mind.",
                forced_txt
            ));
            if q.uncertainty_band > 0.35 {
                out.push(format!(
                    "Outcome uncertainty is wide ({:.2} in goal-units at your horizon), so waiting has genuine information value \u{2014} but set a review date, not an infinite deferral.",
                    q.uncertainty_band
                ));
            }
            if q.floor_risk_if_wait > 0.10 {
                out.push(format!(
                    "Caution: waiting lets floor-risk climb to {} within your horizon. Re-check sooner if things drift down.",
                    pct(q.floor_risk_if_wait)
                ));
            }
        }
        Drop => {
            out.push(format!(
                "Expected net movement of this path over {} is {:.2}: continuing trades effort for decline.",
                fmt_months(p.horizon),
                q.expected_net_gain
            ));
            if q.floor_risk_if_wait > 0.05 {
                out.push("If you drop this, still put a cheap guardrail on the floor \u{2014} disasters do not respect being ignored.".into());
            }
            out.push("Dropping is a decision, not a failure: resources released here are what fund your next target.".into());
        }
    }

    out.push(format!(
        "Floor-risk check: {} if you act, {} if you keep waiting over the same period.",
        pct(q.floor_risk_if_act),
        pct(q.floor_risk_if_wait)
    ));

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats() {
        assert_eq!(fmt_months(0.4), "under a month");
        assert_eq!(fmt_months(1.0), "about 1 month");
        assert_eq!(fmt_months(10.2), "about 10 months");
        assert_eq!(fmt_months(36.0), "about 3.0 years");
        assert_eq!(fmt_months(999.0), "never");
        assert_eq!(pct(0.226), "23%");
    }

    #[test]
    fn lines_nonempty_all_actions() {
        let p = crate::params::DecisionParams::default().resolved();
        let q = crate::five::compute_five(&p);
        for a in [crate::verdict::Action::ActNow, crate::verdict::Action::Wait, crate::verdict::Action::Drop] {
            assert!(!advice_lines(a, &p, &q).is_empty());
            assert!(!headline(a, &p, &q).is_empty());
        }
    }
}
