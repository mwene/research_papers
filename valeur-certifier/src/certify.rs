//! The certifier.
//!
//! Given a valued field described by the data that the transfer theorem
//! consumes, decide whether that theorem applies.
//!
//! Two verdicts are reported separately, because they answer different
//! questions and have different hypotheses.
//!
//!   * Tameness depends only on the residue characteristic, the
//!     `p`-divisibility of the value group, and the perfection of the
//!     residue field. It is meaningful in every characteristic.
//!
//!   * The decidability transfer additionally needs equal characteristic.
//!     In mixed characteristic relative completeness is known to *fail* and
//!     relative decidability is an open question, so no verdict is possible
//!     and none is claimed.
//!
//! Soundness contract:
//!
//!   `certify` returns `Certified` only when every hypothesis has been
//!   independently justified. Otherwise it returns a refusal naming the
//!   blocking clause. It never reports a theory to be undecidable, and it
//!   never reports `Unknown` inputs to be equal in characteristic.

use crate::decide::{field_theory_decidable, group_theory_decidable, is_perfect, p_divisible};
use crate::model::{Char, Tri, ValuedField};

#[derive(Clone, Debug)]
pub struct Check {
    pub name: &'static str,
    pub status: Tri,
    pub detail: String,
}

#[derive(Clone, Debug)]
pub struct Certificate {
    pub field: ValuedField,
    pub tameness: Vec<Check>,
    pub transfer: Vec<Check>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Every transfer hypothesis justified.
    Certified,
    /// The field is tame, but the decidability transfer was not licensed.
    TameOnly,
    /// A tameness hypothesis genuinely failed.
    Refused,
}

impl Verdict {
    pub fn label(self) -> &'static str {
        match self {
            Verdict::Certified => "CERTIFIED",
            Verdict::TameOnly => "TAME, TRANSFER NOT ESTABLISHED",
            Verdict::Refused => "NOT-CERTIFIED",
        }
    }
}

impl Certificate {
    pub fn mixed_characteristic(&self) -> bool {
        let k = self.field.char_field;
        let r = self.field.residue_char();
        !matches!(k, Char::Unknown) && !matches!(r, Char::Unknown) && k != r
    }

    pub fn verdict(&self) -> Verdict {
        let tame = self.tameness.iter().all(|c| c.status == Tri::Yes);
        if !tame {
            return Verdict::Refused;
        }
        if self.transfer.iter().all(|c| c.status == Tri::Yes) {
            Verdict::Certified
        } else {
            Verdict::TameOnly
        }
    }

    pub fn certified(&self) -> bool {
        self.verdict() == Verdict::Certified
    }

    pub fn all_checks(&self) -> impl Iterator<Item = &Check> {
        self.tameness.iter().chain(self.transfer.iter())
    }

    /// Every non-`Yes` clause, tameness first, phrased as a blocker.
    pub fn blockers(&self) -> Vec<String> {
        self.all_checks()
            .filter(|c| c.status != Tri::Yes)
            .map(|c| match c.status {
                Tri::No => format!("{} FAILS: {}", c.name, c.detail),
                _ => format!("{} NOT ESTABLISHED: {}", c.name, c.detail),
            })
            .collect()
    }

    /// The conclusion the certificate licenses, if any.
    pub fn conclusion(&self) -> Option<String> {
        if !self.certified() {
            return None;
        }
        Some(format!(
            "Every maximal immediate extension (Z,w) of {} is tame and has a \
             decidable L_val-theory.",
            self.field.pretty()
        ))
    }

    /// A one-line account of why the transfer did not go through.
    pub fn transfer_note(&self) -> Option<String> {
        match self.verdict() {
            Verdict::Certified => None,
            Verdict::Refused => Some(
                "The field is not tame, so no maximal immediate extension of it can \
                 be tame either."
                    .to_string(),
            ),
            Verdict::TameOnly => {
                if self.mixed_characteristic() {
                    Some(
                        "The field is tame, and that is certified. The decidability \
                         transfer is a separate matter: it requires equal characteristic. \
                         In mixed characteristic relative completeness is known to fail \
                         and relative decidability remains open, so the tool declines \
                         rather than extrapolating from the equal-characteristic theorem."
                            .to_string(),
                    )
                } else {
                    Some(
                        "The field is tame. The transfer needs recursive theories of \
                         the value group and residue field, and no proof is on file."
                            .to_string(),
                    )
                }
            }
        }
    }
}

pub fn certify(vf: &ValuedField) -> Certificate {
    let kappa_char = vf.residue_char();
    let mut tameness = Vec::new();
    let mut transfer = Vec::new();

    // --- Tameness, valid in every characteristic ----------------------------

    let tf1 = match kappa_char {
        Char::Zero => Tri::Yes,
        Char::Prime(p) => p_divisible(&vf.group, p),
        Char::Unknown => Tri::Unknown,
    };
    tameness.push(Check {
        name: "(TF1) p-divisibility of Gamma_v",
        status: tf1,
        detail: match kappa_char {
            Char::Unknown => format!(
                "char(kappa(v)) is unknown, so no verdict on Gamma_v = {}",
                vf.group
            ),
            Char::Zero => format!(
                "char(kappa(v)) = 0, so the condition is vacuous; Gamma_v = {}",
                vf.group
            ),
            Char::Prime(p) => format!(
                "char(kappa(v)) = {}, so Gamma_v = {} must be {}-divisible",
                p, vf.group, p
            ),
        },
    });

    let tf2 = is_perfect(&vf.residue);
    tameness.push(Check {
        name: "(TF2) perfection of kappa(v)",
        status: tf2,
        detail: format!("kappa(v) = {}", vf.residue),
    });

    // --- Transfer hypotheses ------------------------------------------------

    let eq_char = match (vf.char_field, kappa_char) {
        (Char::Unknown, _) | (_, Char::Unknown) => Tri::Unknown,
        (a, b) if a == b => Tri::Yes,
        _ => Tri::No,
    };
    transfer.push(Check {
        name: "equal characteristic",
        status: eq_char,
        detail: format!(
            "char(K) = {} versus char(kappa(v)) = {}",
            vf.char_field, kappa_char
        ),
    });

    match group_theory_decidable(&vf.group) {
        Some(reason) => transfer.push(Check {
            name: "Th(Gamma_v) recursive",
            status: Tri::Yes,
            detail: reason.to_string(),
        }),
        None => transfer.push(Check {
            name: "Th(Gamma_v) recursive",
            status: Tri::Unknown,
            detail: format!(
                "no proof of decidability on file for Gamma_v = {}; refusing rather than guessing",
                vf.group
            ),
        }),
    }

    match field_theory_decidable(&vf.residue) {
        Some(reason) => transfer.push(Check {
            name: "Th(kappa(v)) recursive",
            status: Tri::Yes,
            detail: reason.to_string(),
        }),
        None => transfer.push(Check {
            name: "Th(kappa(v)) recursive",
            status: Tri::Unknown,
            detail: format!(
                "no proof of decidability on file for kappa(v) = {}; refusing rather than guessing",
                vf.residue
            ),
        }),
    }

    Certificate {
        field: vf.clone(),
        tameness,
        transfer,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{GroupExpr, Residue};
    use crate::sexp::parse_all;

    fn vf(src: &str) -> ValuedField {
        ValuedField::parse(&parse_all(src).unwrap()[0]).unwrap()
    }

    #[test]
    fn perfect_hull_five_is_certified() {
        let c = certify(&vf("(field Fp5 (Zinv 5))"));
        assert!(c.certified(), "blockers: {:?}", c.blockers());
        assert!(c.conclusion().is_some());
    }

    #[test]
    fn algebraic_closure_case_is_certified() {
        let c = certify(&vf("(field (ACF (p 5)) (Zinv 5))"));
        assert!(c.certified(), "blockers: {:?}", c.blockers());
    }

    #[test]
    fn characteristic_zero_hahn_field_is_certified() {
        // ACF_0 is perfect and has decidable theory, and Q is divisible, so
        // ACF_0((t^Q)) is tame with both residue and group theories recursive.
        let c = certify(&vf("(field (ACF 0) Q)"));
        assert!(c.certified(), "blockers: {:?}", c.blockers());
    }

    #[test]
    fn totally_ramified_finite_extension_is_certified() {
        // Degree 5 totally ramified extension of F_5((t^{Z[1/5]})). The value
        // group becomes Z[1/25], still 5-divisible, and the residue is unchanged.
        let c = certify(&vf("(finkt 5 1 (field Fp5 (Zinv 5)))"));
        assert!(c.certified(), "blockers: {:?}", c.blockers());
    }

    #[test]
    fn unramified_finite_extension_with_declared_residue_is_certified() {
        // Degree 5 unramified extension: e = 1, so the value group is unchanged
        // and the residue field is declared to be F_{5^5}, perfect and decidable.
        let c = certify(&vf("(finkt 1 5 (field Fp5 (Zinv 5)) (Fpn 5 5))"));
        assert!(c.certified(), "blockers: {:?}", c.blockers());
    }

    #[test]
    fn finite_extension_cannot_rescue_an_untame_base() {
        // F_p((t)) fails (TF1) because Z is not p-divisible, and a finite
        // extension at a prime other than p does not fix the value group.
        let c = certify(&vf("(finkt 3 1 (field Fp5 Z))"));
        assert_eq!(c.verdict(), Verdict::Refused);
        assert!(c.blockers().join(" ").contains("(TF1)"));
        // Ramifying at p does fix it: Z[1/3] is 5-divisible because 5 > 3.
        let d = certify(&vf("(finkt 5 1 (field Fp5 Z))"));
        assert!(d.certified(), "blockers: {:?}", d.blockers());
    }

    #[test]
    fn finite_extension_in_mixed_characteristic_stays_tame_only() {
        // Same arithmetic, but transfer is still unavailable.
        let c = certify(&vf("(finkt 5 1 (field 0 Fp5 (Zinv 5)))"));
        assert_eq!(c.verdict(), Verdict::TameOnly);
        // Only the equal-characteristic hypothesis may fail; the group and
        // residue theories are still established for the finite extension.
        assert!(c
            .transfer
            .iter()
            .any(|t| t.name == "equal characteristic" && t.status != Tri::Yes));
        assert!(c
            .transfer
            .iter()
            .filter(|t| t.name == "Th(Gamma_v) recursive")
            .all(|t| t.status == Tri::Yes));
    }

    #[test]
    fn finite_extension_of_an_undecided_residue_is_refused() {
        // A finite extension of a base whose residue is Q inherits the
        // undecidability, so no arithmetic can license it.
        let c = certify(&vf("(finkt 2 1 (field Q Q))"));
        assert!(!c.certified());
        assert!(c.blockers().join(" ").contains("kappa(v) = Q"));
    }

    #[test]
    fn q_is_perfect_but_its_theory_is_undecidable() {
        // Robinson proved Th(Q) is undecidable, so Q((t^Q)) is tame but the
        // transfer hypotheses fail. Tameness still holds, so this is the
        // "tameness but no transfer" case, not a tameness failure.
        let c = certify(&vf("(field Q Q)"));
        assert!(!c.certified(), "Q must never be certified");
        assert!(
            c.tameness.iter().all(|t| t.status == Tri::Yes),
            "tameness: {:?}",
            c.tameness
        );
        assert_eq!(c.verdict(), Verdict::TameOnly);
        let b = c.blockers().join(" ");
        assert!(b.contains("Th(kappa(v)) recursive NOT ESTABLISHED"), "blockers: {}", b);
        assert!(b.contains("kappa(v) = Q"), "blockers: {}", b);
    }

    #[test]
    fn fp_of_t_is_refused_on_tf1() {
        let c = certify(&vf("(field Fp5 Z)"));
        assert_eq!(c.verdict(), Verdict::Refused);
        let b = c.blockers().join(" ");
        assert!(b.contains("(TF1)"), "blockers: {}", b);
        assert!(c.conclusion().is_none());
    }

    #[test]
    fn rational_function_field_residue_fails_perfection_too() {
        // F_p(t)((t)): char(K) = char(kappa) = p and Gamma = Z, so (TF1)
        // fails; the residue F_p(t) is not perfect, so (TF2) fails as well.
        let c = certify(&vf("(field (rational Fp5) Z)"));
        assert_eq!(c.verdict(), Verdict::Refused);
        let names: Vec<&str> = c
            .tameness
            .iter()
            .filter(|x| x.status != Tri::Yes)
            .map(|x| x.name)
            .collect();
        assert_eq!(names.len(), 2, "expected both clauses to fail: {:?}", names);
    }

    #[test]
    fn mixed_characteristic_certifies_tameness_only() {
        // Models Q_p(p^{1/p^inf}): tame, but the transfer needs equal
        // characteristic, which fails here.
        let c = certify(&vf("(field 0 Fp5 (Zinv 5))"));
        assert_eq!(c.verdict(), Verdict::TameOnly);
        assert!(c.mixed_characteristic());
        assert!(c.conclusion().is_none());
        let note = c.transfer_note().unwrap();
        assert!(note.contains("mixed characteristic"), "note: {}", note);
        // Tameness itself is certified.
        assert!(c.tameness.iter().all(|x| x.status == Tri::Yes));
    }

    #[test]
    fn equal_characteristic_with_unknown_theories_is_tame_only() {
        // F_p((t)): tame? No, TF1 fails, so this must stay Refused.
        let c = certify(&vf("(field Fp5 Z)"));
        assert_eq!(c.verdict(), Verdict::Refused);
        // Perfect hull of F_p(t): tame shape, but the residue is a perfect
        // hull we cannot normalise, so the theory is unknown.
        let d = certify(&vf("(field (phull (p 5)) (Zinv 5))"));
        assert_eq!(d.verdict(), Verdict::TameOnly);
        assert!(!d.mixed_characteristic());
    }

    #[test]
    fn perfect_hull_of_power_series_field_is_certified() {
        // The perfect hull of F_p((t^Z)) is F_p((t^{Z[1/p]})). Written with the
        // perfect hull as the residue and the p-divisible closure of Z as the
        // value group, it is tame and its theory is decidable.
        let c = certify(&vf("(field (phull (hahn Fp5 Z)) (divby 5 Z))"));
        assert!(c.certified(), "blockers: {:?}", c.blockers());
        // Same field, spelled directly.
        let d = certify(&vf("(field Fp5 (Zinv 5))"));
        assert!(d.certified());
    }

    #[test]
    fn perfect_hull_of_rational_function_field_is_refused() {
        // The perfect hull of F_p(t) is not a power series field, so no
        // normal form is available and the tool must decline.
        let c = certify(&vf("(field (phull (hahn (rational Fp5) Z)) (divby 5 Z))"));
        assert!(!c.certified());
        assert!(c.blockers().join(" ").contains("Th(kappa(v)) recursive"));
    }

    #[test]
    fn opaque_input_is_never_certified() {
        for src in [
            "(field (opaque mystery) Z)",
            "(field Fp5 (opaque mystery))",
            "(field (opaque mystery) (opaque mystery))",
        ] {
            let c = certify(&vf(src));
            assert!(!c.certified(), "unexpectedly certified: {}", src);
        }
    }

    #[test]
    fn unknown_characteristic_blocks_the_transfer() {
        let c = certify(&vf("(field (opaque mystery) (Zinv 5))"));
        // Tameness cannot be judged, so we refuse rather than assume.
        assert_eq!(c.verdict(), Verdict::Refused);
        assert!(c.blockers().join(" ").contains("char(kappa(v)) is unknown"));
    }

    #[test]
    fn five_checks_always_present() {
        let c = certify(&vf("(field Fp5 Z)"));
        assert_eq!(c.all_checks().count(), 5);
        assert_eq!(c.tameness.len(), 2);
        assert_eq!(c.transfer.len(), 3);
    }

    #[test]
    fn residue_is_exactly_what_hahn_field_says() {
        let f = vf("(field Fp5 (Zinv 5))");
        assert_eq!(f.residue, Residue::Finite(5, 1));
        assert_eq!(f.group, GroupExpr::ZInv(5));
    }
}