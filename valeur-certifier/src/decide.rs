//! The mathematical engine.
//!
//! Soundness rules obeyed throughout this module:
//!
//!   * We return `Tri::Yes` only when we can justify it.
//!   * Every unanalysable case degrades to `Tri::Unknown`, never to `Tri::No`.
//!   * The theory registry is a *whitelist*. Absence of an entry means
//!     "no proof known", which yields a refusal, not a refutation.

use crate::model::{Char, GroupExpr, Residue, Tri};

/// Is `g` `p`-divisible, i.e. is `g = p g'` solvable for every element?
pub fn p_divisible(g: &GroupExpr, p: u32) -> Tri {
    if p < 2 {
        return Tri::Yes;
    }
    match g {
        GroupExpr::Q | GroupExpr::R => Tri::Yes,
        GroupExpr::Z => Tri::No,
        GroupExpr::ZInv(q) => {
            if *q == p {
                Tri::Yes
            } else {
                Tri::No
            }
        }
        // A direct sum is p-divisible iff each component is: solving
        // p(g1,g2) = (x,y) forces g1 to solve p g1 = x separately.
        GroupExpr::Sum(parts) => parts.iter().fold(Tri::Yes, |acc, x| acc.and(p_divisible(x, p))),
        // Gamma tensor_Z Z[1/q] is q-divisible by construction. For p != q
        // inverting q does not help with p, so we defer to Gamma. A `No`
        // there can only cause a refusal, never a false certification.
        GroupExpr::DivBy(q, inner) => {
            if *q == p {
                Tri::Yes
            } else {
                p_divisible(inner, p)
            }
        }
        // Value group Gamma' of a finite extension of ramification index e.
        // Gamma_L contains Gamma_K with index exactly e, so Gamma' = Gamma[1/e].
        GroupExpr::FinExt(e, inner) => {
            if p <= *e && e % p == 0 {
                // Gamma' contains 1/e, and if p | e then 1/e = p * 1/(pe), so
                // Gamma' is p-divisible. Sound for every Gamma we accept.
                Tri::Yes
            } else {
                // If Gamma is p-divisible then so is Gamma[1/e]: inverting e
                // commutes with p-surjectivity. Any `No` here can only cause a
                // refusal, so this stays sound in the safe direction.
                p_divisible(inner, p)
            }
        }
        GroupExpr::Opaque(_) => Tri::Unknown,
    }
}

/// Divisible by every prime.
pub fn divisible(g: &GroupExpr) -> Tri {
    match g {
        GroupExpr::Q | GroupExpr::R => Tri::Yes,
        GroupExpr::Z | GroupExpr::ZInv(_) => Tri::No,
        GroupExpr::Sum(parts) => parts.iter().fold(Tri::Yes, |acc, x| acc.and(divisible(x))),
        GroupExpr::DivBy(_, inner) => divisible(inner),
        GroupExpr::FinExt(_, inner) => divisible(inner),
        GroupExpr::Opaque(_) => Tri::Unknown,
    }
}

/// Is the residue field perfect (formally: does every polynomial
/// `X^n - a` with `a != 0` have a root)?
pub fn is_perfect(k: &Residue) -> Tri {
    match k {
        // Finite fields are perfect; their algebraic closure is perfect.
        Residue::Finite(_, _) => Tri::Yes,
        Residue::AlgClosed(_) => Tri::Yes,
        // Q is perfect in characteristic zero.
        Residue::Rationals => Tri::Yes,
        // Rational function fields. Every field of characteristic zero is
        // perfect, since all irreducibles are separable. In characteristic
        // p they are never perfect: (k(t))^p = k^p(t^p) does not contain t,
        // so t has no p-th root in k(t).
        Residue::RationalFn(base) => match base.char() {
            Char::Zero => Tri::Yes,
            // In characteristic p, t is never a p-th power in k(t). In
            // characteristic zero all fields are perfect.
            Char::Prime(_) => Tri::No,
            Char::Unknown => Tri::Unknown,
        },
        // k((t^G)) is perfect iff k is perfect and the exponents needed to
        // take roots of t are present in G: p-divisible in char p,
        // divisible in char zero.
        Residue::Hahn(base, g) => match base.char() {
            Char::Prime(p) => is_perfect(base).and(p_divisible(g, p)),
            Char::Zero => is_perfect(base).and(divisible(g)),
            Char::Unknown => Tri::Unknown,
        },
        // Perfect hull by definition.
        Residue::PerfectHull(_) => Tri::Yes,
        Residue::Opaque(..) => Tri::Unknown,
    }
}

/// Whitelist of ordered abelian groups whose complete theory in the
/// language of ordered groups is recursive.
pub fn group_theory_decidable(g: &GroupExpr) -> Option<&'static str> {
    match g {
        GroupExpr::Z => Some("Th(Z) in ordered groups is decidable"),
        GroupExpr::Q => Some("Th(Q) in ordered groups is decidable"),
        GroupExpr::R => Some("Th(R) in ordered groups is decidable"),
        GroupExpr::ZInv(_p) => Some(
            "p is the least positive element of pZ, hence definable, so \
             Th(Z[1/p]) reduces to Presburger arithmetic",
        ),
        GroupExpr::Sum(parts) => {
            let mut reason = String::new();
            for x in parts {
                match group_theory_decidable(x) {
                    Some(r) => {
                        if !reason.is_empty() {
                            reason.push_str("; ");
                        }
                        reason.push_str(r);
                    }
                    None => return None,
                }
            }
            if reason.is_empty() {
                None
            } else {
                Some(Box::leak(format!("finite sum of decidable groups: {}", reason).into_boxed_str()))
            }
        }
        // Z[1/p] is Z with p definable; taking a further [1/q]-closure gives
        // Z[1/pq], again Presburger with a definable constant. Rationals and
        // reals are unchanged. Finite sums are handled componentwise.
        GroupExpr::DivBy(_, inner) => group_theory_decidable(inner),
        // Z[1/q] localized at 1/e is Z[1/(q*e)], again Presburger with a
        // definable constant; Q and R are unchanged; sums stay componentwise.
        GroupExpr::FinExt(_, inner) => group_theory_decidable(inner),
        GroupExpr::Opaque(_) => None,
    }
}

/// Whitelist of fields whose complete theory in the language of rings is
/// recursive.
///
/// Deliberately conservative. Two absences are theorems rather than gaps:
///
///   * `Q` and every number field are **undecidable** (Robinson), and every
///     global field, including `F_q(t)`, is undecidable (Rumely). So no
///     rational function field over a global field may be listed.
///   * Rational function fields in characteristic p are also *not perfect*,
///     since `t` is never a p-th power in `k(t)`, so they can never be the
///     residue of a tame positive-characteristic Hahn field regardless.
pub fn field_theory_decidable(k: &Residue) -> Option<&'static str> {
    let f = k;
    match f {
        Residue::Finite(_, _) => Some("finite fields are decidable (Ax)"),
        Residue::AlgClosed(c) => Some(match c {
            Char::Zero => "ACF_0 is decidable (Tarski)",
            Char::Prime(_) => "ACF_p is decidable (Tarski)",
            // Cannot happen from the parser, but keep the match total.
            Char::Unknown => "ACF of unknown characteristic",
        }),
        // Hahn fields. Relative decidability (Kuhlmann, Thm 1) needs the
        // field to be tame, so we gate on the conditions we can actually
        // check: a perfect residue, and exponents supplying the needed roots.
        //
        // This gate is load-bearing. Without it the whitelist would certify
        // Th(F_p((t))), which is exactly the open case, because F_p and Z
        // are individually decidable. Tameness of a power series field is
        // not automatic in the value group.
        Residue::Hahn(base, g) => {
            let tame_shape = match base.char() {
                Char::Prime(p) => {
                    is_perfect(base) == Tri::Yes && p_divisible(g, p) == Tri::Yes
                }
                Char::Zero => is_perfect(base) == Tri::Yes && divisible(g) == Tri::Yes,
                // An unknown characteristic cannot establish tameness.
                Char::Unknown => false,
            };
            if !tame_shape {
                return None;
            }
            let b = field_theory_decidable(base)?;
            let gr = group_theory_decidable(g)?;
            Some(Box::leak(
                format!(
                    "relative decidability for the tame field {} [base: {}; group: {}];                      in equal positive characteristic Lisinski's Thm 4 gives the                      stronger L_val(t)-decidability",
                    Residue::Hahn(base.clone(), g.clone()),
                    b,
                    gr
                )
                .into_boxed_str(),
            ))
        }
        // Perfect hulls are decidable only when they reduce to a power series
        // field over a perfect residue; otherwise we decline.
        Residue::PerfectHull(_) => {
            let nf = normalise_perfect_hull(f)?;
            let inner_reason = field_theory_decidable(&nf)?;
            Some(Box::leak(
                format!("perfect hull reduced to {} [{}]", nf, inner_reason).into_boxed_str(),
            ))
        }
        _ => None,
    }
}

/// Recognise the perfect hull of a power series field over a perfect residue.
///
/// For perfect `k` of characteristic `p` the perfect hull of `k((t^G))` is the
/// power series field with exponent group `G tensor_Z Z[1/p]`. Returns `None`
/// whenever the input is not of that shape, which is the common case: the
/// perfect hull of `F_p(t)` is not a power series field at all.
pub fn normalise_perfect_hull(f: &Residue) -> Option<Residue> {
    match f {
        Residue::PerfectHull(inner) => match &**inner {
            Residue::Hahn(k, g) => match k.char() {
                Char::Prime(p) if is_perfect(k) == Tri::Yes => Some(Residue::Hahn(
                    k.clone(),
                    GroupExpr::DivBy(p, Box::new(g.clone())),
                )),
                _ => None,
            },
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p_divisibility_basics() {
        assert_eq!(p_divisible(&GroupExpr::Z, 5), Tri::No);
        assert_eq!(p_divisible(&GroupExpr::ZInv(5), 5), Tri::Yes);
        assert_eq!(p_divisible(&GroupExpr::ZInv(3), 5), Tri::No);
        assert_eq!(p_divisible(&GroupExpr::Q, 5), Tri::Yes);
        assert_eq!(p_divisible(&GroupExpr::R, 5), Tri::Yes);
    }

    #[test]
    fn p_divisibility_is_componentwise_on_sums() {
        let s = GroupExpr::Sum(vec![GroupExpr::Q, GroupExpr::Z]);
        assert_eq!(p_divisible(&s, 5), Tri::No);
        let s2 = GroupExpr::Sum(vec![GroupExpr::Q, GroupExpr::ZInv(5)]);
        assert_eq!(p_divisible(&s2, 5), Tri::Yes);
        let s3 = GroupExpr::Sum(vec![GroupExpr::Q, GroupExpr::ZInv(3)]);
        assert_eq!(p_divisible(&s3, 5), Tri::No);
    }

    #[test]
    fn opaque_never_confirms() {
        assert_eq!(p_divisible(&GroupExpr::Opaque("Gamma".into()), 5), Tri::Unknown);
        assert_eq!(is_perfect(&Residue::Opaque(Char::Unknown, "k".into())), Tri::Unknown);
    }

    #[test]
    fn perfectness_of_hahn_fields() {
        // F_p((t)): F_p is perfect but Z is not p-divisible, so t has no
        // p-th root. Not perfect.
        let k = Residue::Hahn(Box::new(Residue::Finite(5, 1)), GroupExpr::Z);
        assert_eq!(is_perfect(&k), Tri::No);
        // F_p((t))^{1/p^\infty} = F_p((t^{Z[1/p]})): perfect.
        let h = Residue::Hahn(Box::new(Residue::Finite(5, 1)), GroupExpr::ZInv(5));
        assert_eq!(is_perfect(&h), Tri::Yes);
        // Overline F_p((t^{1/p^\infty})): perfect.
        let a = Residue::Hahn(Box::new(Residue::AlgClosed(Char::Prime(5))), GroupExpr::ZInv(5));
        assert_eq!(is_perfect(&a), Tri::Yes);
        // Q((t^Q)): perfect.
        let q = Residue::Hahn(Box::new(Residue::Rationals), GroupExpr::Q);
        assert_eq!(is_perfect(&q), Tri::Yes);
        // Q((t^Z)): Q is perfect but Z is not divisible, so not perfect.
        let q2 = Residue::Hahn(Box::new(Residue::Rationals), GroupExpr::Z);
        assert_eq!(is_perfect(&q2), Tri::No);
    }

    #[test]
    fn rational_function_fields_are_perfect_only_in_characteristic_zero() {
        // Every field of characteristic zero is perfect.
        let zero = Residue::RationalFn(Box::new(Residue::Rationals));
        assert_eq!(is_perfect(&zero), Tri::Yes);
        // In characteristic p they never are: (F_p(t))^p = F_p(t^p) omits t.
        let pos = Residue::RationalFn(Box::new(Residue::Finite(5, 1)));
        assert_eq!(is_perfect(&pos), Tri::No);
        // Even over an algebraically closed residue.
        let over_alg = Residue::RationalFn(Box::new(Residue::AlgClosed(Char::Prime(5))));
        assert_eq!(is_perfect(&over_alg), Tri::No);
        // Even over a perfect residue in characteristic zero-... which is Yes.
        assert_eq!(is_perfect(&Residue::Rationals), Tri::Yes);
    }

    #[test]
    fn hahn_gate_refuses_the_open_case() {
        // F_p((t)) is NOT tame, so relative decidability does not apply and
        // its theory must not be certified. This is the paper's open case.
        let k = Residue::Hahn(Box::new(Residue::Finite(5, 1)), GroupExpr::Z);
        assert!(field_theory_decidable(&k).is_none());
        // The perfect hull of the same field is tame, hence decidable.
        let hull = Residue::PerfectHull(Box::new(k));
        assert!(field_theory_decidable(&hull).is_some());
    }

    #[test]
    fn perfect_hull_normalises_over_perfect_residue() {
        use crate::decide::normalise_perfect_hull;
        let k = Residue::Hahn(Box::new(Residue::Finite(5, 1)), GroupExpr::Z);
        let nf = normalise_perfect_hull(&Residue::PerfectHull(Box::new(k))).unwrap();
        assert_eq!(
            nf,
            Residue::Hahn(Box::new(Residue::Finite(5, 1)), GroupExpr::DivBy(5, Box::new(GroupExpr::Z)))
        );
        // The perfect hull of F_p(t) is not a power series field: decline.
        let k2 = Residue::Hahn(Box::new(Residue::RationalFn(Box::new(Residue::Finite(5, 1)))), GroupExpr::Z);
        assert!(normalise_perfect_hull(&Residue::PerfectHull(Box::new(k2))).is_none());
        // The opaque shorthand carries only a characteristic: decline.
        assert!(normalise_perfect_hull(&Residue::PerfectHull(Box::new(Residue::Opaque(Char::Unknown, "x".into())))).is_none());
    }

    #[test]
    fn divisible_closure_is_p_divisible_and_keeps_theory() {
        // Gamma[1/p] is p-divisible by construction.
        let d = GroupExpr::DivBy(5, Box::new(GroupExpr::Z));
        assert_eq!(p_divisible(&d, 5), Tri::Yes);
        // Z[1/5] is not 3-divisible.
        assert_eq!(p_divisible(&d, 3), Tri::No);
        // and is not divisible by every prime.
        assert_eq!(divisible(&d), Tri::No);
        // Its ordered-group theory is still decidable (Presburger).
        assert!(group_theory_decidable(&d).is_some());
        // Q[1/p] = Q stays divisible.
        assert_eq!(divisible(&GroupExpr::DivBy(5, Box::new(GroupExpr::Q))), Tri::Yes);
    }

    #[test]
    fn characteristic_zero_hahn_field_is_certified_but_open_case_is_not() {
        // ACF_0((t^Q)): tame, relative decidability applies.
        assert!(field_theory_decidable(&Residue::Hahn(
            Box::new(Residue::AlgClosed(Char::Zero)),
            GroupExpr::Q
        ))
        .is_some());
        // ACF_0((t^Z)): char zero but Z is not divisible, so not tame.
        assert!(field_theory_decidable(&Residue::Hahn(
            Box::new(Residue::AlgClosed(Char::Zero)),
            GroupExpr::Z
        ))
        .is_none());
    }

    #[test]
    fn registry_is_conservative() {
        assert!(field_theory_decidable(&Residue::PerfectHull(Box::new(Residue::Opaque(Char::Unknown, "mystery".into()))))
        .is_none());
        assert!(field_theory_decidable(&Residue::RationalFn(Box::new(Residue::Finite(5, 1)))).is_none());
        assert!(group_theory_decidable(&GroupExpr::Opaque("x".into())).is_none());
    }

    #[test]
    fn registry_covers_paper_examples() {
        assert!(field_theory_decidable(&Residue::Finite(5, 1)).is_some());
        assert!(group_theory_decidable(&GroupExpr::ZInv(5)).is_some());
    }
}
#[cfg(test)]
mod arithmetic_tests {
    use super::*;

    fn g(src: &str) -> GroupExpr {
        GroupExpr::parse(&crate::sexp::parse_all(src).unwrap()[0]).unwrap()
    }

    #[test]
    fn divby_only_helps_its_own_prime() {
        // Z[1/3] localized again at 3 is still Z[1/3], which is NOT 5-divisible.
        // Passing the localized prime instead of the queried prime would have
        // wrongly answered YES here.
        assert_eq!(p_divisible(&g("(divby 3 (Zinv 3))"), 5), Tri::No);
        // The queried prime is still answered when it is the localising one.
        assert_eq!(p_divisible(&g("(divby 5 Z)"), 5), Tri::Yes);
        assert_eq!(p_divisible(&g("(divby 5 Z)"), 3), Tri::No);
    }

    #[test]
    fn finite_extension_localises_the_value_group() {
        // Gamma[1/e] is p-divisible when p | e, whatever Gamma is.
        assert_eq!(p_divisible(&g("(finkt 5 Z)"), 5), Tri::Yes);
        assert_eq!(p_divisible(&g("(finkt 6 Z)"), 5), Tri::No);
        // ... and inherits p-divisibility from Gamma otherwise.
        assert_eq!(p_divisible(&g("(finkt 3 (Zinv 5))"), 5), Tri::Yes);
        assert_eq!(p_divisible(&g("(finkt 2 (Zinv 5))"), 5), Tri::Yes);
        // Z[1/5] is never 3-divisible, and localising at 2 does not help.
        assert_eq!(p_divisible(&g("(finkt 2 (Zinv 5))"), 3), Tri::No);
    }

    #[test]
    fn finite_extension_preserves_group_decidability() {
        // Z[1/5][1/e] = Z[1/(5e)] is Presburger with a definable constant.
        assert!(group_theory_decidable(&g("(finkt 7 (Zinv 5))")).is_some());
        assert!(group_theory_decidable(&g("(finkt 7 Q)")).is_some());
        assert!(group_theory_decidable(&g("(finkt 7 (opaque mystery))")).is_none());
    }

    #[test]
    fn finite_extension_does_not_invent_divisibility() {
        // Gamma[1/7] is still not divisible, so a finite extension of a
        // non-tame base stays non-tame in characteristic zero.
        assert_eq!(divisible(&g("(finkt 7 Z)")), Tri::No);
        assert_eq!(divisible(&g("(finkt 7 (Zinv 5))")), Tri::No);
    }

    #[test]
    fn q_and_global_rational_function_fields_are_not_listed() {
        // Robinson: Th(Q) is undecidable. Rumely: every global field, hence
        // F_p(t) and F_q(t), is undecidable. Neither may be listed.
        assert_eq!(field_theory_decidable(&Residue::Rationals), None);
        for base in [Residue::Rationals, Residue::Finite(5, 1), Residue::Finite(2, 3)] {
            assert_eq!(
                field_theory_decidable(&Residue::RationalFn(Box::new(base))),
                None,
                "rational function field must not be listed"
            );
        }
        // Being absent for that reason is consistent with Q being perfect.
        assert_eq!(is_perfect(&Residue::Rationals), Tri::Yes);
    }

    #[test]
    fn finite_extension_of_finite_field_residue_is_listed() {
        // F_{p^f} over F_p: decidable and perfect, so a finite extension of
        // the residue causes no obstruction once the residue is declared.
        assert!(field_theory_decidable(&Residue::Finite(5, 5)).is_some());
        assert_eq!(is_perfect(&Residue::Finite(5, 5)), Tri::Yes);
    }
}
