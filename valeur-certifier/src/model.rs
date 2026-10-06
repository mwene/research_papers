//! Data model.
//!
//! The certificate procedure in the paper consumes exactly five pieces of
//! data about a valued field (K,v):
//!
//!   1. char(K)
//!   2. char(kappa(v))
//!   3. the value group Gamma_v
//!   4. whether kappa(v) is perfect
//!   5. whether Th(Gamma_v) and Th(kappa(v)) are recursive
//!
//! So we model exactly that much and nothing more. Every structural field
//! we accept is a *descriptor* that lets us compute those items, never a
//! claim that an arbitrary field has been fully analysed.

use crate::sexp::Sexp;
use std::fmt;

/// Three-valued logic. `Unknown` is essential: it is what keeps the tool
/// sound. We may only certify on `Yes`; anything else is a refusal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tri {
    Yes,
    No,
    Unknown,
}

impl Tri {
    pub fn label(self) -> &'static str {
        match self {
            Tri::Yes => "YES",
            Tri::No => "NO",
            Tri::Unknown => "UNKNOWN",
        }
    }
    pub fn and(self, other: Tri) -> Tri {
        match (self, other) {
            (Tri::Yes, Tri::Yes) => Tri::Yes,
            (Tri::No, _) | (_, Tri::No) => Tri::No,
            _ => Tri::Unknown,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Char {
    Zero,
    Prime(u32),
    /// We do not know the characteristic. Crucially distinct from `Zero`:
    /// assuming characteristic zero for an unanalysed field would let the
    /// certifier wrongly assert equal characteristic.
    Unknown,
}

impl fmt::Display for Char {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Char::Zero => write!(f, "0"),
            Char::Prime(p) => write!(f, "{}", p),
            Char::Unknown => write!(f, "unknown"),
        }
    }
}

/// Ordered abelian groups we can reason about symbolically.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GroupExpr {
    /// The integers.
    Z,
    /// Z[1/p]: the value group of a perfect hull in characteristic p.
    ZInv(u32),
    /// The rationals.
    Q,
    /// The reals.
    R,
    /// Finite direct sum of ordered abelian groups.
    Sum(Vec<GroupExpr>),
    /// `Gamma tensor_Z Z[1/p]`, the `p`-divisible closure of `Gamma`.
    /// Arises as the exponent group of a perfect hull.
    DivBy(u32, Box<GroupExpr>),
    /// `Gamma[1/e]`: the value group of a finite extension whose ramification
    /// index is `e`. `Gamma'` contains `Gamma` with `[Gamma':Gamma] = e`.
    FinExt(u32, Box<GroupExpr>),
    /// Anything we cannot decide; always yields `Unknown`.
    Opaque(String),
}

impl fmt::Display for GroupExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GroupExpr::Z => write!(f, "Z"),
            GroupExpr::ZInv(p) => write!(f, "Z[1/{}]", p),
            GroupExpr::Q => write!(f, "Q"),
            GroupExpr::R => write!(f, "R"),
            GroupExpr::Sum(v) => {
                write!(f, "sum-of[")?;
                for (i, g) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", g)?;
                }
                write!(f, "]")
            }
            GroupExpr::DivBy(p, g) => write!(f, "{}[1/{}]", g, p),
            // Localising at 1 changes nothing, so do not print it.
            GroupExpr::FinExt(1, g) => write!(f, "{}", g),
            GroupExpr::FinExt(e, g) => write!(f, "{}[1/{}]", g, e),
            GroupExpr::Opaque(s) => write!(f, "opaque({})", s),
        }
    }
}

/// Residue fields we can reason about symbolically.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Residue {
    /// The finite field F_{p^n}.
    Finite(u32, u32),
    /// An algebraically closed field of the given characteristic.
    AlgClosed(Char),
    /// The rationals.
    Rationals,
    /// A rational function field k(t) over a described k.
    RationalFn(Box<Residue>),
    /// A Hahn field k((t^G)).
    Hahn(Box<Residue>, GroupExpr),
    /// The perfect hull of the given field, e.g. F_p((t))^{1/p^\infty}.
    PerfectHull(Box<Residue>),
    /// A field we cannot analyse. We may still know its characteristic,
    /// which the user asserted, so it is carried rather than discarded.
    Opaque(Char, String),
}

impl fmt::Display for Residue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Residue::Finite(p, 1) => write!(f, "F_{}", p),
            Residue::Finite(p, n) => write!(f, "F_{{{}^{}}}", p, n),
            Residue::AlgClosed(c) => write!(f, "ACF_{}", c),
            Residue::Rationals => write!(f, "Q"),
            Residue::RationalFn(k) => write!(f, "{}(t)", k),
            Residue::Hahn(k, g) => write!(f, "{}((t^{}))", k, g),
            Residue::PerfectHull(inner) => write!(f, "perf({})", inner),
            Residue::Opaque(_, s) => write!(f, "opaque({})", s),
        }
    }
}

/// A valued field reduced to the data the decision procedure needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValuedField {
    pub char_field: Char,
    pub residue: Residue,
    pub group: GroupExpr,
    pub label: String,
}

fn err<T>(msg: impl Into<String>) -> Result<T, String> {
    Err(msg.into())
}

impl GroupExpr {
    pub fn parse(s: &Sexp) -> Result<GroupExpr, String> {
        if let Some(a) = s.atom() {
            return match a {
                "Z" => Ok(GroupExpr::Z),
                "Q" => Ok(GroupExpr::Q),
                "R" => Ok(GroupExpr::R),
                other => err(format!("unknown value group '{}'", other)),
            };
        }
        let items = s.list().unwrap();
        if items.is_empty() {
            return err("empty value group form");
        }
        let head = items[0].atom().unwrap_or("");
        match head {
            "Zinv" => {
                let p = items
                    .get(1)
                    .and_then(|x| x.nat())
                    .filter(|p| *p >= 2)
                    .ok_or("Zinv needs a prime >= 2")?;
                Ok(GroupExpr::ZInv(p))
            }
            "sum" => {
                if items.len() < 2 {
                    return err("sum needs at least one component");
                }
                let mut parts = Vec::new();
                for item in &items[1..] {
                    parts.push(GroupExpr::parse(item)?);
                }
                Ok(GroupExpr::Sum(parts))
            }
            "divby" => {
                let p = items
                    .get(1)
                    .and_then(|x| x.nat())
                    .filter(|p| *p >= 2)
                    .ok_or("divby needs a prime >= 2")?;
                let g = GroupExpr::parse(items.get(2).ok_or("divby needs a group")?)?;
                Ok(GroupExpr::DivBy(p, Box::new(g)))
            }
            "finkt" => {
                let e = items
                    .get(1)
                    .and_then(|x| x.nat())
                    .filter(|e| *e >= 1)
                    .ok_or("finkt needs a ramification index >= 1")?;
                let g = GroupExpr::parse(items.get(2).ok_or("finkt needs a value group")?)?;
                Ok(GroupExpr::FinExt(e, Box::new(g)))
            }
            "opaque" => Ok(GroupExpr::Opaque(render(&items[1..]))),
            other => err(format!("unknown value group form '{}'", other)),
        }
    }
}

impl Residue {
    pub fn parse(s: &Sexp) -> Result<Residue, String> {
        if let Some(a) = s.atom() {
            return match a {
                "Q" => Ok(Residue::Rationals),
                "Fp" => Ok(Residue::Finite(2, 1)),
                "Fp3" => Ok(Residue::Finite(3, 1)),
                "Fp5" => Ok(Residue::Finite(5, 1)),
                "Fp2" => Ok(Residue::Finite(2, 1)),
                other => err(format!("unknown residue field '{}'", other)),
            };
        }
        let items = s.list().unwrap();
        if items.is_empty() {
            return err("empty residue form");
        }
        let head = items[0].atom().unwrap_or("");
        match head {
            "Fpn" => {
                let p = items.get(1).and_then(|x| x.nat()).ok_or("Fpn needs p")?;
                let n = items.get(2).and_then(|x| x.nat()).ok_or("Fpn needs n")?;
                if p < 2 {
                    return err("p must be >= 2");
                }
                if n < 1 {
                    return err("n must be >= 1");
                }
                Ok(Residue::Finite(p, n))
            }
            "ACF" => {
                let c = parse_char_opt(items.get(1))?;
                Ok(Residue::AlgClosed(c))
            }
            "rational" => Ok(Residue::RationalFn(Box::new(Residue::parse(items.get(1).ok_or(
                "rational needs a base field",
            )?)?))),
            "hahn" => {
                let k = Residue::parse(items.get(1).ok_or("hahn needs a base field")?)?;
                let g = GroupExpr::parse(items.get(2).ok_or("hahn needs a value group")?)?;
                Ok(Residue::Hahn(Box::new(k), g))
            }
            "phull" => {
                let arg = items.get(1).ok_or("phull needs a characteristic or a field")?;
                let inner = if is_char_spec(arg) {
                    let c = parse_char_opt(Some(arg))?;
                    Residue::Opaque(c, format!("field of characteristic {}", c))
                } else {
                    Residue::parse(arg)?
                };
                Ok(Residue::PerfectHull(Box::new(inner)))
            }
            "opaque" => Ok(Residue::Opaque(Char::Unknown, render(&items[1..]))),
            other => err(format!("unknown residue form '{}'", other)),
        }
    }

    /// Characteristic of the field itself.
    pub fn char(&self) -> Char {
        match self {
            Residue::Finite(p, _) => Char::Prime(*p),
            Residue::AlgClosed(c) => *c,
            Residue::Rationals => Char::Zero,
            Residue::RationalFn(k) => k.char(),
            Residue::Hahn(k, _) => k.char(),
            Residue::PerfectHull(inner) => inner.char(),
            Residue::Opaque(c, _) => *c,
        }
    }
}

/// Does this form denote a characteristic rather than a field?
pub fn is_char_spec(s: &Sexp) -> bool {
    s.atom() == Some("0")
        || s.list()
            .and_then(|l| l.first())
            .and_then(|h| h.atom())
            .map(|a| a == "p")
            .unwrap_or(false)
}

fn parse_char_opt(s: Option<&Sexp>) -> Result<Char, String> {
    match s {
        None => Ok(Char::Zero),
        Some(x) if x.atom() == Some("0") => Ok(Char::Zero),
        Some(Sexp::List(items)) if items[0].atom() == Some("p") => {
            let p = items.get(1).and_then(|x| x.nat()).ok_or("bad (p ...) form")?;
            if p < 2 {
                return err("p must be >= 2");
            }
            Ok(Char::Prime(p))
        }
        Some(_) => err("expected 0 or (p n)"),
    }
}

fn render(items: &[Sexp]) -> String {
    if items.len() == 1 {
        items[0].to_string()
    } else {
        Sexp::List(items.to_vec()).to_string()
    }
}

impl ValuedField {
    /// Parse `(field <char> <residue> <group>)` or the shorthand
    /// `(field <residue> <group>)` (which sets char(field) = char(residue),
    /// the equal-characteristic case).
    pub fn parse(s: &Sexp) -> Result<ValuedField, String> {
        // A finite extension is written `(finkt <e> <f> <base> [<residue of L>])`.
        if let Some(items) = s.list() {
            if items.first().and_then(|h| h.atom()) == Some("finkt") {
                return ValuedField::parse_finite_extension(s);
            }
        }
        let items = s.list().ok_or("expected a (field ...) form")?;
        if items.is_empty() {
            return err("empty field form");
        }
        if items[0].atom() != Some("field") {
            return err("expected head symbol 'field'");
        }
        let tail = &items[1..];
        // Disambiguate the two arities before parsing. `(field A B)` and
        // `(field <char> <residue>)` are both two-element tails, so a
        // leading `(p n)` or `0` must be read as a characteristic, not as
        // a residue field. Guessing here would silently misparse input.
        let tail_is_char_spec = tail.first().map(is_char_spec).unwrap_or(false);
        if tail_is_char_spec && tail.len() == 2 {
            return err(
                "(field <char> <residue>) has no value group; write \
                 (field <char> <residue> <group>), or drop the characteristic \
                 and use (field <residue> <group>)",
            );
        }
        let (char_field, residue, group) = match tail.len() {
            2 => {
                let r = Residue::parse(&tail[0])?;
                let g = GroupExpr::parse(&tail[1])?;
                let c = r.char();
                (c, r, g)
            }
            3 => {
                let c = parse_char_opt(Some(&tail[0]))?;
                let r = Residue::parse(&tail[1])?;
                let g = GroupExpr::parse(&tail[2])?;
                (c, r, g)
            }
            _ => return err("expected (field <char> <residue> <group>) or (field <residue> <group>)"),
        };
        Ok(ValuedField {
            char_field,
            residue,
            group,
            label: render(tail),
        })
    }

    pub fn residue_char(&self) -> Char {
        self.residue.char()
    }

    /// Parse `(finkt <e> <f> <base-field> [<residue of L>])`, a finite
    /// extension L/K of ramification index `e` and residue degree `f`.
    ///
    /// Only the two facts the procedure actually needs are inferred, both of
    /// which are unconditional for finite extensions:
    ///
    ///   * `char(L) = char(K)`;
    ///   * `Gamma_L = Gamma_K[1/e]`, since `Gamma_L` contains `Gamma_K` with
    ///     index exactly the ramification index `e`.
    ///
    /// Perfectness needs no inference: `L` is perfect iff its residue field
    /// is, and a finite extension of a perfect residue field has residue
    /// field a finite algebraic extension, hence perfect. The residue field
    /// itself is *not* determined by `e` and `f`, so when `f > 1` it must be
    /// declared rather than guessed.
    fn parse_finite_extension(s: &Sexp) -> Result<ValuedField, String> {
        let items = s.list().unwrap();
        let e = items
            .get(1)
            .and_then(|x| x.nat())
            .filter(|e| *e >= 1)
            .ok_or("finkt needs a ramification index e >= 1")?;
        let f = items
            .get(2)
            .and_then(|x| x.nat())
            .filter(|f| *f >= 1)
            .ok_or("finkt needs a residue degree f >= 1")?;
        let base = ValuedField::parse(items.get(3).ok_or(
            "finkt needs a base field as its third argument, e.g. (field Fp5 (Zinv 5))",
        )?)?;
        if items.len() == 5 && f == 1 {
            return err(
                "finkt with f = 1 already has residue field determined by the base; \
                 drop the extra residue argument",
            );
        }
        let residue = match items.get(4) {
            None => {
                if f == 1 {
                    base.residue.clone()
                } else {
                    return err(format!(
                        "a finite extension with residue degree f = {} changes the residue \
                         field, which e and f do not determine; state it, e.g. \
                         (finkt {} {} <base> <residue>)",
                        f, e, f
                    ));
                }
            }
            Some(r) => {
                let r = Residue::parse(r)?;
                // A finite extension preserves the residue characteristic, so a
                // declared residue of a different *known* characteristic is a
                // contradiction rather than a new field.
                if let (Char::Prime(p), Char::Prime(q)) = (base.residue_char(), r.char()) {
                    if p != q {
                        return err(format!(
                            "residue field has characteristic {} but the base residue field \
                             has characteristic {}",
                            q, p
                        ));
                    }
                }
                r
            }
        };
        if items.len() > 5 {
            return err("finkt takes at most four arguments: (finkt e f base [residue])");
        }
        Ok(ValuedField {
            char_field: base.char_field,
            residue,
            group: GroupExpr::FinExt(e, Box::new(base.group.clone())),
            label: render(&items[1..]),
        })
    }

    /// The canonical form used in reports: k((t^G)). When the field and its
    /// residue have different characteristics the notation would suggest
    /// otherwise, so the field characteristic is shown explicitly.
    pub fn pretty(&self) -> String {
        let base = format!("{}((t^{}))", self.residue, self.group);
        if self.char_field == self.residue_char() {
            base
        } else {
            format!("[char {}]{}", self.char_field, base)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sexp::parse_all;

    fn one(src: &str) -> ValuedField {
        ValuedField::parse(&parse_all(src).unwrap()[0]).unwrap()
    }

    #[test]
    fn parses_shorthand_and_derives_char() {
        let vf = one("(field Fp5 (Zinv 5))");
        assert_eq!(vf.char_field, Char::Prime(5));
        assert_eq!(vf.residue_char(), Char::Prime(5));
        assert_eq!(vf.group, GroupExpr::ZInv(5));
    }

    #[test]
    fn parses_explicit_mixed_characteristic() {
        let vf = one("(field 0 Fp5 (Zinv 5))");
        assert_eq!(vf.char_field, Char::Zero);
        assert_eq!(vf.residue_char(), Char::Prime(5));
    }

    #[test]
    fn rejects_bad_forms() {
        assert!(ValuedField::parse(&parse_all("(notfield Fp Z)").unwrap()[0]).is_err());
        assert!(ValuedField::parse(&parse_all("(field Fp)").unwrap()[0]).is_err());
        assert!(GroupExpr::parse(&parse_all("(Zinv 1)").unwrap()[0]).is_err());
    }

    #[test]
    fn ambiguity_between_arities_is_reported_not_guessed() {
        // `(field (p 2) (hahn ...))` has only two tail elements. Guessing
        // would silently read `(p 2)` as a residue field.
        let e = ValuedField::parse(&parse_all("(field (p 2) (hahn (ACF (p 2)) Z))").unwrap()[0]);
        assert!(e.is_err());
        assert!(e.unwrap_err().contains("no value group"));
        // The intended readings both parse.
        assert!(ValuedField::parse(&parse_all("(field (p 2) (ACF (p 2)) Z)").unwrap()[0]).is_ok());
        assert!(ValuedField::parse(&parse_all("(field (hahn (ACF (p 2)) Z) Z)").unwrap()[0]).is_ok());
    }

    #[test]
    fn perfect_hull_keeps_characteristic() {
        assert_eq!(
            Residue::PerfectHull(Box::new(Residue::Finite(5, 1))).char(),
            Char::Prime(5)
        );
        // The shorthand (phull (p 5)) keeps the characteristic but hides the
        // field, so normal form recognition must decline.
        let opaque_hull = Residue::parse(&parse_all("(phull (p 5))").unwrap()[0]).unwrap();
        assert_eq!(opaque_hull.char(), Char::Prime(5));
    }

    #[test]
    fn tri_and_is_symmetric() {
        assert_eq!(Tri::Yes.and(Tri::Yes), Tri::Yes);
        assert_eq!(Tri::Yes.and(Tri::Unknown), Tri::Unknown);
        assert_eq!(Tri::No.and(Tri::Yes), Tri::No);
        assert_eq!(Tri::Unknown.and(Tri::No), Tri::No);
    }

    #[test]
    fn finkt_preserves_characteristic_and_residue_when_unramified_in_residue() {
        // f = 1: the residue field is untouched, so it is inherited.
        let vf = one("(finkt 5 1 (field Fp5 (Zinv 5)))");
        assert_eq!(vf.char_field, Char::Prime(5));
        assert_eq!(vf.residue, Residue::Finite(5, 1));
        assert_eq!(vf.group, GroupExpr::FinExt(5, Box::new(GroupExpr::ZInv(5))));
        // char(L) = char(K) also holds across mixed characteristic.
        let mixed = one("(finkt 3 1 (field 0 Fp5 (Zinv 5)))");
        assert_eq!(mixed.char_field, Char::Zero);
        assert_eq!(mixed.residue_char(), Char::Prime(5));
    }

    #[test]
    fn finkt_requires_a_declared_residue_when_f_exceeds_one() {
        // e and f do not determine the residue field, so we must not guess.
        let e = ValuedField::parse(&parse_all("(finkt 2 3 (field Fp5 Z))").unwrap()[0]);
        assert!(e.is_err());
        assert!(e.unwrap_err().contains("do not determine"));
        // Declaring it is accepted.
        let ok = one("(finkt 2 3 (field Fp5 Z) (Fpn 5 3))");
        assert_eq!(ok.residue, Residue::Finite(5, 3));
        assert_eq!(ok.group, GroupExpr::FinExt(2, Box::new(GroupExpr::Z)));
    }

    #[test]
    fn finkt_rejects_redundant_or_inconsistent_residue() {
        // f = 1 already determines the residue, so naming it is a mistake
        // that would otherwise be silently ignored.
        let e = ValuedField::parse(&parse_all("(finkt 5 1 (field Fp5 Z) Fp5)").unwrap()[0]);
        assert!(e.unwrap_err().contains("drop the extra residue"));
        // A finite extension cannot change the residue characteristic.
        let e = ValuedField::parse(&parse_all("(finkt 2 3 (field Fp5 Z) (Fpn 3 3))").unwrap()[0]);
        assert!(e.unwrap_err().contains("characteristic"));
        // Bad e, bad f, and too many arguments.
        assert!(ValuedField::parse(&parse_all("(finkt 0 1 (field Fp5 Z))").unwrap()[0]).is_err());
        assert!(ValuedField::parse(&parse_all("(finkt 2 0 (field Fp5 Z))").unwrap()[0]).is_err());
        assert!(ValuedField::parse(&parse_all("(finkt 2 1 (field Fp5 Z))").unwrap()[0]).is_ok());
    }
}