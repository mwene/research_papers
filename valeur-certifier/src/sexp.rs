//! Minimal S-expression reader. Zero dependencies on purpose: the whole
//! point of this tool is that its verdict is auditable, and an auditable
//! tool should have a small, inspectable core.

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Sexp {
    Atom(String),
    List(Vec<Sexp>),
}

impl Sexp {
    pub fn atom(&self) -> Option<&str> {
        match self {
            Sexp::Atom(s) => Some(s),
            Sexp::List(_) => None,
        }
    }

    pub fn list(&self) -> Option<&[Sexp]> {
        match self {
            Sexp::List(v) => Some(v),
            Sexp::Atom(_) => None,
        }
    }

    pub fn nat(&self) -> Option<u32> {
        let s = self.atom()?;
        if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        s.parse().ok()
    }
}

impl fmt::Display for Sexp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sexp::Atom(s) => write!(f, "{}", s),
            Sexp::List(v) => {
                write!(f, "(")?;
                for (i, e) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", e)?;
                }
                write!(f, ")")
            }
        }
    }
}

pub fn parse_all(input: &str) -> Result<Vec<Sexp>, String> {
    let tokens = tokenize(input)?;
    let mut cursor = 0usize;
    let mut out = Vec::new();
    while cursor < tokens.len() {
        out.push(parse(&tokens, &mut cursor)?);
    }
    if out.is_empty() {
        return Err("empty input".to_string());
    }
    Ok(out)
}

fn tokenize(input: &str) -> Result<Vec<String>, String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    for ch in input.chars() {
        match ch {
            '(' | ')' => {
                if !cur.is_empty() {
                    tokens.push(std::mem::take(&mut cur));
                }
                tokens.push(ch.to_string());
            }
            c if c.is_whitespace() => {
                if !cur.is_empty() {
                    tokens.push(std::mem::take(&mut cur));
                }
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }
    Ok(tokens)
}

fn parse(tokens: &[String], cursor: &mut usize) -> Result<Sexp, String> {
    let token = tokens
        .get(*cursor)
        .ok_or_else(|| "unexpected end of input".to_string())?;
    match token.as_str() {
        "(" => {
            *cursor += 1;
            let mut items = Vec::new();
            loop {
                match tokens.get(*cursor) {
                    None => return Err("unbalanced '('".to_string()),
                    Some(t) if t == ")" => {
                        *cursor += 1;
                        return Ok(Sexp::List(items));
                    }
                    Some(_) => items.push(parse(tokens, cursor)?),
                }
            }
        }
        ")" => Err("unexpected ')'".to_string()),
        other => {
            *cursor += 1;
            Ok(Sexp::Atom(other.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nested_lists() {
        let v = parse_all("(field (p 5) Fp (Zinv 5))").unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].to_string(), "(field (p 5) Fp (Zinv 5))");
    }

    #[test]
    fn parses_multiple_forms() {
        let v = parse_all("Fp  Q").unwrap();
        assert_eq!(v.len(), 2);
        assert_eq!(v[1].atom(), Some("Q"));
    }

    #[test]
    fn rejects_unbalanced() {
        assert!(parse_all("(field (p 5)").is_err());
        assert!(parse_all(")").is_err());
    }

    #[test]
    fn nat_rejects_non_digits() {
        assert_eq!(Sexp::Atom("12".into()).nat(), Some(12));
        assert_eq!(Sexp::Atom("1p".into()).nat(), None);
        assert_eq!(Sexp::List(vec![]).nat(), None);
    }
}
