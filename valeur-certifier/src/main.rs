//! valeur-cert — a sound one-sided certifier for decidability of maximal
//! immediate extensions.
//!
//! Usage:
//!   valeur-cert                       run the built-in examples
//!   valeur-cert "(field Fp5 (Zinv 5))" "(field Fp5 Z)"
//!   valeur-cert --json ...
//!   echo "(field Fp5 (Zinv 5))" | valeur-cert --stdin
//!
//! Exit status is 0 when every input was certified and 1 otherwise, so the
//! tool drops into a shell pipeline as a filter.

mod certify;
mod decide;
mod model;
mod sexp;

use certify::{certify, Verdict};
use model::ValuedField;
use sexp::parse_all;

const USAGE: &str = "\
valeur-cert — certify decidability of maximal immediate extensions

USAGE
  valeur-cert [--json] [(field <residue> <group>) ...]
  valeur-cert --stdin [--json]

INPUT LANGUAGE
  (field <residue> <group>)            char(K) := char(kappa(v))
  (field <char> <residue> <group>)    explicit characteristic

  <char>     0 | (p n)
  <residue>  Q | Fp2 | Fp3 | Fp5 | (Fpn p n) | (ACF 0) | (ACF (p n))
             | (rational <residue>) | (hahn <residue> <group>) | (phull 0)
             | (phull (p n)) | (opaque <anything>)
  <group>    Z | Q | R | (Zinv p) | (sum <group> ...) | (divby p <group>)
             | (finkt e <group>) | (opaque <anything>)

  A finite extension L/K of ramification index e and residue degree f:
  (finkt e f <field ...> [<residue of L>])
  char(L) := char(K) and Gamma_L := Gamma_K[1/e]. When f = 1 the residue
  field is inherited; when f > 1 it must be stated, since e and f do not
  determine it.

VERDICT CONTRACT
  CERTIFIED     all five hypotheses of the transfer theorem were justified
  TAME, TRANSFER NOT ESTABLISHED
                 tameness holds but the transfer was not licensed, e.g. the
                 residue theory is undecided or the characteristics differ
  NOT-CERTIFIED a tameness hypothesis genuinely failed
  The tool never reports a theory as undecidable. It has no way to know.

EXAMPLES
  valeur-cert                                          built-in suite
  valeur-cert \"(field Fp5 (Zinv 5))\"                  F_p((t))^{1/p^inf}
  valeur-cert \"(field Fp5 Z)\"                         the hard case
  valeur-cert \"(field Q Q)\"                           tame, Th(Q) undecidable
  valeur-cert \"(finkt 5 1 (field Fp5 (Zinv 5)))\"       totally ramified degree 5
";

/// Examples taken from the worked-examples table of the paper.
const EXAMPLES: &[(&str, &str)] = &[
    ("F_p((t))  [the paper's open case]", "(field Fp5 Z)"),
    ("F_p((t^{1/p^inf}))  [perfect hull of F_p((t))]", "(field Fp5 (Zinv 5))"),
    (
        "overline F_p((t^{1/p^inf}))",
        "(field (ACF (p 5)) (Zinv 5))",
    ),
    ("F_p(t)((t))", "(field (rational Fp5) Z)"),
    ("ACF_0((t^Q))", "(field (ACF 0) Q)"),
    ("Q((t^Q))  [tame, but Th(Q) is undecidable]", "(field Q Q)"),
    (
        "Q_p(p^{1/p^inf})  [tame, mixed characteristic]",
        "(field 0 Fp5 (Zinv 5))",
    ),
    (
        "perfect hull of F_p((t)), given as a perfect hull",
        "(field (phull (hahn Fp5 Z)) (divby 5 Z))",
    ),
    (
        "perfect hull of F_p(t)  [not a power series field]",
        "(field (phull (hahn (rational Fp5) Z)) (divby 5 Z))",
    ),
    (
        "degree 5 totally ramified extension of F_p((t^{Z[1/p]}))",
        "(finkt 5 1 (field Fp5 (Zinv 5)))",
    ),
    (
        "degree 5 unramified extension, residue F_{p^5}",
        "(finkt 1 5 (field Fp5 (Zinv 5)) (Fpn 5 5))",
    ),
];

fn report(cert: &certify::Certificate, json: bool) {
    let verdict = cert.verdict();
    if json {
        println!("{{");
        println!("  \"field\": \"{}\",", escape(&cert.field.pretty()));
        println!("  \"input\": \"{}\",", escape(&cert.field.label));
        println!(
            "  \"verdict\": \"{}\",",
            match verdict {
                Verdict::Certified => "CERTIFIED",
                Verdict::TameOnly => "TAME_TRANSFER_NOT_ESTABLISHED",
                Verdict::Refused => "NOT-CERTIFIED",
            }
        );
        println!(
            "  \"mixed_characteristic\": {},",
            cert.mixed_characteristic()
        );
        println!("  \"checks\": [");
        let checks: Vec<_> = cert.all_checks().collect();
        for (i, chk) in checks.iter().enumerate() {
            let comma = if i + 1 == checks.len() { "" } else { "," };
            println!(
                "    {{ \"name\": \"{}\", \"status\": \"{}\", \"detail\": \"{}\" }}{}",
                escape(chk.name),
                chk.status.label(),
                escape(&chk.detail),
                comma
            );
        }
        println!("  ],");
        match cert.conclusion() {
            Some(c) => println!("  \"conclusion\": \"{}\"", escape(&c)),
            None => println!("  \"conclusion\": null"),
        }
        match cert.transfer_note() {
            Some(n) => println!("  \"note\": \"{}\"", escape(&n)),
            None => println!("  \"note\": null"),
        }
        println!("}}");
        return;
    }

    println!("field      : {}", cert.field.pretty());
    println!("input      : {}", cert.field.label);
    if cert.mixed_characteristic() {
        println!("note       : mixed characteristic");
    }
    println!("{}", "-".repeat(74));
    println!("tameness of the base field");
    for chk in &cert.tameness {
        println!("  [{:^7}] {}", chk.status.label(), chk.name);
        println!("             {}", chk.detail);
    }
    println!("decidability transfer");
    for chk in &cert.transfer {
        println!("  [{:^7}] {}", chk.status.label(), chk.name);
        println!("             {}", chk.detail);
    }
    println!("{}", "-".repeat(74));
    println!("VERDICT    : {}", verdict.label());
    if let Some(c) = cert.conclusion() {
        println!("{}", c);
    }
    if let Some(n) = cert.transfer_note() {
        for line in wrap(&n, 66) {
            println!("             {}", line);
        }
    }
    for b in cert.blockers() {
        for (i, line) in wrap(&b, 66).iter().enumerate() {
            if i == 0 {
                println!("  - {}", line);
            } else {
                println!("    {}", line);
            }
        }
    }
    if !cert.certified() {
        let note = "A refusal is not a refutation: either the field is genuinely \
                    non-tame, or no proof is on file here.";
        for line in wrap(note, 66) {
            println!("             {}", line);
        }
    }
}

/// Greedy word wrap, so the tool has no dependencies and no surprises.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if line.is_empty() {
            line.push_str(word);
        } else if line.chars().count() + 1 + word.chars().count() <= width {
            line.push(' ');
            line.push_str(word);
        } else {
            out.push(std::mem::take(&mut line));
            line.push_str(word);
        }
    }
    if !line.is_empty() {
        out.push(line);
    }
    out
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn read_stdin() -> Result<Vec<sexp::Sexp>, String> {
    let mut buf = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)
        .map_err(|e| format!("cannot read stdin: {}", e))?;
    parse_all(&buf)
}

/// Runs every form, reporting each. Returns `Ok(true)` when all forms were
/// certified, `Ok(false)` when at least one was refused, and `Err` only for
/// malformed input, which is a distinct failure mode with its own exit code.
fn run(forms: Vec<sexp::Sexp>, json: bool) -> Result<bool, String> {
    let mut all_ok = true;
    let multi = forms.len() > 1;
    for form in &forms {
        let vf = ValuedField::parse(form).map_err(|e| format!("input error: {}", e))?;
        let cert = certify(&vf);
        if !cert.certified() {
            all_ok = false;
        }
        if !json && multi {
            println!();
        }
        report(&cert, json);
    }
    Ok(all_ok)
}

fn finish(result: Result<bool, String>) -> ! {
    match result {
        Ok(true) => std::process::exit(0),
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("{}", e);
            eprintln!("try `valeur-cert --help`");
            std::process::exit(2);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print!("{}", USAGE);
        return;
    }

    let json = args.iter().any(|a| a == "--json");
    let rest: Vec<&String> = args.iter().filter(|a| a.as_str() != "--json").collect();

    if rest.iter().any(|a| a.as_str() == "--stdin") {
        finish(read_stdin().and_then(|forms| run(forms, json)));
    }

    if rest.is_empty() {
        let mut forms = Vec::new();
        println!("built-in examples from the worked-examples table");
        println!("{}", "=".repeat(72));
        for (name, src) in EXAMPLES {
            match parse_all(src) {
                Ok(v) => {
                    println!("  {}", name);
                    forms.push(v[0].clone());
                }
                Err(e) => panic!("built-in example '{}' is malformed: {}", src, e),
            }
        }
        println!();
        finish(run(forms, json));
    }

    let joined = rest.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" ");
    match parse_all(&joined) {
        Ok(forms) => finish(run(forms, json)),
        Err(e) => finish(Err(format!("input error: {}", e))),
    }
}