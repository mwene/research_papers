# valeur-cert

A sound one-sided certifier for decidability of maximal immediate
extensions of valued fields, implementing the transfer theorem of
`interesting_number_theory.pdf`.

No dependencies. `cargo build --release` and it runs.

## Two questions, two verdicts

The tool reports **tameness** and the **decidability transfer** separately,
because they have different hypotheses and are separately useful.

| Group | Checks | Hypotheses |
|-------|--------|------------|
| Tameness | (TF1) `p`-divisibility of `Γ_v`; (TF2) perfection of `κ(v)` | valid in **every** characteristic |
| Transfer | equal characteristic; `Th(Γ_v)` recursive; `Th(κ(v))` recursive | requires **equal** characteristic |

| Verdict | Meaning | Exit |
|---------|---------|------|
| `CERTIFIED` | every transfer hypothesis justified | 0 |
| `TAME, TRANSFER NOT ESTABLISHED` | the field is tame, but the transfer was not licensed | 1 |
| `NOT-CERTIFIED` | a tameness hypothesis genuinely failed | 1 |
| input error | malformed spec | 2 |

The split exists because in **mixed characteristic relative completeness is
known to fail and relative decidability is open**. A tameness certifier can
still be sound there; a decidability certifier cannot. Rather than
extrapolate, the tool reports tameness and declines the transfer.

## What this is not

It is **not** a total decision procedure, and it never says a theory is
undecidable. Deciding whether `Th(Γ)` is recursive is itself undecidable in
general. Three guards enforce this:

- every unanalysable input degrades to `UNKNOWN`, never to `NO`;
- an unknown characteristic is `Char::Unknown`, never silently `0`, so an
  opaque field cannot be *claimed* to be of equal characteristic;
- the theory registry is a whitelist — a missing entry means *no proof on
  file*, yielding a refusal rather than a refutation.

So the three-valued logic is load-bearing, not decoration.

## Two correctness traps this design closes

**The tame-shape gate.** `F_p` and `Z` are each individually decidable, so a
naive registry would certify `Th(F_p((t)))` — the paper's *open case*.
Relative decidability (Kuhlmann, Thm 1) requires the field to be tame, and
tameness of a power series field is not automatic: the value group must
supply the roots. `src/decide.rs` therefore gates the Hahn entry on
`is_perfect(base)` together with `p_divisible(group, p)` (resp. `divisible`
in characteristic zero). Without the gate this tool would have asserted a
long-standing open problem.

**`Q` is perfect but its theory is undecidable.** Tameness and decidability
are independent obligations, and conflating them is the sharpest trap here.
Every field of characteristic zero is perfect, so `Q` clears (TF2) and
`Q((t^Q))` *is* tame. But Robinson proved `Th(Q)` is undecidable, so the
transfer has no licence. An earlier version of this tool listed `Q` as
decidable and duly certified `Q((t^Q))` — a false certification, since the
residue-field hypothesis of the transfer theorem fails. The registry now
carries no entry for `Q`, and `(field Q Q)` returns **TAME, TRANSFER NOT
ESTABLISHED**. Rumely's theorem puts every global field out of reach for the
same reason, including each `F_q(t)`.

**Perfectness of rational function fields.** Separately, in characteristic `p`
a rational function field is never perfect: `(k(t))^p = k^p(t^p)` omits `t`.
Lisinski's Theorem 4 requires the residue `F` to be *perfect*, which is why
`F_p(t)` cannot feed a positive-characteristic Hahn field even though `F_p`
itself can.

## Usage

```sh
cargo run --release
cargo run --release -- "(field Fp5 (Zinv 5))"
cargo run --release -- --json "(field Q Q)"
echo '(field Fp5 Z)' | cargo run --release -- --stdin
```

## Input language

```
(field <residue> <group>)            char(K) := char(kappa(v))
(field <char> <residue> <group>)    explicit characteristic

<char>     0 | (p n)
<residue>  Q | Fp2 | Fp3 | Fp5 | (Fpn p n) | (ACF 0) | (ACF (p n))
           | (rational <residue>) | (hahn <residue> <group>)
           | (phull <char>) | (phull <residue>) | (opaque <anything>)
<group>    Z | Q | R | (Zinv p) | (sum <group> ...)
           | (divby p <group>) | (opaque <anything>)
```

Notes on two forms:

- `(phull (p 5))` records only a characteristic, so no normal form can be
  recognised and the tool declines. `(phull <residue>)` is the precise
  form, and normalises when the inner field is a power series field over a
  perfect residue.
- `(divby p G)` is `G ⊗_Z Z[1/p]`, the exponent group of a perfect hull.

The two arities of `(field ...)` collide when the first element is a
characteristic, so that case is rejected with an actionable message rather
than misparsed.

## Perfect-hull normalisation

For perfect `k` of characteristic `p`, the perfect hull of `k((t^G))` is
the power series field with exponent group `G ⊗_Z Z[1/p]`. This is
implemented (`normalise_perfect_hull`) and is what lets the tool certify
perfect hulls of power series fields.

It correctly declines for the perfect hull of `F_p(t)`, which is not a
power series field at all.

## Built-in examples

| Field | Verdict | Blocking clause |
|-------|---------|-----------------|
| `F_p((t))` | NOT-CERTIFIED | (TF1): `Z` is not `p`-divisible |
| `F_p((t^{1/p^inf}))` | CERTIFIED | — |
| `overline F_p((t^{1/p^inf}))` | CERTIFIED | — |
| `F_p(t)((t))` | NOT-CERTIFIED | (TF1) *and* (TF2) |
| `ACF_0((t^Q))` | CERTIFIED | — |
| `Q((t^Q))` | TAME, TRANSFER NOT ESTABLISHED | `Th(Q)` is undecidable |
| degree 5 totally ramified extension of `F_p((t^{Z[1/p]}))` | CERTIFIED | — |
| degree 5 unramified extension, residue `F_{p^5}` | CERTIFIED | — |
| `Q_p(p^{1/p^inf})` | TAME, TRANSFER NOT ESTABLISHED | mixed characteristic |
| perfect hull of `F_p((t^Z))` | CERTIFIED | — |
| perfect hull of `F_p(t)` | TAME, TRANSFER NOT ESTABLISHED | residue not normalisable |

Refusing the first row is the correct output, not a bug: it is the paper's
open case.

## Layout

| File | Role |
|------|------|
| `src/sexp.rs` | minimal S-expression reader |
| `src/model.rs` | `Char`, `GroupExpr`, `Residue`, `ValuedField` |
| `src/decide.rs` | `p`-divisibility, perfectness, perfect-hull normal form, registry |
| `src/certify.rs` | the two check groups and the three-valued verdict |
| `src/main.rs` | CLI, report rendering, exit codes |

## Tests

```sh
cargo test
```

49 unit tests: the parser, the two-arity ambiguity guard, three-valued
logic, `p`-divisibility on direct sums, divisible closures and finite-extension
localisations, the `divby` argument bug that made `Z[1/3]` answer `5`-divisible,
perfectness of Hahn fields in both characteristics, the conservative registry
(including the absence of `Q` and of `F_q(t)`), finite-extension parsing and
certification in equal and mixed characteristic, the refusal to certify
`Th(F_p((t)))`, and a verdict for every row of the paper's examples table.

## Known limitations

- No entry for rational function fields in the decidability registry, and
  this is a theorem rather than a missing feature. In characteristic `p` they
  are not perfect, so Lisinski's Theorem 4 cannot apply; over a global base
  the theory is undecidable by Rumely, so no entry could be sound anyway.
- The perfect hull of a field that is not a power series field is
  unrecognised.
- Field arithmetic covers finite extensions of a declared residue and value
  group via `finkt`. It does not compute residue field extensions: a
  descriptor never produces an unlisted residue field on its own, and it
  never infers an unlisted value group. Specific extensions must still be
  named, so the tool certifies families rather than resolving polynomials.
