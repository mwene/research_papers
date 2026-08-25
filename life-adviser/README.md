# The Decider — a generic life adviser

You face a decision. The Decider asks you a short, adaptive set of questions,
distills your situation into **five quantities** from Stochastic Control Theory,
and returns one of three verdicts:

- **Act now** — act voluntarily, while the choice is still yours
- **Wait & watch** — gather information; nothing forces your hand yet
- **Let it go** — this path loses more than it wins

It is built on the framework developed in *The Control Within*
(`stochastic_control_theory.pdf`): every decision shares one
skeleton — a state, actions, payoffs, a floor that must not be crossed, and a
target worth reaching. The engine's sharpest edge is timing: across finance,
medicine, logistics and control, **chosen actions succeed two-to-three times
more often than forced ones**, so the engine constantly asks *"when does this
stop being your choice?"*

Everything runs **locally in your browser** via WebAssembly. No accounts, no
server, no data leaves the machine.

## Quick start

Prerequisites (one-time):

```sh
curl -sSf https://rustwasm.github.io/wasm-pack/installer/init.sh | sh   # wasm-pack
rustup target add wasm32-unknown-unknown
```

Build and run (from this directory):

```sh
# 1. Compile the Rust engine to WebAssembly (~2 min first time)
cd decider && ~/.cargo/bin/wasm-pack build --target web --release --out-dir ../webapp/pkg

# 2. Serve the website (any static server works)
cd ../webapp && python3 -m http.server 8765

# 3. Open http://localhost:8765 in your browser
```

That's it — describe a decision, answer ~12 questions, get a verdict.

Verify the install without a browser:

```sh
node tests/smoke.mjs     # drives the compiled WASM through 5 scenarios
cd decider && cargo test # 22 unit tests of the math and verdict logic
```

## How it works

```
you describe a decision
        │
        ▼
adaptive questionnaire ── follow-up questions appear based on earlier answers
(js/questions.js)         e.g. "how close are you to that line?" only if a floor exists
        │
        ▼
parameter extraction ──── answers → DecisionParams (js/params.js)
        │
        ▼
┌─────────────────────────────────────────────────┐
│  decider (Rust crate, compiled to WASM)         │
│                                                 │
│  params ──► five quantities Θ1..Θ5              │
│         ──► utility model                       │
│         ──► softmax scores                      │
│         ──► verdict + advice text               │
└─────────────────────────────────────────────────┘
        │
        ▼
result screen: verdict badge, conviction gauge,
option scores, five-quantity readout, advice list
```

## The five quantities (Θ1–Θ5)

| # | Quantity | Meaning |
|---|----------|---------|
| 1 | Expected net gain | drift if acting minus drift if waiting, over your horizon |
| 2 | Uncertainty band | ±1σ of progress at horizon |
| 3 | Time to goal | distance ÷ gain rate (`unreachable` when gain ≤ 0) |
| 4 | Floor risk | P(min X < floor) for drifted Brownian motion — acting vs waiting |
| 5 | Urgency | how fast the voluntary window closes |

The utilities are transparent by design: acting pays expected gain minus act
cost and residual floor risk; waiting adds information value (grows with
uncertainty, shrinks as the window closes) but suffers forced-action penalties;
dropping forfeits upside but escapes losing paths.

## Reusing the `decider` crate

The engine has no web dependencies — use it from any Rust program, CLI, or
server (disable the WASM layer with `default-features = false`):

```toml
[dependencies]
decider = { path = "../decider" }
```

```rust
use decider::{DecisionParams, decide_from_params};

let p = DecisionParams {
    state_now: Some(0.45),       // [0,1] position toward goal
    floor_distance: Some(0.4),   // distance above the hard line
    gain_rate: Some(0.06),       // progress/month if acting
    wait_drift: Some(-0.01),     // progress/month if waiting (neg = decay)
    volatility: Some(0.10),      // per sqrt(month)
    horizon_months: Some(12.0),
    forced_in_months: Some(6.0), // ≥900 means "never forced"
    act_cost: Some(0.2),         // [0,1]
    floor_severity: Some(0.1),   // [0,1]
    data_quality: Some(0.9),     // [0,1], scales confidence
    ..Default::default()         // all fields optional; defaults are neutral
};

let v = decide_from_params(&p);
println!("{} — {:?}", v.headline, v.action);
for line in v.advice { println!("· {line}"); }
```

From JavaScript (as the site does):

```js
import init, { decide_json } from "./pkg/decider.js";
await init();
const verdict = JSON.parse(decide_json(JSON.stringify(params)));
// verdict.action: "act_now" | "wait" | "drop"
```

## Customizing the questionnaire

Questions live in `webapp/js/questions.js`. Each question maps an answer onto
a parameter; `showIf` makes questions adaptive. Add a domain-specific question
by appending a node — no engine changes needed unless you introduce new
parameters.

## Project layout

```
life-adviser/
├── decider/          Reusable Rust crate (the engine)
│   └── src/
│       ├── mathutil.rs    erf / normal CDF (no external deps)
│       ├── quantities.rs  Brownian hitting probabilities, hitting times, urgency
│       ├── five.rs        The five quantities Θ1–Θ5
│       ├── verdict.rs     Utility model → softmax scores → ActNow/Wait/Drop
│       ├── advice.rs      Headline + advice-text generation
│       ├── params.rs      DecisionParams + validation/clamping
│       └── wasm.rs        wasm-bindgen JSON API (feature "wasm", on by default)
├── webapp/           Plain HTML/CSS/JS site
│   ├── js/questions.js   Adaptive questionnaire definitions
│   ├── js/params.js      Answers → DecisionParams mapping
│   ├── js/main.js        Screens, rendering, WASM calls
│   └── pkg/              Generated by wasm-pack (do not edit)
└── tests/smoke.mjs   Node end-to-end smoke test of the compiled WASM
```

## Honest limits (per the theory itself)

The guarantees are conditional on the inputs being approximately honest — the
engine clamps nonsense values and reports a confidence scaled by data quality,
but it cannot audit its own model (the Gödelian limit applies to it too).
Treat verdicts as a well-reasoned floor, not an oracle.
