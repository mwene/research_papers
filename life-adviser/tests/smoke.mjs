import { readFileSync } from "fs";
import init, { decide_json, decider_version } from "../webapp/pkg/decider.js";

await init(readFileSync(new URL("../webapp/pkg/decider_bg.wasm", import.meta.url)));
console.log("engine version:", decider_version());

const cases = [
  ["strong deal, window closing", { state_now: 0.45, floor_distance: 0.4, gain_rate: 0.06, wait_drift: -0.01, volatility: 0.10, horizon_months: 12, forced_in_months: 6, act_cost: 0.2, floor_severity: 0.1 }],
  ["uncertain, open-ended", { state_now: 0.55, floor_distance: 0.5, gain_rate: 0.02, wait_drift: 0.0, volatility: 0.5, horizon_months: 12, forced_in_months: 9999, act_cost: 0.5, floor_severity: 0.05 }],
  ["slow loser", { state_now: 0.5, floor_distance: 0.6, gain_rate: -0.02, wait_drift: 0.0, volatility: 0.08, horizon_months: 12, forced_in_months: 9999, act_cost: 0.4, floor_severity: 0.05 }],
  ["floor emergency", { state_now: 0.30, floor_distance: 0.12, gain_rate: 0.03, wait_drift: -0.06, volatility: 0.20, horizon_months: 12, forced_in_months: 18, act_cost: 0.4, floor_severity: 0.9 }],
  ["minimal input", {}]
];

for (const [name, p] of cases) {
  const v = JSON.parse(decide_json(JSON.stringify(p)));
  console.log(`${name.padEnd(30)} -> ${v.action.padEnd(8)} strength=${(v.decision_value * 100).toFixed(0)}% conf=${(v.confidence * 100).toFixed(0)}% | ${v.headline}`);
}
