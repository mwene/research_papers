export function buildParams(a, answeredCount, shownCount) {
  const best = Number(a.best_case ?? 5);
  const prob = Number(a.best_prob ?? 0.45);
  const worst = Number(a.worst_case ?? -4);
  const swing = best * prob + worst * (1 - prob);

  return {
    description: a.description || null,
    domain: a.domain || null,
    state_now: Number(a.progress ?? 0.5),
    floor_distance: Number(a.floor_proximity ?? 0.6),
    gain_rate: round4(swing / 30),
    wait_drift: Number(a.wait_outcome ?? 0),
    volatility: Number(a.predictability ?? 0.16),
    horizon_months: Number(a.horizon ?? 24),
    forced_in_months: a.forced_when !== undefined ? Number(a.forced_when) : null,
    act_cost: Number(a.act_cost ?? 0.45),
    floor_severity: Number(a.floor_exists ?? 0.05),
    data_quality: shownCount > 0 ? round2(answeredCount / shownCount) : 1
  };
}

function round4(x) { return Math.round(x * 10000) / 10000; }
function round2(x) { return Math.round(x * 100) / 100; }

export function paramsToText(p) {
  return JSON.stringify(p, null, 2);
}
