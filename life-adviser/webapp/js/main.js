import init, { decide_json, decider_version } from "../pkg/decider.js";
import { QUESTIONS } from "./questions.js";
import { buildParams, paramsToText } from "./params.js";

const app = document.getElementById("app");

const state = {
  answers: {},
  shownIds: [],
  index: 0,
  ready: false,
};

function el(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

function visibleQuestions() {
  return QUESTIONS.filter(q => !q.showIf || q.showIf(state.answers));
}

function renderLoading() {
  app.innerHTML = "";
  app.appendChild(el(`<div class="card loading">Waking up the decision engine…</div>`));
}

function renderIntro() {
  state.answers = {};
  state.index = 0;
  app.innerHTML = "";
  const card = el(`
    <div class="card">
      <h1>What are you <em>deciding</em>?</h1>
      <p class="sub">Describe the situation in a sentence or two — then answer a short set of questions.
      The engine extracts five quantities from stochastic control theory
      (expected gain, uncertainty, time-to-goal, floor risk, urgency) and returns one of three verdicts:
      <strong style="color:var(--gold)">act now</strong>,
      <strong style="color:var(--blue)">wait</strong>, or
      <strong style="color:var(--red)">let it go</strong>.</p>
      <textarea id="desc" placeholder="e.g. Should I quit my job to start the company I keep sketching in notebooks?"></textarea>
      <div class="domains">
        ${["money", "work", "health", "relationship", "purchase", "other"]
          .map(d => `<button class="domain-chip" data-d="${d}">${d}</button>`).join("")}
      </div>
      <div class="navrow">
        <span></span>
        <button class="primary" id="begin" disabled disabled-title="Pick what kind of decision this is first">Begin →</button>
      </div>
    </div>`);
  card.querySelectorAll(".domain-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      card.querySelectorAll(".domain-chip").forEach(c => c.classList.remove("sel"));
      chip.classList.add("sel");
      state.answers.domain = chip.dataset.d;
      card.querySelector("#begin").disabled = false;
    });
  });
  card.querySelector("#desc").addEventListener("input", e => {
    state.answers.description = e.target.value;
  });
  card.querySelector("#begin").addEventListener("click", () => {
    state.index = 0;
    renderQuestion();
  });
  app.appendChild(card);
}

function renderQuestion() {
  const qs = visibleQuestions();
  if (state.index >= qs.length) {
    renderResult();
    return;
  }
  const q = qs[state.index];
  state.shownIds = qs.map(x => x.id);
  app.innerHTML = "";

  const pct = Math.round((state.index / qs.length) * 100);
  const card = el(`<div class="card">
    <div class="progress"><div class="fill" style="width:${Math.max(pct, 4)}%"></div></div>
    <h2 class="qtitle">${q.title}</h2>
    ${q.help ? `<p class="qhelp">${q.help}</p>` : ""}
    <div id="qbody"></div>
    <div class="navrow">
      <button class="ghost" id="back">← Back</button>
      <button class="primary" id="next" disabled>Next →</button>
    </div>
  </div>`);

  const body = card.querySelector("#qbody");
  const nextBtn = card.querySelector("#next");
  let canProceed = false;

  if (q.type === "choice") {
    const wrap = el(`<div class="choices"></div>`);
    for (const opt of q.options) {
      const b = el(`<button class="choice">${opt.label}${opt.hint ? `<span class="hint">${opt.hint}</span>` : ""}</button>`);
      b.addEventListener("click", () => {
        wrap.querySelectorAll(".choice").forEach(c => c.style.borderColor = "");
        b.style.borderColor = "var(--gold)";
        state.answers[q.id] = opt.value;
        canProceed = true;
        nextBtn.disabled = false;
      });
      wrap.appendChild(b);
    }
    body.appendChild(wrap);
  } else if (q.type === "slider") {
    const val = Number(state.answers[q.id] ?? Math.round((q.min + q.max) / 2));
    const wrap = el(`<div class="slider-wrap">
      <div class="slider-val"></div>
      <input type="range" min="${q.min}" max="${q.max}" step="${q.step}" value="${val}">
      <div class="slider-ends"><span>${q.leftLabel}</span><span>${q.rightLabel}</span></div>
    </div>`);
    const range = wrap.querySelector("input");
    const label = wrap.querySelector(".slider-val");
    const update = () => {
      label.textContent = q.fmt(Number(range.value));
      state.answers[q.id] = Number(range.value);
      canProceed = true;
      nextBtn.disabled = false;
    };
    range.addEventListener("input", update);
    update();
    body.appendChild(wrap);
  }

  card.querySelector("#back").addEventListener("click", () => {
    if (state.index === 0) { renderIntro(); return; }
    state.index--;
    renderQuestion();
  });

  nextBtn.addEventListener("click", () => {
    if (!canProceed) return;
    state.index++;
    renderQuestion();
  });

  app.appendChild(card);
}

function fmtQty(v) {
  return v.five.time_to_goal_months === null || v.five.time_to_goal_months === undefined
    ? "unreachable"
    : v.five.time_to_goal_months >= 890 ? "—" : `${v.five.time_to_goal_months.toFixed(1)} mo`;
}

function renderResult() {
  const qs = visibleQuestions();
  const answered = qs.filter(q => state.answers[q.id] !== undefined).length;
  const p = buildParams(state.answers, answered, qs.length);
  const raw = decide_json(JSON.stringify(p));
  const v = JSON.parse(raw);

  const actionWord = { act_now: "Act now", wait: "Wait", drop: "Let it go" }[v.action];
  const cls = { act_now: "verdict-act", wait: "verdict-wait", drop: "verdict-drop" }[v.action];
  const colors = { act_now: "var(--gold)", wait: "var(--blue)", drop: "var(--red)" };

  const arc = (frac, color) => {
    const r = 52, c = 2 * Math.PI * r;
    return `<svg width="140" height="140" viewBox="0 0 140 140">
      <circle cx="70" cy="70" r="${r}" fill="none" stroke="#10162a" stroke-width="12"/>
      <circle cx="70" cy="70" r="${r}" fill="none" stroke="${color}" stroke-width="12"
        stroke-linecap="round" stroke-dasharray="${(c * frac).toFixed(1)} ${c.toFixed(1)}"
        transform="rotate(-90 70 70)"/>
      <text x="70" y="66" text-anchor="middle" fill="#e8ecf5" font-size="26" font-family="Georgia">${(frac * 100).toFixed(0)}%</text>
      <text x="70" y="88" text-anchor="middle" fill="#93a0b8" font-size="11">conviction</text>
    </svg>`;
  };

  const scoreRow = (name, val, color) => `
    <div class="score-row">
      <div class="name"><span>${name}</span><span>${(val * 100).toFixed(0)}%</span></div>
      <div class="bar"><div class="fillbar" style="width:${(val * 100).toFixed(1)}%;background:${color}"></div></div>
    </div>`;

  const pctS = x => `${(x * 100).toFixed(0)}%`;

  app.innerHTML = "";
  const card = el(`<div class="card">
    <div class="verdict-badge ${cls}">
      <div class="label">The Decider says</div>
      <div class="word">${actionWord}</div>
      <div class="headline">${v.headline}</div>
    </div>

    <div class="gauges">
      <div class="gaugebox">${arc(v.decision_value, colors[v.action])}<h3>Decision strength</h3></div>
      <div class="scorebox">
        <h3>Option scores</h3>
        ${scoreRow("Act now", v.scores.act_now, "var(--gold)")}
        ${scoreRow("Wait & watch", v.scores.wait, "var(--blue)")}
        ${scoreRow("Let it go", v.scores.drop, "var(--red)")}
        <div style="margin-top:10px;color:var(--muted);font-size:13px">confidence ${(v.confidence * 100).toFixed(0)}% · data quality ${pctS(p.data_quality)}</div>
      </div>
    </div>

    <div class="five-grid">
      <div class="five-item"><div class="k">Expected net gain</div><div class="v">${v.five.expected_net_gain.toFixed(2)}</div><div class="n">goal-units over your horizon</div></div>
      <div class="five-item"><div class="k">Uncertainty band</div><div class="v">±${v.five.uncertainty_band.toFixed(2)}</div><div class="n">one sigma at horizon</div></div>
      <div class="five-item"><div class="k">Time to goal</div><div class="v">${fmtQty(v)}</div><div class="n">at current pace</div></div>
      <div class="five-item"><div class="k">Floor risk · act / wait</div><div class="v">${pctS(v.five.floor_risk_if_act)} / ${pctS(v.five.floor_risk_if_wait)}</div><div class="n">chance of crossing the line</div></div>
      <div class="five-item"><div class="k">Urgency</div><div class="v">${pctS(v.five.urgency)}</div><div class="n">how fast the voluntary window closes</div></div>
    </div>

    <div class="advice">
      <h3>Advice</h3>
      <ul>${v.advice.map(a => `<li>${a}</li>`).join("")}</ul>
    </div>

    <div class="theory-note">
      Verdicts weigh acting early against being forced late — chosen actions succeed two-to-three times more often
      than forced ones across finance, medicine, logistics and control. Perfect self-reflection is mathematically
      unreachable (the Gödelian limit), so treat this as a floor, not an oracle. Engine v${decider_version()}.
    </div>

    <div class="result-actions">
      <button class="primary" id="again">Decide something else</button>
    </div>
  </div>`);

  card.querySelector("#again").addEventListener("click", renderIntro);
  app.appendChild(card);
}

async function boot() {
  renderLoading();
  await init();
  state.ready = true;
  renderIntro();
}

boot().catch(err => {
  app.innerHTML = "";
  app.appendChild(el(`<div class="card loading">Failed to load engine: ${err}</div>`));
});
