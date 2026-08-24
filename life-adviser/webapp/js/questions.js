export const QUESTIONS = [
  {
    id: "domain",
    type: "choice",
    title: "What kind of decision is this?",
    help: "This tunes the language of the advice, not the mathematics — the same five quantities govern every domain.",
    options: [
      { value: "money", label: "Money & investments" },
      { value: "work", label: "Work & career" },
      { value: "health", label: "Health" },
      { value: "relationship", label: "Relationships" },
      { value: "purchase", label: "A purchase or commitment" },
      { value: "other", label: "Something else" }
    ]
  },
  {
    id: "best_case",
    type: "slider",
    title: "If this goes as well as it realistically can, how much better does your situation get?",
    help: "0 means no real improvement. 10 means transformative.",
    min: 0, max: 10, step: 1,
    leftLabel: "no real change",
    rightLabel: "transformative",
    fmt: v => ["none", "tiny", "small", "small+", "moderate", "moderate+", "large", "large+", "very large", "huge", "transformative"][v]
  },
  {
    id: "best_prob",
    type: "choice",
    title: "How likely is that best case?",
    options: [
      { value: 0.85, label: "Very likely", hint: "the pieces are already in place" },
      { value: 0.65, label: "Likely", hint: "some things must go right" },
      { value: 0.45, label: "Could go either way" },
      { value: 0.25, label: "Uncertain", hint: "more must go right than wrong" },
      { value: 0.1, label: "Long shot" }
    ]
  },
  {
    id: "worst_case",
    type: "choice",
    title: "And if it goes badly?",
    help: "Think of the realistic bad outcome, not the nightmare.",
    options: [
      { value: -1.5, label: "A slight setback", hint: "wasted time or a little money" },
      { value: -4, label: "Serious damage", hint: "a meaningful loss I'd feel for months" },
      { value: -8, label: "Close to disaster", hint: "near the worst thing on my plate" }
    ]
  },
  {
    id: "predictability",
    type: "choice",
    title: "How predictable is the path from here to the outcome?",
    options: [
      { value: 0.07, label: "Fairly predictable", hint: "I mostly know what follows what" },
      { value: 0.16, label: "Moderately uncertain" },
      { value: 0.34, label: "Very uncertain", hint: "many moving parts, little visibility" },
      { value: 0.55, label: "Near-chaotic", hint: "outcomes swing wildly on small events" }
    ]
  },
  {
    id: "wait_outcome",
    type: "choice",
    title: "If you do nothing for now, what happens to the situation?",
    options: [
      { value: 0.006, label: "It improves on its own" },
      { value: 0, label: "It stays about the same" },
      { value: -0.02, label: "It slowly gets worse" },
      { value: -0.06, label: "It deteriorates fast" }
    ]
  },
  {
    id: "floor_exists",
    type: "choice",
    title: "Is there a hard line here that must not be crossed?",
    help: "Money running out, a health crisis, an irreversible rupture — a floor you cannot afford to lose.",
    options: [
      { value: 0.05, label: "No hard line", hint: "bad, but recoverable at any point" },
      { value: 0.45, label: "Yes, a serious one" },
      { value: 0.9, label: "Yes, and crossing it would be devastating" }
    ]
  },
  {
    id: "floor_proximity",
    type: "choice",
    showIf: a => a.floor_exists > 0.2,
    title: "How close are you to that line today?",
    options: [
      { value: 0.6, label: "Comfortably far from it" },
      { value: 0.3, label: "Getting closer" },
      { value: 0.12, label: "Alarmingly close" }
    ]
  },
  {
    id: "progress",
    type: "choice",
    title: "Where do you stand today on this?",
    options: [
      { value: 0.25, label: "Just started thinking about it" },
      { value: 0.5, label: "In motion", hint: "partway toward the goal" },
      { value: 0.75, label: "Nearly there" }
    ]
  },
  {
    id: "forced_when",
    type: "choice",
    title: "When does this stop being your choice?",
    help: "The theory's sharpest edge: acting voluntarily beats being forced. When does the decision get made by circumstances instead of you?",
    options: [
      { value: 0.5, label: "Within weeks" },
      { value: 3, label: "In about 3 months" },
      { value: 6, label: "In about 6 months" },
      { value: 12, label: "Within a year" },
      { value: 24, label: "A year or more away" },
      { value: 9999, label: "Never — the choice stays mine indefinitely" }
    ]
  },
  {
    id: "act_cost",
    type: "choice",
    title: "What does acting now cost you?",
    help: "Effort, money, awkwardness, risk of moving too early.",
    options: [
      { value: 0.15, label: "Small", hint: "an email, a call, a small fee" },
      { value: 0.45, label: "Significant", hint: "real work or real money" },
      { value: 0.75, label: "Heavy sacrifice" }
    ]
  },
  {
    id: "horizon",
    type: "slider",
    title: "Over what period should this play out?",
    help: "How long until you'd judge whether this worked?",
    min: 3, max: 120, step: 1,
    leftLabel: "3 months",
    rightLabel: "10 years",
    fmt: v => (v < 12 ? `${v} mo` : `${(v / 12).toFixed(v % 12 === 0 ? 0 : 1)} yr`)
  }
];
