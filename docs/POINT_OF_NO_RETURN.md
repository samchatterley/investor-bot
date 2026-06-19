# POINT_OF_NO_RETURN.md — the experiment freeze protocol

> **Status:** DRAFT (not yet crossed). Operationalises the freeze that `docs/EXPERIMENT.md`
> requires: *"Freeze before collecting any primary data. Any change after data collection begins
> requires a new version and a new evaluation period."* This doc defines **what** freezes, the
> **checklist** that must be true before we cross, **how** we cross, and the **governance** afterward.
> Companion: `docs/EXPERIMENT.md` (the pre-registration / design), `docs/strategic_review.md` (why).
>
> **Path decision (2026-06-19): FULL GAUNTLET.** `evidence_score_v2` is fitted and frozen *before*
> the PNR, so the standable benchmark (EXPERIMENT.md §15.7) exists at t0 and there is a single clean
> freeze. No headline AI claim is possible until v2 exists, and under this path v2 exists at t0.

---

## 1. What the PNR is

The **Point of No Return (PNR)** is a single timestamped boundary, `t0`. Before `t0` the system is in
**shakedown** — code and design may change freely and any logged observation is *pilot* data. At and
after `t0`:

- every **frozen artifact** (section 2) is immutable; changing one requires a new `experiment_version`
  and a **new evaluation period** for the affected arm (section 6);
- **primary data collection is live**: the first observation whose `decision_time ≥ t0`, stamped with
  the frozen `experiment_version`, is the first primary observation.

`t0` is **not "now."** It is the terminus of the checklist in section 4 — because several checklist
items (Gate A granularity, Gate B scope, the A3.1 eligible-set, the v2 fit) *determine the design that
gets frozen*. Freezing earlier would freeze a half-specified instrument.

**Operational definition of t0.** The moment the freeze commit lands with `EXPERIMENT_FROZEN = True`
and `logs/experiment_freeze_manifest.json` written. The first scheduler run after that commit produces
the first primary observations. All `experiment_observations.jsonl` / `decisions.jsonl` rows before
that commit are **quarantined as pilot** (section 7).

---

## 2. The freeze manifest

At `t0` a freeze script records the exact identity of every frozen artifact into
`logs/experiment_freeze_manifest.json`. **Two pinning methods, by artifact kind** — because hashing a
raw source file is both too sensitive (a comment edit forces a version bump nobody intended) and too
blind (it misses a behaviour change that comes from an upstream config/dependency the file imports):

- **Data artifacts** (literal values — weights, the universe list, the packet spec, the model id, the
  flags): pinned by a **hash of the value**, plus a version constant where one exists.
- **Code-behaviour artifacts** (the prompt template + substitution, the detectors, the gate/sampling
  logic, the forward-R computation): pinned by a **golden-fixture hash** — fixed input fixtures → the
  rendered/decided output → hash *that*. Catches real behavioural drift (including drift driven by an
  upstream dependency) while ignoring cosmetic refactors.

A CI guard (section 6) re-derives both and fails the build if any pin drifts without an
`EXPERIMENT_VERSION` bump.

| # | Frozen artifact | Source of truth | Pinned by |
|---|---|---|---|
| 1 | Tradeable universe (907) | `config.STOCK_UNIVERSE` | value hash |
| 2 | Signal book + priority (active/disabled) | `config.SIGNAL_PRIORITY`, `SHORT_SIGNAL_PRIORITY`, `GLOBALLY_DISABLED`, `SHORT_GLOBALLY_DISABLED` | value hash |
| 3 | `evidence_score_v1` weights (Arm 1 floor) | `experiment/evidence_score.py::_WEIGHTS`, `EVIDENCE_SCORE_VERSION` | version + value hash |
| 4 | `evidence_score_v2` (Arm 1 standable benchmark) | v2 module + fitted-weights artifact + train/val/holdout split spec | version + value(weights) hash |
| 5 | `context_packet_v1` spec | EXPERIMENT.md §6 / its config | version + value hash |
| 6 | Arm 2 prose template | the mechanical template module | version + **golden-fixture hash** |
| 7 | Arm 3 prompt incl. material-context flags | `analysis/ai_analyst.SYSTEM_PROMPT` / `build_prompt` | **golden-fixture hash** |
| 8 | LLM identity | model id + temperature + any sampling params | recorded values |
| 9 | Arm 3 output contract (granularity) | EXPERIMENT.md §7 (coarse vs {-2..+2} — set by Gate A) | version |
| 10 | Material-context detectors (§15.1) | the 10-category detector code paths | **golden-fixture hash** |
| 11 | Context-presence gate + random-control sampling | §8 logic | **golden-fixture hash** |
| 12 | Eligible-candidate-set filters | risk/liquidity filters + A3.1 resolution | **golden-fixture hash** |
| 13 | Forward-R metric (H=5, ATR14, cost model, delisting rule) | the measurement-overlay code | version + **golden-fixture hash** |
| 14 | `ADAPTIVE_PROMPT_ENABLED = False` | `config.py` | recorded value (must be False) |
| 15 | Testing schedule (N_eff 200 interim / 400 primary) | EXPERIMENT.md §10 | recorded |

The manifest also records: `experiment_version`, `t0` (UTC + ET), git commit SHA, the pilot-data
cutoff row counts, and the **Gate A noise characterisation** (sign/veto flip-rate, conviction
variance) — the frozen instrument's measured noise floor, which the clustered-SE inference depends on.

---

## 3. Frozen / safe-after / the trap  (which requests stay safe after t0)

This is the practical guide: after `t0`, a request is safe **only if it leaves the decision and every
logged variable bit-identical**.

**Frozen — change ⟹ new `experiment_version` + new evaluation period:** anything in the manifest
(section 2). In plain terms: what the bot *trades*, how it *selects / scores / ranks / vetoes*, what
point-in-time data *enters the decision*, the *prompt / model*, and the *definition of any logged
experimental variable*.

**Safe after t0 — touches neither the decision nor any logged field:**
- scheduler / launchd operations, restarts;
- the `logs/cache/` foldering (option a remainder), cache pruning, run-file naming;
- **purely additive** observability that never alters an existing field's value or meaning;
- analysis / reporting / plotting scripts that read the logs read-only.

**The trap — feels like a fix, is actually frozen (must land BEFORE t0):**
- **any data-feed repair** — it shifts the context distribution Arm 3 sees (e.g. the kind of fix the
  insider Form-4 / FinBERT / 8-K repairs were);
- **any fail-open → fail-safe change** (A5.1) — it changes *which* candidates trade/score on a
  degraded day, altering the sample;
- **any prompt wording change**, even "harmless" — Arm 3 is the treatment;
- **changing an existing observation field**'s semantics (adding a new field is fine; redefining one is not).

---

## 4. The PNR checklist (full gauntlet) — all must be TRUE before crossing

| # | Prerequisite | Why it gates the freeze | Status (2026-06-19) |
|---|---|---|---|
| P1 | `ADAPTIVE_PROMPT_ENABLED = False` (§15.9) | Otherwise Arm 3 is non-stationary + self-fitting; pooling invalid | ✅ met (1.112) |
| P2 | **Gate A** noise audit — on the *to-be-frozen* model/temperature, stratified real candidates, 5 repeats | Sets the Arm 3 output granularity frozen in §7 **and** its measured noise floor (recorded in the manifest); a noisy instrument changes the whole design | ❌ not run (only `.example.json`) |
| P3 | **Gate B** power analysis decided (= A7.1) | Scopes the live track: primary statistical test vs trend+qualitative layer | ❌ not decided |
| P4 | **A3.1 parity** resolved | Defines the eligible candidate set; live must equal the pre-registered baseline | ❌ open (sector gate advisory since 1.108; conc. cap + churn still live-only) |
| P5 | **`evidence_score_v2`** fitted + frozen (train/val/holdout) | Full-gauntlet: the standable benchmark must exist at t0 (§15.7) | ❌ not built |
| P6 | §15.3 controls wired | Positive controls detectable (harness sensitive); negative controls null (no leakage) | ⚠️ verify |
| P7 | §13 as-of context ledger enforces `provider_seen_at ≤ decision_time − buffer` | Lookahead safety for the context treatment | ⚠️ verify operating |
| P8 | Observation schema stamps `experiment_version` + `adaptive_prompt` mode (§15.9) | So every primary row is attributable to this freeze epoch | ❌ no per-obs stamp found |
| P9 | Version pinning + freeze manifest script | The freeze is recorded + machine-enforceable | ❌ not built |
| P10 | Feed health green at the freeze (`scripts/feed_health_check.py`) | Don't freeze on a degraded feed | ⚠️ check at t0 |
| P11 | **A10.1** cost-model realism sanity check (modeled vs realized fills) | item 13 freezes the cost model into forward-R; an unverified model can't be fixed post-t0 | ❌ open (needs accumulated fills) |
| P12 | **A8.1** ET-anchoring — no experiment decision path uses `date.today()` (must use `today_et()`) | a local-date path inherits a latent DST bug at the late-Oct BST→GMT shift | ⚠️ audit experiment paths |
| — | A2.1 live-pipeline lookahead | (verified clean, 1.107) | ✅ met |
| — | Universe frozen | bare list; pin by value hash at P9 | ✅ frozen / pin pending |

---

## 5. Crossing the line (the ritual)

0. **Dry-run:** run the freeze script `--dry-run` to produce the manifest *without* committing; eyeball
   every pin, the Gate A noise floor, and the pilot cutoff counts before anything is locked.
1. All P1–P12 are TRUE (Gate A/B passed or their pivots adopted; v2 frozen; parity resolved).
2. Run the freeze script for real → writes `logs/experiment_freeze_manifest.json` with all
   hashes/versions, the Gate A noise floor, the git SHA, and the pilot cutoff counts.
3. Set `EXPERIMENT_FROZEN = True` and `EXPERIMENT_VERSION = "<v>"` in config.
4. Commit (this commit's timestamp is `t0`), push, restart the scheduler **after market close**.
5. The first scheduler run after the freeze commit produces the first primary observations.

---

## 6. Governance after t0

- A **discretionary** change to any manifest artifact ⟹ bump `EXPERIMENT_VERSION`, re-run the freeze
  script (new manifest), and **start a new evaluation period** — `N_eff` for the affected arm resets;
  pre-change and post-change observations are never pooled in one test.
- **Enforcement:** a CI / pre-commit guard re-derives the manifest pins (section 2) and **fails** if a
  frozen artifact changed without an `EXPERIMENT_VERSION` bump — i.e. the artifact and the version must
  move *in the same commit* for the change to be allowed; an artifact moving alone is silent drift and
  is blocked. This is what makes the freeze real rather than a promise (consistent with the project's
  "all commits pass CI" discipline).
- The **testing schedule** (§10) still governs *when* we look (N_eff ≥ 200 interim, ≥ 400 primary); the
  PNR governs *what* is allowed to change between looks.

**Exogenous vs discretionary changes.** The rule above is for *discretionary* changes. A year-long
freeze also meets changes the world forces; these are handled by **kind**, not by a blanket reset:

- **(i) Definition unchanged, world changed → no version bump, observations keep pooling.** The frozen
  artifact still *is* what it was; reality moved. Handled by pre-registered rules:
  - *Universe attrition* — a frozen name is acquired / delisted / halted. The frozen artifact is the
    *eligibility definition*, not a promise every name trades forever: the name drops out (no new
    entries) and any open forward-R window closes per the delisting rule (EXPERIMENT.md §4). Quarterly
    index reconstitution does **not** re-open the frozen list.
  - *Healthy-but-empty feed* — a feed legitimately returns nothing (no qualifying event); the machinery
    is fine and nothing changed.
- **(ii) The artifact itself must change → version bump + new epoch + bridging.** A vendor retires the
  pinned model, a load-bearing feed permanently dies (cf. the FRED/ISM retirement, 1.103), or a bug is
  found in a frozen path. Bump `EXPERIMENT_VERSION` and start a new evaluation period — **and** run a
  *bridging characterisation* so the epochs are comparable rather than merely severed: re-run Gate A on
  the new instrument, and score an overlap sample of decisions on both old and new where feasible.
  **Model deprecation is the most likely trigger over a year** — pin a model with a known availability
  window and treat its forced replacement as a *planned* epoch boundary, not a surprise.

---

## 7. Disposition of pre-PNR (pilot) data

The observations accumulated so far (`experiment_observations.jsonl`, `decisions.jsonl`) are **pilot /
shakedown**, excluded from every primary and secondary test, because they were collected:
(a) under `ADAPTIVE_PROMPT_ENABLED = True` (non-stationary prompt, §15.9);
(b) before the Gate A granularity decision (§7 instrument not finalised);
(c) before the A3.1 eligible-set was settled; and
(d) before `evidence_score_v2` existed.

They are retained (useful for the FDE narrative, Gate A stratification, and power estimation) but are
fenced off — the freeze manifest records the cutoff row counts so the boundary is auditable. If the
observation schema can attribute rows to mode/version (P8), a clean subset *may* later be reclassified,
but the default is exclusion.
