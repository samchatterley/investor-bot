# Experiment Log (monitoring)

> **Monitoring only, not a hypothesis test.** This file records weekly descriptive telemetry for the
> study pre-registered in [`EXPERIMENT.md`](EXPERIMENT.md): sample accumulation, Phase-0 gate status,
> and operational health. It is **not** the paper's Results section. Per EXPERIMENT.md section 2.6,
> formal hypothesis tests fire only at the pre-registered N_eff milestones; nothing recorded here is a
> confirmatory result or a headline claim. Entries are appended automatically by the Sunday weekly
> review, newest last.

<!-- entries appended below -->

## 2026-06-21

- Monitoring only, not a hypothesis test. Formal tests fire at the pre-registered N_eff milestones.
- Phase: Phase 0 (pre-data).
- Noise audit (Gate A): not run.
- Power analysis (Gate B): projected underpowered for the live track; scoped as a trend and qualitative layer.
- Effective sample accumulated: N_eff = 0 (next formal test at N_eff >= 200).

## 2026-07-05

- Monitoring only, not a hypothesis test. Formal tests fire at the pre-registered N_eff milestones.
- Phase: Phase 0 (pre-data).
- Noise audit (Gate A): not run.
- Power analysis (Gate B): projected underpowered for the live track; scoped as a trend and qualitative layer.
- Effective sample accumulated: N_eff = 0 (next formal test at N_eff >= 200).

## 2026-07-12

- Monitoring only, not a hypothesis test. Formal tests fire at the pre-registered N_eff milestones.
- Phase: Phase 0 (pre-data).
- Noise audit (Gate A): not run.
- Power analysis (Gate B): projected underpowered for the live track; scoped as a trend and qualitative layer.
- Effective sample accumulated: N_eff = 0 (next formal test at N_eff >= 200).
- Edge anatomy (5d net R, monitoring only): field n=2256 mean=+0.368R; AI picks n=52.
-   conf<=7: n=24 net=+0.368R edge=+0.000R
-   conf=8: n=28 net=+0.647R edge=+0.278R
-   Pre-registered 7->8 trigger: accumulating (need n>=50/bucket; have conf<=7 n=24, conf=8 n=28).
-   extended(rsi>=60): field n=866 +0.265R | AI pick-rate 3.3% edge +0.068R
-   not-extended: field n=1390 +0.433R | AI pick-rate 1.7% edge +0.319R
-   signal pead: field +0.022R | AI picks n=45 +0.582R edge +0.560R
