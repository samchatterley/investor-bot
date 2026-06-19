# Pending Work

### Pending
- [ ] PNR prereq P2 — Gate A (noise audit): run experiment/noise_audit.py on the to-be-frozen model/temperature, stratified real candidates, 5 repeats; decide Arm 3 granularity (§7); record noise floor in manifest; or pivot per §9.
- [ ] PNR prereq P3 — Gate B (power) = A7.1: run scripts/phase0_power_analysis.py; decide live track = primary statistical test vs trend+qualitative layer; record decision.
- [ ] PNR prereq P4 — A3.1 parity: define + freeze the eligible-candidate-set filters (add live-only sector gate/cap/churn to backtest + re-validate, OR remove from live) so live eligible set == pre-registered baseline.
- [ ] PNR prereq P5 — fit + freeze evidence_score_v2 (train/validation/holdout) per §15.7 (full-gauntlet: standable benchmark must exist at t0).
- [ ] PNR prereq P6/P7 — controls + ledger: confirm §15.3 negative/positive controls wired (positives detectable, negatives null), and the §13 as-of context ledger enforces provider_seen_at ≤ decision_time − buffer.
- [ ] PNR prereq P8 — stamp every observation with experiment_version + adaptive_prompt mode (§15.9). NOTE: main.py:1798/1806 already log adaptive_prompt; confirm experiment_version is added too.
- [ ] PNR prereq P9 — freeze manifest script + CI guard: value-hash data artifacts, golden-fixture-hash code-behaviour artifacts; guard fails build if a frozen pin drifts without an EXPERIMENT_VERSION bump.
- [ ] PNR prereq P11 — A10.1 cost-model realism sanity check (modeled vs realized fills) before freezing the cost model into forward-R.
- [ ] PNR prereq P12 — A8.1 ET-anchoring: audit experiment decision paths for date.today(); switch any to today_et() to avoid the late-Oct DST bug.
- [ ] Cleanup (non-blocking, freeze-neutral): silence benign numpy RuntimeWarning at sector_data.py:251 via np.errstate (the intentional 0/0→NaN skip path leaks a warning).
- [ ] Review GitHub Dependabot moderate vuln (#9) on the repo — separate from PNR work.
- [ ] Post-PNR-safe ops (deferred, freeze-neutral): logs/cache foldering, cache pruning, run-file naming.

