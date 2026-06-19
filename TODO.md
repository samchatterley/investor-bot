# Pending Work

### In Progress
- [ ] Define the PNR / experiment freeze protocol: freeze manifest (pinned versions/hashes), frozen-vs-mutable classification, governance rule (change to frozen artifact → new experiment_version + new eval period), shakedown-data disposition. Codify as doc + (optional) CI guard.

### Pending
- [ ] PNR prereq — freeze the prompt: set ADAPTIVE_PROMPT_ENABLED=False (§15.9) so Arm 3 is stationary; the self-learning loop becomes a separate pre-registered ablation. Deploys after close.
- [ ] PNR prereq — Gate A (noise audit): run experiment/noise_audit.py on stratified real candidates (5 repeats); decide Arm 3 granularity (coarse+veto vs {-2..+2}); freeze §7 contract or pivot per §9.
- [ ] PNR prereq — Gate B (power) = A7.1: run scripts/phase0_power_analysis.py; decide live track = primary statistical test vs trend+qualitative layer; record decision.
- [ ] PNR prereq — A3.1 parity: define + freeze the eligible-candidate-set filters (add live-only sector gate/cap/churn to backtest + re-validate, OR remove from live) so live eligible set == pre-registered baseline.
- [ ] PNR prereq — controls + ledger: confirm §15.3 negative/positive controls wired (positives detectable, negatives null), and the §13 as-of context ledger enforces provider_seen_at ≤ decision_time − buffer.
- [ ] PNR prereq — version pinning + freeze manifest: add EXPERIMENT_VERSION + hashes of universe / evidence_score_v1 weights / context_packet_v1 / prompt+model+temperature / material-context detectors; record disposition of pre-PNR shakedown observations.
- [ ] Post-PNR-safe ops (deferred, freeze-neutral): logs/cache foldering (option a remainder), cache pruning, run-file naming — confirmed not to touch decisions or logged variables.

