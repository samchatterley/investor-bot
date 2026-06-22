# Pending Work

### In Progress
- [ ] Phase 1 logs cleanup [IN PROGRESS]: repoint 19+ regenerable API caches to logs/caching/ (where user moved them) so Monday's prefetch stays warm. Inline LOG_DIR/caching joins; config.CACHE_DIR ensures dir. Commit + restart this weekend.

### Pending
- [ ] BOTH Phase-0 gates cleared → experiment is GO. Proceed to freeze the PNR with veto/risk (H1, ~1yr) + IC (H2, ~quarter via B+A) as co-primaries, after prereqs P4-P12.
- [ ] Phase 2 logs cleanup [DEFERRED, no deadline]: fold live STATE (DB, baselines, decisions/audit/observations records, regime_state, signal_stats, scheduler.log/pid, .lock, positions_meta, 2026/Week run records) into state/records/run/. The risky half (F1-class if buggy) — dedicated pass + migration, NOT end-of-session.
- [ ] v1.4 amendment (B+A) DRAFTED in EXPERIMENT.md §15.10 — needs your review; not yet committed.
- [ ] First-weeks measurement: realised post-market-adjustment same-day ICC (confirms IC ~1 quarter vs ~3-4 months).
- [ ] PNR prereq P4 — A3.1 parity (veto/selection eligible set).
- [ ] PNR prereq P6/P7 — controls + as-of ledger.
- [ ] PNR prereq P8 — stamp observations with experiment_version + adaptive_prompt.
- [ ] PNR prereq P9 — freeze manifest script + CI guard (incl. market-relative IC metric, broadened population, Arm-3 budget, temp=0.0, coarse contract).
- [ ] PNR prereq P11/P12 — A10.1 cost realism; A8.1 ET-anchoring.
- [ ] Build Lever A+B IC machinery (post-Gate-A, pre-freeze): score material-context layer ~50-100/day; market-relative forward R / day-FE.
- [ ] P5/build_dataset — OPTIONAL (veto controls for v1, not v2).
- [ ] Signal-book prune finding (breadth_thrust) — separate research track.

