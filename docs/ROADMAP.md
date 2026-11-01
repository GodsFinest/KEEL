# Roadmap (phase gates)

| Phase | Status | Gate evidence |
|---|---|---|
| 0 — Repository & engineering contract | **done** | docs/ frozen; build green from commit 1; git history per milestone |
| 1 — Exact oracle + baselines + feasibility gate | **done** | `experiments/results/FEASIBILITY_REPORT.md` (v1.0) + `docs/PHASE1_AUDIT_REMEDIATION.md` (v1.1, PASS); 46 test cases green from clean checkout |
| 2 — Production IR + bounded planner | **authorized — in progress** | full-runtime mandate 2026-11-01 |
| 3 — Independent concrete plan checker | **authorized — in progress** | mandate Part III |
| 4A — Real controlled memory arena | **authorized — in progress** | mandate Part IV |
| 4B — Numerical CPU executor | **authorized — in progress** | mandate Part IV |
| 5 — Live two-task handover demo | **authorized — in progress** | mandate Part V (13 observable properties) |
| 6 — Adversarial testing + hardening | **authorized** | mandate Part VI |
| 7 — Real-runtime benchmarking | **authorized** | mandate Part VII |
| 8 — Evidence-driven improvement | **authorized** | mandate Part VIII |
| 9 — Demonstration + reproducibility | **authorized** | mandate Part IX |

Authorization history: Phase 0+1 only (2026-11-01 user directive) — superseded
2026-11-01 by the full-runtime mandate (Phases 2–9). Do not push to remote;
local git checkpoints per milestone. Test-registration audit (mandate §3):
verified at e126b62 — all 8 test TUs registered, 46/46 cases discovered and
executed from a clean checkout, 15,262 assertions, 0 failures.
