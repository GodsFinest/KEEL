# Roadmap (phase gates)

| Phase | Status | Gate evidence |
|---|---|---|
| 0 — Repository & engineering contract | **done** | docs/ frozen; build green from commit 1; git history per milestone |
| 1 — Exact oracle + baselines + feasibility gate | **done** | `docs/PHASE1_AUDIT_REMEDIATION.md` (v1.1, PASS); historical v1.0 raw report is not part of the current checkout |
| 2 — Production IR + bounded planner | **done** | `src/rt/ir.cpp`, `src/policy.cpp`, runtime IR tests |
| 3 — Independent concrete plan checker | **done** | `src/checker.cpp`, abstract/concrete handover validation |
| 4A — Real controlled memory arena | **done** | `src/rt/arena.cpp`, arena invariant and transfer tests |
| 4B — Numerical CPU executor | **done** | `src/rt/executor.cpp`, independent reference and executor tests |
| 5 — Live two-task handover demo | **done** | `keel_demo`, live overlap and ownership-transfer assertions |
| 6 — Adversarial testing + hardening | **done** | adversarial, malformed-input, failure-atomicity regressions |
| 7 — Real-runtime benchmarking | **done** | `src/rt/bench.cpp`, `experiments/rt_aggregate.py` |
| 8 — Evidence-driven improvement | **done** | frozen experiment methodology and retained research records |
| 9 — Demonstration + reproducibility | **done** | CMake/CTest registration, demo, sanitizer verification |

Authorization history: Phase 0+1 only (2026-11-01 user directive) — superseded
2026-11-01 by the full-runtime mandate (Phases 2–9). Do not push to remote;
local git checkpoints per milestone. The runtime CTest registration is kept in
`CMakeLists.txt`, including the unit, differential, and Python integrity/control
checks.
