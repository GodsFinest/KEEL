# Status

## Current phase
Phases 0-9: full CPU runtime **AUDITED AND VERIFIED** (full-runtime mandate
2026-11-01). See the runtime correctness audit below and
docs/ARCHITECTURE.md / docs/INVARIANTS.md.

## Audit remediation (2026-11-01)
Independent review findings 1-7 investigated, reproduced where real, fixed,
and re-evaluated under SPEC v1.1:
- F1 async output reservations: REPRODUCED + CONFIRMED + FIXED (in-flight ops
  reserve outputs+scratch until WAIT); the audit counterexample is a
  permanent regression test.
- F2 checker per-action invariant validation + mutation tests: done.
- F3 validation: negative scratch found live and fixed; request/state/range
  validation added; no silent saturation; checked arithmetic everywhere.
- F4 planning-cost accounting: all continuation checks counted; checker
  workload tracked; executor steps labeled non-comparable.
- F5 stronger baselines: prep_greedy (77.6%), limited_lookahead (93.8%),
  hybrid (98.6% — strongest; recommended Phase 2 shape).
- F6 v1.1 re-evaluation: new held-out seeds 71-120, 554,400 rows, zero
  invalid/false accepts corpus-wide; v1.0 preserved with SHA-256 manifest;
  trap_concurrent_launch added (204 feasible; search policies 204/204 vs
  guarded 72/204).
- F7 sanitizers (ASan/UBSan) clean on tests + experiment slice; randomized
  model-checker differential + mutation equivalence green.
Full detail: docs/PHASE1_AUDIT_REMEDIATION.md.

## Verdict of the Phase 1 gate
**GO** (all three pre-declared criteria pass on held-out data): planned,
continuation-aware handovers are missed by every drop-only policy on 21.4% of
oracle-feasible organic instances (≥10% in all 7 families), keel captures
94.8% of those within the planning-cost envelope at median, and there are zero
invalid accepts and zero false accepts corpus-wide. Negative findings are
preserved in `docs/PHASE1_AUDIT_REMEDIATION.md` (§7).

## Done (evidence)
- 2026-11-01 Phase 0: docs frozen (SPEC v1.0, EXPERIMENTS pre-declared,
  RESEARCH ledger incl. Coop correction, DECISIONS D1–D11), AGENTS.md,
  doctest vendored, buildable from commit 1.
- 2026-11-01 Model + independent checker + mutation battery (1069 assertions,
  34 test cases, all green).
- 2026-11-01 Exact oracle: UCS + memoized keys, FEASIBLE/INFEASIBLE/UNKNOWN,
  checker-validated witnesses; frontier-capture K semantics.
- 2026-11-01 Generators: 7 organic families × 3 distributions + 4 declared
  traps; sequential + async frontier sampling. **Release amplification
  discovered and pinned** (D10).
- 2026-11-01 Policies: conservative / greedy / guarded_greedy / keel / oracle;
  flagship separations pinned in unit tests.
- 2026-11-01 Dominance pruning: two unsound rules caught by the differential
  harness, sound rule validated corpus-wide (24,120 instances, 0 mismatches)
  (D11 + addendum).
- 2026-11-01 Dev phase (seeds 1–20): three planner defects found and fixed
  with regression tests (async WAIT vocabulary ×2, keel WAIT/RUN independence).
- 2026-11-01 **Held-out run of record (seeds 21–70, frozen grid): 342,000
  rows / 7,600 instances; gate criteria PASS; results summarized in the audit
  record.**

## Reproduction
See the reproduction commands in `docs/PHASE1_AUDIT_REMEDIATION.md`. Raw JSONL
outputs are generated locally under the ignored `experiments/results/raw/`
directory and are not source-controlled.

## Blockers
- None. Awaiting user review of the completed runtime audit.

## Full-runtime acceptance audit (2026-11-01)

The final source audit read the complete runtime path before editing: typed IR
validation → guarded abstract conversion → frontier capture/planning and
independent checking → concrete preparation → held physical extents → atomic
COMMIT → post-COMMIT continuation, with worker completion and two-task arena
access traced end to end. Confirmed defects were reproduced with permanent
regressions; the Phase 1 async-reservation finding was re-verified as already
fixed rather than rewritten.

Final runtime remediation evidence:

- BlockArena validates power-of-two alignment, initializes its aligned deleter
  correctly, checks size arithmetic, and makes claims, frees, held transfers,
  and failed dispatch reservations failure-atomic. Arena metadata is
  mutex-protected for concurrent task controllers.
- RuntimeGraph rejects unsupported dtypes, invalid arity, incompatible or
  misdeclared shapes, undersized scratch, and overflowing byte products.
  Elementwise execution uses trailing-dimension broadcasting.
- In-flight inputs and outputs are both protected. Invalid in-flight output
  pointers cannot be exposed; worker failures are surfaced to the controller.
  The plain-vector reference executor uses separate arithmetic code and the
  test suite includes hand-calculated numerical checks.
- DROP and RELEASE are distinct. RELEASE holds the exact physical extent;
  owner bits change only at COMMIT. A cloned-arena continuation preflight
  rejects abstractly feasible but physically fragmented continuations before
  transfer.
- keel_demo and test_rt_handover use a latch, not timing luck: B's output
  allocation is observed inside A's transferred blocks while A is blocked at
  an unfinished worker completion. Both outputs are independently checked.
- keel_bench now emits committed, a_ok, b_ok, a_correct, b_correct,
  joint_success, overlap_observed, b_used_transferred, and separate timing
  fields. The aggregator reports committed and joint-success rates separately
  and preserves failed rows.

| Contract item | Evidence |
|---|---|
| Clean checkout builds; all tests registered+executed | current CTest: 4 registered tests; `keel_tests` runs 100 cases / 18,286 assertions |
| Exact oracle retained | src/oracle.cpp; used as benchmark reference instrument |
| Bounded practical planner + hybrid evaluated | keel_plan/hybrid + retained Phase 8 decisions (`docs/DECISIONS.md`) |
| Valid complete continuations; correct status classification | hybrid/keel statuses; UNKNOWN never converted to INFEASIBLE (test_rt_adversarial) |
| Every accepted handover checker-validated; per-action invariants; malformed plans rejected | perform_handover VALIDATED gate; Phase 1 checker suites |
| Stale plans cannot execute silently | control-thread exclusivity + per-action re-checking + version guard (test_rt_adversarial) |
| Real bounded backing storage; ids/owners/generations; physical transfer; fragmentation accounted; no dual ownership | rt::BlockArena + verify() (test_rt_arena) |
| Real numerical DAGs; workers+completion events; reservations; recompute; independent reference agreement | test_rt_executor, including hand-calculated broadcast/matmul values; 18,286 assertions |
| Live handover: request mid-flight; valid reduction; B uses transferred blocks while A remains unfinished; A/B finish under their grants; invalid handovers rejected | test_rt_handover, test_rt_adversarial, keel_demo (14 live properties) |
| Adversarial + sanitizers | 100-case suite; ASan/UBSan runtime tests + demo and full TSan suite + demo pass |
| Strong baselines compared; raw data; honest joint failures and timing | `keel_bench` + `experiments/rt_aggregate.py`; generated rows include joint-success and overlap/reuse metrics |
| Phase 8 evidence-driven optimization | retained in `docs/DECISIONS.md`; fragmentation remains documented as negative/future work |
| README accurate; demo reproducible; limitations explicit | README.md (honest labels), docs/INVARIANTS.md (model-reality boundary) |

Known honest limitations: fragmentation can still cause a normal execution
stall, and release selection is not yet allocator-aware; the concrete
handover preflight rejects such a continuation rather than claiming it is
feasible. The abstract planner/checker/oracle remain capped at 64 ops/tensors.
The synchronous protocol has no wall-clock expiry; a caller-driven
pre-COMMIT cancellation is reported as EXPIRED and rolls back staging. A
wall-clock asynchronous request protocol remains out of scope. TSan is
available and passed for the current concurrent runtime path.
