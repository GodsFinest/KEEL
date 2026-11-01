# KEEL — Phase 1 Audit Remediation Report

**Date:** 2026-11-01 · **Scope:** Phase 1 only (no Phase 2 work) · **Model version:** v1.0 → **v1.1**
**Trigger:** independent source-code review findings (user mandate, 2026-11-01).

**New verdict: PASS (conditional improvements noted).** Every reproducible
defect was fixed with regression tests, the formal contract is versioned
(v1.1), the full evaluation was re-run under corrected semantics with new
held-out seeds and corrected cost accounting, and three stronger baselines now
compete with the KEEL planner — one of which (the hybrid) is now the strongest
policy overall and is recommended as the Phase 2 deployment shape. KEEL's core
advantages **survive** the corrected semantics, marginally reduced in rate
(98.0% → 97.6% feasible-acceptance; separation 21.4% → 21.7%; capture
94.8% → 93.7%) and with the tail-cost caveat unchanged.

---

## 1. Finding-by-finding verdicts

### Finding 1 — Asynchronous output reservations — **REPRODUCED, CONFIRMED, FIXED**

- **Reproduction:** the audit's exact counterexample (Bo=10, Bn=9, 1-unit
  releasable init block, two independent 6-unit ops) was encoded as a red test
  (`tests/test_audit_repro.cpp`). Under v1.0 the checker ACCEPTED the trace
  `RELEASE, COMMIT, LAUNCH(A), LAUNCH(B), WAIT(A), WAIT(B), DROP(A)` and the
  oracle emitted a witness for an equivalent plan. Root cause: `memory_of()`
  counted in-flight **scratch only**; the LAUNCH peak check accounted for the
  launched op's outputs but not reservations already held by *other* in-flight
  ops; WAIT materialized outputs with no invariant re-validation. Two 6-unit
  launches could be concurrently in flight against a 9-unit region.
- **Fix (SPEC v1.1 §4):** `usage = Σ resident + Σ_{v∈inflight}(scratch(v) +
  Σ_{o∈O(v)} size(o))`. LAUNCH adds the reservation to usage; WAIT converts it
  into residency (usage drops by the scratch only — holding scratch until the
  completion event is observed is conservative and documented). No new
  unaccounted allocation exists at completion, under any completion ordering.
  The fix lives in the shared model (`src/model.cpp`: `inflight_reserved`)
  and, **independently re-derived**, in the checker (`src/checker.cpp`).
- **Evidence:** the audit trace is now rejected at LAUNCH(B)
  ("op exceeds memory ceiling"); `can_launch(B)` after `LAUNCH(A)` returns
  `OVER_CEILING`; WAIT provably never grows usage; a legal serialized plan
  (LAUNCH A, WAIT, DROP disposable, LAUNCH B, WAIT) is still accepted.
  Permanent regression tests in `tests/test_audit_repro.cpp`,
  `tests/test_gen.cpp` (trap_concurrent_launch).

### Finding 2 — Checker invariant coverage — **CONFIRMED, FIXED**

- **Confirmed:** v1.0 validated the invariant only initially and finally.
- **Fix:** the checker now validates the full memory invariant after EVERY
  action (Run/Launch/Wait/Drop/Release/Commit), with its own reservation
  accounting — no model code imported (independence preserved; D6).
- **Mutation tests** (both phases): preparation-phase overcommit with cleanup
  (LAUNCH, LAUNCH, WAIT, WAIT, DROP before COMMIT) → rejected; post-WAIT
  residency overcommit (12 > 9) with final cleanup → rejected; the legal
  serialization → accepted. All in `tests/test_audit_repro.cpp`.

### Finding 3 — Input and arithmetic validation — **PARTIALLY CONFIRMED, FIXED**

- **Confirmed defects:** (a) negative scratch accepted by the builder (found
  live by the new regression test — it would silently *reduce* usage);
  (b) no request validation (Bo ≤ 0, Bn < 0, Bn ≥ Bo, K < 0 all accepted);
  (c) no id-range checks in `can_drop` / `can_release` / `can_wait`
  (out-of-range ids were UB via shifts); (d) unchecked `+=` chains in the
  checker's usage sum and the generator's `never_drop_peak` / `remaining_work`;
  (e) no structural validation of captured states.
- **Already correct (verified, not assumed):** producer/output consistency
  (single producer, no in-place, no duplicates, orphans, cycles — all enforced
  by `GraphBuilder` since Phase 0); overflow-checked byte sums in
  `tensor_bytes` and `run_core`; checker id-range checks on every action.
- **Fix:** `add_op` rejects scratch < 0; `validate_request` (Bo > 0,
  0 ≤ Bn < Bo, K ≥ 0) enforced by the model and, independently, by the checker;
  id-range guards on every legality predicate; checked arithmetic everywhere —
  **no silent saturation**: checker accounting throws, converted into an
  explicit rejected-witness message (documented contract); `validate_state`
  (residency only from completed producers; no op both done and in flight;
  in-flight ops hold resident inputs and non-resident outputs; released-bytes
  range; invariant) used for frontiers and available to the harness.
- **Evidence:** 14 new regression tests (`tests/test_audit_repro.cpp`).

### Finding 4 — Planning-cost accounting — **CONFIRMED, FIXED**

- **Confirmed:** `keel_plan()` reported outer expansions + only the successful
  candidate's continuation expansions; unsuccessful candidates' continuation
  work vanished; checker workload was untracked; greedy executor steps were
  the only metric for constructive policies and not labeled as incomparable.
- **Fix:** `PolicyOutcome` now carries `outer_expansions`,
  `cont_expansions_total` (summed over ALL continuation searches),
  `candidates_checked`, `checker_invocations`, `checker_us`, `executor_steps`,
  and `planning_nodes = outer + cont_total`. The runner emits all of them per
  row; the aggregator reports them per policy. Executor steps are never summed
  with UCS expansions; cross-family comparisons use wall-clock plus labeled
  fields.
- **Corrected measurements** (organic-feasible accepted, held-out v1.1):
  keel median outer 3 + continuation 37 nodes, 1 candidate check, 1 checker
  call, 22 µs total; guarded_greedy median 5 executor steps (not comparable —
  labeled), 2 checker calls, 3 µs. The v1.0 headline "44 nodes" was an
  undercount of true search work by the discarded candidates' share; the
  corrected median total (~40) happens to be similar because most instances
  succeed on the first candidate, but the tail is now honestly visible
  (cont p90 = 393, max 202,546 nodes).

### Finding 5 — Baseline strength — **CONFIRMED (gap was real), ADDRESSED**

Three policies added (`src/policy.cpp`), all using the same legality rules and
independent checker validation:

| policy | design | organic-feasible accept (v1.1 held-out) |
|---|---|---|
| `prep_greedy` (5A) | guarded greedy + unlock-runs phase (runs ready ops that strictly increase releasable bytes, best ratio first, within K) | **27,411 / 35,348 (77.6%)** |
| `limited_lookahead` (5B) | keel machinery, budget capped at 2,000 prep expansions / 100 candidate checks | **33,171 (93.8%)** |
| `hybrid` | guarded_greedy fast path → keel escalation, work summed | **34,842 (98.6%)** — strongest policy overall |

vs conservative 70.4%, greedy 66.0% (+22.1% FAILED), guarded_greedy 71.2%,
keel 97.6%. Read: (i) prep compute alone (prep_greedy) closes ~21% of the
drop-only gap; (ii) most of KEEL's advantage is available at 1/10th the search
budget (limited_lookahead); (iii) the hybrid dominates keel alone because the
fast path also covers instances where keel's best-first ordering burns budget.
**The honest conclusion for Phase 2: the deployment shape should be the
hybrid; KEEL's search is the escalation stage, not the whole system.**

### Finding 6 — Re-evaluation under corrected semantics — **DONE (v1.1)**

- **Preservation:** v1.0 results untouched:
  `experiments/results/v1.0/{FEASIBILITY_REPORT.md, dev_summary.txt,
  heldout_summary.txt}` + `MANIFEST.sha256` (SHA-256 of the raw JSONL);
  git tag `phase1-model-v1.0` at the pre-audit commit.
- **Versioned corrected run:** SPEC v1.1 (changelog in the SPEC header;
  DECISIONS D12); corrected oracle/checker; 8 policies; original organic and
  adversarial families PLUS `trap_concurrent_launch` (audit finding 6: a
  grid-aligned concurrent-launch oversubscription trap — the concurrent trace
  is exactly the v1.0 defect and is now illegal at the second LAUNCH; only
  serialization survives); **new held-out seeds 71–120** (dev 1–20 re-run for
  iteration only); full work + wall-clock accounting.
- **Results (held-out v1.1, 554,400 rows, 7,700 instances — see §2):** all
  v1.0 headline numbers were recomputed, not assumed. Zero invalid accepts
  survives; the 98.0% becomes 97.6%; separation 21.4% → 21.7%; capture
  94.8% → 93.7%. On the concurrent-launch trap, drop-only guarded greedy
  accepts 72/204 while every search policy accepts 204/204.
- **Artifacts:** `experiments/results/raw/{dev_v1_1,heldout_v1_1}.jsonl`,
  `heldout_v1_1.jsonl.gz` (reviewable compressed artifact),
  `MANIFEST_v1_1.sha256`, `dev_summary_v1_1.txt`, `heldout_summary_v1_1.txt`.

### Finding 7 — Verification checklist — **SATISFIED (evidence per item)**

| requirement | evidence |
|---|---|
| minimal async-overcommit counterexample rejected | `test_audit_repro.cpp` (red → green) |
| invariant holds after every modeled action | checker per-action validation; randomized differential |
| oracle ↔ checker agreement on valid witnesses | every witness in every run checker-validated (0 failures); `test_oracle.cpp` |
| randomized differential + mutation tests pass | `test_random_differential.cpp`: model↔checker trace agreement + mutation-equivalence across all 12 families × 3 distributions (46 test cases, 15,262 assertions total green) |
| corrected baselines run successfully | 8 policies × full grid, zero invalid accepts |
| experiments + aggregates reproducible | §4 commands; manifests with SHA-256 |
| sanitizers | ASan+UBSan build: full test suite (15,262 assertions) + experiment slice (33,264 rows) clean, exit 0 |
| no known correctness-critical defects in Phase 1 scope | findings 1–7 resolved; none deferred |

## 2. Versioned corrected results (v1.1, held-out seeds 71–120)

Oracle labels: organic 35,348/64,800 FEASIBLE (54.5%; 28 UNKNOWN = 0.04%);
adversarial 1,348/4,500 (30.0%). Policy behavior on organic-feasible:

| policy | ACCEPT | REJECT | FAILED | UNKNOWN |
|---|---|---|---|---|
| conservative | 24,864 (70.4%) | 10,484 | — | — |
| greedy | 23,340 (66.0%) | 4,212 | **7,796 (22.1%)** | — |
| guarded_greedy | 25,172 (71.2%) | 10,176 | — | — |
| prep_greedy | 27,411 (77.6%) | 7,937 | — | — |
| limited_lookahead | 33,171 (93.8%) | 0 | — | 2,177 |
| **hybrid** | **34,842 (98.6%)** | 0 | — | 506 |
| keel | 34,501 (97.6%) | 0 | — | 847 |
| oracle | 35,348 | — | — | — |

On oracle-INFEASIBLE instances: **zero accepts by any policy** (greedy's 8,876
FAILED are self-destructive commits, reported not hidden). Defect audit
(accepted-but-invalid), corpus-wide incl. adversarial: **0 for all 8
policies**. H1 separation: 21.7% missed by both drop-only policies (all 7
families ≥ 10.6%); keel captures 93.7% (per-family 84.5–100%); hybrid captures
more. Async slice: keel 96.8%, hybrid 97.2% of feasible vs conservative 59.8%.
Adversarial (separate reporting): keel/limited/hybrid 1,348/1,348;
guarded 1,035; greedy 819 (+513 FAILED). Concurrent-launch trap: 204 feasible;
keel/limited/hybrid 204/204 vs guarded 72/204.

**v1.0 → v1.1 deltas (honest):** feasible-acceptance keel 98.0% → 97.6% (the
reservation semantics tighten async launches; sequential results are
bit-identical); separation 21.4% → 21.7%; capture 94.8% → 93.7%. The
conclusion is unchanged in direction and slightly strengthened in mechanism
(the hybrid now dominates, and the concurrent-launch trap shows planning
achieving what no ladder can).

## 3. Uncertainty and failure analysis

- The oracle's UNKNOWN rate is 0.04% organic (async-heavy corners at large K);
  those instances are excluded from feasibility rates and labeled.
- keel's residual gaps: 847 UNKNOWN on feasible organic instances (2.4%);
  0 REJECT. limited_lookahead's larger UNKNOWN tail (6.2%) is the price of
  the tight budget — reported, not averaged away.
- keel's worst-case wall-clock remains unbounded-ish by design (budget-limited
  cases reach ~0.4 s in v1.0; the v1.1 max on accepted instances is 108 ms
  with the corrected accounting) — the hybrid caps this in practice.
- All results remain on synthetic DAGs (≤ 9 ops, abstract byte units); the
  model still abstracts layout (fixed shrinking region, D1); executor steps
  and UCS node counts are explicitly not comparable across policy families.
- Sanitizer coverage: the full test suite (15,262 assertions) and an
  experiment slice (33,264 rows) run clean under ASan+UBSan on macOS arm64
  (AppleClang 17). The dominance differential under sanitizers was attempted
  and descoped for runtime (ASan slows the exhaustive double-search by orders
  of magnitude); it is not part of the frozen verification set — the
  differential itself passes unsanitized corpus-wide. TSan was unattempted for
  the single-threaded Phase 1 audit; runtime-threading sanitizer status belongs
  to the later full-runtime verification rather than this historical record.

## 4. Reproduction

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
ctest --test-dir build --output-on-failure        # 46 cases, 15,262 assertions
./build/keel_domcheck 20                          # dominance differential: 0 mismatches
./build/keel_experiment --seed-lo 1  --seed-hi 20 --out experiments/results/raw/dev_v1_1.jsonl
./build/keel_experiment --seed-lo 71 --seed-hi 120 --held-out \
    --out experiments/results/raw/heldout_v1_1.jsonl
python3 experiments/aggregate.py experiments/results/raw/heldout_v1_1.jsonl
sha256sum -c experiments/results/MANIFEST_v1_1.sha256
cmake -S . -B build-san -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" && \
  cmake --build build-san && ./build-san/keel_tests
```

## 5. Verdict

**PASS.** The Phase 1 foundation is sound under independently reviewed and
versioned semantics: the critical async defect is fixed and permanently
pinned; the checker validates every action and agrees with the model on
randomized traces and mutations; input validation has no known gaps; cost
accounting is complete and normalized; stronger baselines exist and one of
them (hybrid) is the recommended Phase 2 shape. The go/no-go question — does
continuation-aware planning beat competent simpler algorithms? — is answered
YES with the hybrid as the deployment form and KEEL's bounded search as its
escalation stage; the margin shrank slightly under corrected semantics and
that shrinkage is reported rather than smoothed. Phase 2 remains **not
authorized** until the user reviews this report.
