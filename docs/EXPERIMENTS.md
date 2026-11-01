# Phase 1 Experiment Plan — PRE-DECLARED (frozen 2026-11-01)

This file is written **before** any evaluation run. After the held-out run
starts, no success criterion, metric, grid entry, or gate below may change.
Anything discovered later goes into docs/DECISIONS.md as a proposed v2 for user
review — never as a silent edit.

## Research question (user-fixed)
Can a bounded, continuation-aware memory handover planner deliver meaningful
advantages over strong, practical memory-reclamation policies, at a justified
planning cost?

## Hypotheses (pre-declared)
- **H1 (value of planned prep compute):** there exist non-adversarial families
  where drop-only policies (conservative, guarded_greedy) REJECT a request the
  oracle proves FEASIBLE with prep compute > 0, at a nontrivial rate.
- **H2 (bounded planner quality):** `keel` at Phase 1 budgets accepts ≥ 90% of
  oracle-FEASIBLE held-out instances on non-adversarial families and emits
  UNKNOWN (never false INFEASIBLE) when it fails.
- **H3 (necessity of guarding):** `greedy` produces accepted-but-FAILED traces
  at a nonzero rate, concentrated on adversarial families.
- **H4 (justified cost):** `keel` planning node counts are within ~10× the
  validation-only policies' effort on instances where it accepts.

## Go/no-go gate (pre-declared; user approves any change)
Proceed to Phase 2 only if, **on held-out data**:
1. H1 rate ≥ 10% of pressure instances in ≥ 2 non-adversarial families
   (a "pressure instance" = Bn < Bo and the frontier holds data whose naive
   surrender is nontrivial), AND
2. `keel` captures ≥ 2/3 of those oracle-feasible instances that drop-only
   policies miss, at planning cost within the H4 envelope, AND
3. independent validation shows zero accepted-but-invalid plans for any
   validated-acceptance policy across the whole held-out set.
If the gate fails, the deliverable is a negative FEASIBILITY_REPORT.md with the
strongest honest characterization of *when* planning would help, and the
recommendation to stop or redirect. **The gate is not moved after results.**

## Grid (frozen)
- Families (non-adversarial): `chain`, `diamond`, `shared_ancestor`,
  `fork_join`, `residual`, `big_workspace`, `heterogeneous`.
  Adversarial (reported separately, never pooled): `trap_recompute`,
  `trap_nonreplayable`, `trap_memory`, `prep_unlock`.
- Cost distributions: `uniform`, `heavytail` (≈ log-uniform costs/sizes),
  `decorrelated` (large tensors on cheap ops and vice versa).
- Seeds: **dev 1–20** (iteration; may be inspected freely), **held-out 21–70**
  (run once, frozen grid, results untouched by tuning).
- Frontiers per (family, seed, distribution): 2 sequential live states via
  seeded legal prefix execution under Bo. Async slice (order-abstract §5):
  1 async frontier per instance on `chain` and `fork_join` only, to exercise
  in-flight protection without exploding the grid.
- Grants: `Bn ∈ {0.50, 0.65, 0.80} × Bo`; prep budget
  `K ∈ {0, 10%, 25%} × remaining_work(S0)` (work = Σ cost of not-yet-done ops).
- Policies: `conservative`, `greedy`, `guarded_greedy`, `keel`,
  `oracle_assisted` (ground truth) — per docs/SPEC.md §10–§12 budgets.

## Metrics and reporting (fixed)
Per docs/SPEC.md §11. Reporting: per (family, Bn, K, distribution) tables of
accept/valid/fail rates, bytes returned, prep/recompute/total compute,
planning nodes/time; **full distributions** (min/median/p90/max) in the report;
raw JSONL per instance is generated under the ignored
`experiments/results/raw/` directory for local analysis. Adversarial
families in their own section. Every accepted witness checker-validated;
validation failures are bugs with regression tests, not data points to discard.

## Falsification duties
- If the oracle proves more than trivial numbers of requests INFEASIBLE under
  every K, report that the interesting regime is narrower than assumed.
- If guarded_greedy matches the oracle's acceptance everywhere off the
  adversarial families, report that continuation-aware *search* adds nothing
  over validate-then-act, and recommend against building the full runtime.

---

## v1.1 addendum (2026-11-01, audit remediation)

The frozen v1.0 section above is preserved untouched; its historical results
are summarized in `docs/PHASE1_AUDIT_REMEDIATION.md` (the raw v1.0 directory is
not part of the current checkout). The corrected
evaluation (SPEC v1.1) reuses the same grid, metrics, and budgets with these
changes, declared before the v1.1 held-out run:

1. Model: in-flight ops reserve outputs + scratch until completion
   (docs/SPEC.md v1.1 changelog; DECISIONS D12).
2. Policies: `prep_greedy`, `limited_lookahead`, `hybrid` added (finding 5);
   budgets per docs/PHASE1_AUDIT_REMEDIATION.md; work accounting extended
   (outer/continuation/candidate/checker fields; executor steps labeled
   non-comparable with UCS expansions).
3. Families: `trap_concurrent_launch` added to the declared adversarial set.
4. Held-out seeds: **71-120** (fresh, never used for tuning); dev 1-20 re-run
   under v1.1 for iteration only.
5. Raw artifacts: `experiments/results/raw/{dev_v1_1,heldout_v1_1}.jsonl`
   (+ optional `.gz`) are local generated outputs and are intentionally not
   part of the source checkout.
Results and the go/no-go verdict: `docs/PHASE1_AUDIT_REMEDIATION.md` (PASS).
