# KEEL Phase 1 — Formal Execution Contract

**Version: v1.1 (2026-11-01) — audit remediation revision.**
v1.0 (frozen 2026-11-01) governed the original Phase 1 experiments; its
results are preserved under `experiments/results/` and the tag
`phase1-model-v1.0`. v1.1 changes ONLY the in-flight memory semantics and
adds input/state validation (changelog at the bottom). Sequential-mode
semantics and all experiment methodology are unchanged. Changes were NOT
silent: see docs/DECISIONS.md D12 and docs/PHASE1_AUDIT_REMEDIATION.md.

This document is normative. The checker is written from this document, not from
the planner's code. Any change to semantics requires a version bump and a note
in docs/DECISIONS.md.

## 1. Scope of the model

Phase 1 models a **single lender task** whose live state is captured mid-run,
plus the abstract fact that released bytes become usable by a co-tenant. The
co-tenant is not simulated: the contract is that released bytes are gone from
the lender's world forever and are allocatable elsewhere. Modeled feasibility is
a statement about the action model below, **not** a real-time guarantee and not
a claim about memory KEEL does not control (process RSS, driver, GPU).

Restrictions (explicit, enforced by construction of the IR): known DAG
topology; known tensor sizes; immutable values; deterministic, repeatable ops;
declared scratch; no mutation, no aliases, no control flow, no hidden
allocations. Non-replayable source ops model captured external inputs.

## 2. IR

A graph `G = (V ops, T tensors, Q, Pinned)`:

- Each tensor `t ∈ T` has `size(t)` (bytes, > 0) and exactly one producer op
  `producer(t) ∈ V` (source ops produce captured external inputs).
- Each op `v ∈ V` has:
  - inputs `I(v) ⊆ T`, outputs `O(v) ⊆ T` (disjoint across ops; every tensor
    has exactly one producer);
  - `scratch(v)` bytes reserved while `v` executes;
  - `cost(v)` ≥ 1 work units (the unit of K and of all compute metrics);
  - `replayable(v) ∈ {yes, no}`. A `no` op (source) may execute **at most
    once** over the life of the task.
- `Q ⊆ T` — required outputs: tensors that must be resident in the goal state.
- `Pinned ⊆ T` — never droppable (empty in all experiments; exercised in tests).

Derived: `deps(v) = producer ops of I(v)`; a tensor is *available* iff resident.

## 3. Live state

`S = (done, resident, inflight, released, prep_used, phase)` where

- `done ⊆ V` — ops that have completed ≥ 1 time.
- `resident ⊆ T` — tensors whose bytes the task currently owns.
- `inflight ⊆ V` — launched, not completed (empty in sequential mode).
- `released` — set of tensors whose current-or-former held block has been
  physically transferred away; `released_bytes(S) = Σ_{t∈released_keys} size(t)`
  (see §5 for the exact bookkeeping; recomputation after release occupies fresh
  bytes).
- `prep_used` — compute charged during the PREPARING phase (frozen at COMMIT).
- `phase ∈ {PREPARING, CONTINUING}`; `committed` ≡ `phase = CONTINUING`.

Mandate state-tuple mapping: `G = V \ done` (remaining graph), `R = resident`,
`P = Pinned`, `F = inflight`, `Q = Q`. `A` (physical allocation layout) is
abstracted in Phase 1 to byte accounting plus the released set; per-block
offsets/generations are deferred to the Phase 4 arena and are **not** claimed
here.

## 4. Memory accounting (the soundness-critical rule)

```
usage(S) = Σ_{t ∈ resident} size(t)
         + Σ_{v ∈ inflight} ( scratch(v) + Σ_{o ∈ O(v)} size(o) )   # v1.1
ceiling(S) = Bo − released_bytes(S)          # in force in BOTH phases
INVARIANT (every reachable state, validated after EVERY action): usage(S) ≤ ceiling(S)
```

**v1.1 change:** an in-flight operation reserves its output bytes as well as
its scratch, from LAUNCH until its completion is observed at WAIT. Under
v1.0 the launch-time peak check accounted for the launched op's outputs but
not for reservations already held by *other* in-flight ops, and WAIT
materialized outputs without re-validating the invariant — permitting
concurrent launches whose collective reservations exceed the region
(audit finding 1; see docs/PHASE1_AUDIT_REMEDIATION.md). Under v1.1 two
6-unit launches against a 9-unit region cannot both be in flight. WAIT
converts the reservation into residency (usage drops by the scratch only —
holding scratch until the completion event is observed is deliberately
conservative), so no new unaccounted allocation exists at completion, under
any completion ordering. Sequential mode (the default; inflight ≡ ∅) is
bit-for-bit unchanged.

Rationale (fixed-region physical model): the task owns a Bo-byte arena region.
RELEASE physically hands a held block to the co-tenant; the region shrinks by
the released bytes permanently — before and after COMMIT. COMMIT(Bn) requires
`released ≥ Bo − Bn`, which makes `ceiling ≤ Bn` from acknowledgment onward, so
every post-commit footprint fits the new grant (mandate conditions 4/5).
Over-returning (`released > Bo − Bn`) shrinks the ceiling further and is
allowed; the surplus is measured as extra returned bytes. Released bytes are
never regained; recomputation after release occupies bytes still inside the
shrinking region. This is deliberately conservative: Phase 1 has no layout, so
"cede a slice that happens to be free" is not expressible — ceded bytes must
come from blocks the task actually held and emptied. Documented as an
abstraction, revisited in Phase 4 (real arena) where block identity exists.

All sizes/offsets use checked 64-bit arithmetic; overflow aborts planning with
an explicit error status.

## 5. Actions and legality

Grants: `Bo` (in force during PREPARING), `Bn < Bo` (in force after COMMIT).
Preparation budget `K` (work units): `prep_used ≤ K` must hold in PREPARING;
K bounds only compute charged **after the handover capture** — capture resets
`prep_used` to 0, since work executed before the request existed was paid under
the old regime; CONTINUING compute is unconstrained (it is a measured cost,
not a budget).

- **RUN(v)** (sequential execution; also the async LAUNCH's core check):
  - legality: `v ∉ inflight`; `I(v) ⊆ resident`; no output of `v` is resident;
    `¬done(v) ∨ replayable(v)`; and
    `usage' = usage + Σ_{o∈O(v)} size(o) + scratch(v) ≤ ceiling`.
    (Inputs stay resident through the op — conservative, sound peak accounting.)
  - effect: `done += v`; `resident += O(v)`; compute `cost(v)` charged (against
    K in PREPARING: `prep_used += cost(v) ≤ K`).
- **LAUNCH(v)** (async mode only): same legality check as RUN — the check now
  includes all pre-existing in-flight reservations (v1.1 §4). Effect:
  `inflight += v`; compute charged; inputs of `v` become **protected**
  (§5 DROP/RELEASE).
- **WAIT(v)** (async mode only): requires `v ∈ inflight`. Effect:
  `done += v; resident += O(v); inflight -= v`. The reservation converts into
  residency; usage never grows at WAIT, under any completion ordering. The model is
  completion-**order**-abstract: WAIT may pick any inflight op. This models any
  real completion order without wall-clock semantics. The future runtime (Phase
  4) enforces real completion events; nothing here claims to.
- **DROP(t)**: requires `t ∈ resident`; `t ∉ Pinned`; `t` is not an input of any
  inflight op. Effect: `resident -= t`. (Block returns to the task's own free
  pool; recomputation of `t` later costs bytes + compute again.)
- **RELEASE(t)**: requires `t ∈ resident` (a held, live block is surrendered);
  `t` not an input of any inflight op; `t ∉ Pinned`. Effect:
  `resident -= t`; `released_bytes += size(t)`. Legal only in PREPARING
  (Phase 1 models exactly one handover; further handovers are out of scope).
  **This is the only action that transfers storage.** DROP alone never does.
- **COMMIT(Bn)**: requires `released_bytes ≥ Bo − Bn` (the co-tenant physically
  received at least the promised reduction) and `phase = PREPARING`. Effect:
  `phase = CONTINUING`; `prep_used` frozen. By the invariant,
  `usage ≤ Bn − released ≤ Bn` afterwards (mandate conditions 4/5).
- Goal state: `phase = CONTINUING ∧ inflight = ∅ ∧ Q ⊆ resident`.
  (Every `q ∈ Q` resident implies its producer is done.)

Failure/expiry semantics: every action is individually legal and leaves a state
that is legal under the currently in-force grant. Therefore an aborted or
expired handover — at any point — leaves the task able to continue under Bo.
This is a theorem of the action model, tested in the suite (the checker rejects
any trace that violates it; policies exploit it by simply stopping).

## 6. Recomputation semantics

Values are immutable and deterministic: re-running a replayable op with all
inputs resident reproduces byte-identical outputs (Phase 1 asserts this
abstractly; Phase 4 verifies numerically against a reference executor).
Consequences: (a) a dropped tensor `t` is recoverable iff `replayable(producer(t))`
and its inputs are (recursively) recoverable; (b) dropping the output of a
non-replayable source op that has already executed loses that value permanently;
(c) RUN on an op with any output already resident is illegal (no partial
re-materialization, no refresh). The oracle does not reason with a compiled
"recoverable" predicate — it searches; these consequences are emergent.

## 7. The planning queries

Given `(G, S0, Bo, Bn, K)`:

- **Transition feasibility**: does there exist a legal action sequence
  `τ = (prep actions…) , COMMIT(Bn), σ = (continuation…)` with `σ` ending in a
  goal state, prep compute ≤ K, and every state satisfying §4/§5?
- The **oracle** answers this exactly by uniform-cost search (edge cost =
  `cost(v)` of RUN/LAUNCH, 0 for all other actions) over reachable states with
  memoization, returning a concrete witness (full action sequence), the optimal
  total compute under the search's complete exploration, and one of:
  - `FEASIBLE(witness)` — found; total compute is minimal (UCS completeness);
  - `INFEASIBLE` — the reachable state space was exhausted within K; proof;
  - `UNKNOWN` — expansion/node cap hit before exhaustion; **no claim**.

## 8. Memoization correctness

Canonical key: `(phase, done, resident, inflight, min(released_bytes, Bo),
min(prep_used, K))`. Two states with equal keys have identical legal futures:
every legality rule in §5 depends only on key components (usage is derived from
resident + inflight; ceiling from phase, Bo/Bn, released_bytes; K from
prep_used; goal from phase/resident/inflight). Tensor *identity* of released
blocks does not affect feasibility, only logging. `prep_used` is capped at K
because states beyond K are equivalent (no further prep compute allowed; in
CONTINUING prep_used is frozen and irrelevant).

## 9. Dominance pruning (opt-in, validated, never trusted by default)

State A dominates B (same phase) if `done_A ⊇ done_B` (every extra done op
replayable) `∧ resident_A = resident_B ∧ released_A = released_B ∧
prep_used_A ≤ prep_used_B ∧ inflight_A = inflight_B = ∅`. Every action legal
for B is legal for A with identical memory semantics (same residency, same
ceiling, same COMMIT threshold), so A's expansion covers B's future at ≤ cost.

**Why the subset relation is NOT sound here** (two implementation-validated
lessons, docs/DECISIONS.md D11):
1. *Released is two-dimensional.* More released helps COMMIT's threshold but
   shrinks the ceiling under which the mimic replays B's sequence; both
   monotone directions failed differential agreement.
2. *Superset residency breaks completeness structurally.* Coverage of a
   dominated state B requires expanding the dominator's drop-closure state
   (A minus its extra residents) — but that state is itself dominated by A
   (it is A's own drop-closure), so naive pruning deletes exactly the states
   the mimic argument needs. Free-drop mimicking and pop-time dominance
   pruning are therefore incompatible unless residency matches exactly.

Dominance is an optional flag; the differential harness (`keel_domcheck`)
requires status-and-optimal-cost agreement with the unpruned oracle across the
corpus before any use (mandate: never label a heuristic safe without
justification).

## 10. Baselines and the bounded planner (fixed for the experiment)

All policies either REJECT (task continues legally under Bo) or emit a full
witness `(τ, COMMIT, σ)` that the independent checker validates; validation is
part of the policy, and accepted-but-invalid counts as a policy defect.

1. `conservative` — zero-compute prep: repeatedly DROP+RELEASE droppable
   tensors (largest first, skipping Q and protected) while `released < Bo − Bn`;
   then require the continuation to be feasible via a bounded feasibility search
   (cap = planner budget, §12); any failure or UNKNOWN → REJECT.
2. `greedy` — unguarded reclamation: same drop+release ladder until COMMIT is
   legal; COMMIT; then a greedy continuation executor (topological RUNs,
   drop-after-last-use, evict-largest-recomputable on pressure). May FAIL
   (reported as `FAILED`, a policy defect by construction, never hidden).
3. `guarded_greedy` — up to 3 deterministic drop orders (largest-first,
   last-use-farthest-first, smallest-first); each candidate is fully validated
   by the checker before acceptance; first valid witness wins; else REJECT.
4. `keel` — bounded planner prototype: greedy incumbent seed, then beam search
   over prep choices (DROP/RELEASE/RUN scored by released-bytes-per-cost with
   feasibility-leaning tie-breaks), each candidate continuation solved by a
   bounded UCS with the same continuation cap as other policies. Statuses:
   ACCEPT(witness) / REJECT / UNKNOWN (never INFEASIBLE — it is not exact).
5. `oracle_assisted` — the oracle itself (uncapped): the upper bound of the
   achievable; also the source of ground-truth feasibility labels.

## 11. Metrics (per instance × policy)

status ∈ {ACCEPT, REJECT, FAILED, UNKNOWN}; `valid` (checker); `released_bytes`;
`prep_compute`; `cont_compute`; `recompute_compute` (Σ cost of continuation runs
of ops already done at COMMIT); `total_compute = prep + cont`;
`planning_nodes`; `planning_us`; plus oracle labels
(`oracle_status`, `oracle_total_compute`). Full distributions are reported
(min/median/p90/max + raw CSV); never only means.

## 12. Comparable planning budgets (fixed before held-out runs)

- Oracle: ≤ 300,000 expanded nodes (uncapped per-instance when the space is
  smaller; cap hits recorded as UNKNOWN and excluded from exact-label claims).
- Policy continuation feasibility checks: ≤ 20,000 nodes each.
- `keel`: ≤ 20,000 nodes per candidate continuation, ≤ 200 prep beam width,
  ≤ 2000 total candidate evaluations. Wall-clock is reported alongside node
  counts; node counts are the budget unit of record (machine-independent).

## 13. Adversarial families (declared)

`trap_recompute` (greedy drops mutually-recomputable pair → continuation
deadlock), `trap_nonreplayable` (greedy destroys a captured input's only
materialization), `trap_memory` (indiscriminate eviction makes any single
recompute exceed the ceiling; a planned partial-retention succeeds),
`prep_unlock` (one extra prep RUN unlocks a chain release that drop-only
policies cannot achieve). These families are declared adversarial: they are
reported in a separate section and are never pooled into headline rates.

---

## Changelog

### v1.1 (2026-11-01) — audit remediation
1. §4: in-flight ops reserve output bytes + scratch until completion
   (concurrent-launch oversubscription eliminated; finding 1).
2. §4: the invariant is validated after EVERY action, by both the model and
   the independent checker (finding 2).
3. §3/§5: `validate_state` / `validate_request` — structural state sanity
   (residency only from completed producers; no op both done and in flight;
   in-flight ops hold resident inputs and non-resident outputs; budgets:
   `Bo > 0`, `0 ≤ Bn < Bo`, `K ≥ 0`) and id-range checks on every legality
   predicate (finding 3).
4. Checked arithmetic in every accounting path, model and checker (finding 3).
No action vocabulary, goal condition, dominance rule, or baseline definition
changed. Sequential-mode semantics are bit-for-bit identical to v1.0.
