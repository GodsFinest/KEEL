# Decision Log

## D1 (2026-11-01) — Phase 1 memory model is a fixed shrinking region
The task owns a Bo-byte region; RELEASE physically transfers a held block out
(ceiling shrinks by the released bytes permanently); COMMIT(Bn) requires
released ≥ Bo − Bn. No "cede free slice" action: without layout (Phase 1 has
none) that would be unsound. Consequence: requests whose reduction exceeds what
the task can free from held data are honestly INFEASIBLE. Layout-aware ceding
is deferred to the Phase 4 arena. *Alternative rejected:* entitlement-only
accounting — it admits paper reductions the mandate explicitly forbids.

## D2 (2026-11-01) — Conservative peak accounting for ops
RUN/LAUNCH reserves outputs + scratch while keeping inputs resident: the check
is `usage + out_bytes + scratch ≤ ceiling`. Slightly over-strict (a real
allocator could free early-consumed inputs mid-op) but sound without intra-op
lifetime analysis. documented in SPEC §5.

## D3 (2026-11-01) — One handover per Phase 1 execution
RELEASE only in PREPARING; the continuation runs under the post-COMMIT ceiling
and may DROP but not RELEASE. Repeated/renewed handovers are Phase 2+ material.

## D4 (2026-11-01) — Async is completion-order-abstract
LAUNCH/WAIT with an explicit WAIT choice models any completion order without
wall-clock time. Real completion events are Phase 4 runtime semantics; no
timing claims are made here.

## D5 (2026-11-01) — K is a compute budget, not an action count
`cost(v)` work units; K bounds Σ cost of PREPARING RUN/LAUNCH only. DROP /
RELEASE / WAIT / COMMIT are control actions with zero modeled cost.

## D6 (2026-11-01) — Checker independence
The checker re-derives all legality from SPEC.md and shares only IR structs
with the planner stack. It is written before the oracle and never imports
planner transition code. (Mandate: shared faulty transition logic is not
independence.)

## D7 (2026-11-01) — Coop citation corrected
Verified single-job scope; see docs/RESEARCH.md. Affects positioning only.

## D8 (2026-11-01) — doctest vendored
Single header in external/doctest (9119 lines, upstream master). Keeps builds
hermetic (no network at configure time). License file to add alongside.

## D9 (2026-11-01) — Repo location
Workspace is iCloud Drive (`~/Library/Mobile Documents/com~apple~CloudDocs/KEEL`).
Git works there; fsync can lag — kept builds inside `build/` and commits at
milestones only.

## D10 (2026-11-01) — Release amplification (model discovery, pinned in tests)
Released bytes are NOT bounded by frontier residency. The oracle found (and the
checker validated): surrender x's block, re-derive x from still-resident inputs
into a fresh block, consume it, then surrender the fresh block as well. The
true bound is the region invariant `released ≤ Bo − goal footprint`, not the
resident sum. Consequences: (a) hand analyses of "max achievable release" that
sum resident bytes are unsound — the oracle is the only authority; (b) the
experiment's "bytes returned" metric can exceed frontier residency, which is
correct behavior, not a bug; (c) amplification costs compute, so K and the
compute metrics govern it. Pinned in tests/test_gen.cpp (trap_recompute).

## D11 (2026-11-01) — Dominance pruning: released equality is the only sound rule
The "obvious" dominance rule (more released = better) is UNSOUND, and so is the
opposite direction. Released bytes help COMMIT's threshold but shrink the
ceiling the mimicking dominator replays under; a dominator that surrendered
more can fail a memory check the dominated state passes. Both directions were
implemented and both failed the differential harness (`keel_domcheck`: 484 and
2021 mismatches on the corpus); exact equality on released (plus done-superset
with replayable extras, resident-superset, prep ≤, both quiescent) passes.
Lesson recorded per the mandate: pruning rules are validated, never assumed.
### D11 addendum (same day) — superset-residency pruning loses completeness
The released-equality fix alone still failed the differential (1980
mismatches). Root cause: with resident_A ⊋ resident_B, the mimic argument
needs A's drop-closure state (A minus extras) to be expanded — but that state
is itself dominated by A under pop-time pruning, so it is never expanded and
B's futures are covered by nobody. Sound Phase 1 rule: exact residency and
exact released equality (plus done-superset with replayable extras, prep ≤,
both quiescent). Richer sound schemes (e.g., dominance on canonical drop-
closures) are future work and must re-pass keel_domcheck.

## D12 (2026-11-01) — Audit remediation: async output reservations (SPEC v1.1)
Independent review finding 1 REPRODUCED and CONFIRMED: under v1.0, two
in-flight launches' output reservations were invisible to the shared usage
accounting, so the checker and the oracle both accepted a trace whose
transient footprint (two 6-unit outputs in flight) exceeded the region —
with legal-looking cleanup at the end. Fix (v1.1): LAUNCH reserves
outputs + scratch as usage until WAIT converts the reservation into
residency; the invariant is validated after every action by the checker
(independently re-derived) as well as the model. The audit counterexample
(Bo=10, Bn=9, two 6-unit ops) is a permanent regression test. v1.0 results
are preserved untouched and tagged; all headline numbers are re-derived
under v1.1 (see docs/PHASE1_AUDIT_REMEDIATION.md).

## D13 (2026-11-01) — Physical handover is a transactional two-stage protocol
The final runtime audit found that conflating DROP with RELEASE could make a
plan appear to return memory without naming the physical blocks, and that a
sequential extent loop could partially commit on metadata failure. Runtime
DROP therefore returns a run to the donor's free pool; RELEASE removes that
exact run from the pool into a held set; COMMIT validates a cloned physical
continuation and then transfers all held extents in one arena transaction.
The donor grant changes only after the transfer, and an optional recipient
runtime grant is increased by the exact transferred block bytes. Abort
unholds staged extents and reports a non-committed result.

## D14 (2026-11-01) — Concrete feasibility is a required gate after abstract checking
The abstract checker proves the SPEC witness, but it cannot see contiguous
physical runs or the allocator's current fragmentation. Every accepted runtime
witness is therefore additionally replayed against a cloned BlockArena with
the staged extents transferred, including every post-COMMIT RUN/LAUNCH/WAIT and
DROP. A byte-feasible but fragmented continuation is rejected before owner
bits move. This preserves the abstract research evidence while preventing a
physical false accept.

## D15 (2026-11-01) — Numerical reference independence is intentional
The production executor and the plain-vector reference use separate kernel
implementations. Fixed-order accumulation is made explicit so optimized
floating-point contraction cannot turn a printed-equal result into a bitwise
difference. Runtime acceptance tests include hand-calculated broadcast,
matmul, reduction, and elementwise results in addition to reference comparison.

## D16 (2026-11-01) — Concurrent data-plane overlap requires a synchronized arena
The original centralized-controller contract was sufficient for one task but
not for a live A/B overlap demonstration. Task controllers may now overlap
while workers execute; BlockArena protects ownership and allocation metadata
with a recursive mutex, while task metadata remains controller-owned and each
task still has one controller. TSan covers the resulting overlap path.

## D17 (2026-11-01) — Joint benchmark success is the primary outcome
An accepted COMMIT is not a successful trial. Runtime benchmark rows separately
record committed, A/B completion, independent A/B correctness, invariants,
physical overlap, B reuse of transferred blocks, and joint_success. The
aggregator reports distributions and joint rates without converting stalls,
planner UNKNOWN, or B failures into successes.
