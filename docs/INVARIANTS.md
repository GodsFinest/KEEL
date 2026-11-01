# KEEL Invariants (verified obligations)

Every invariant below is enforced by code and pinned by tests. The letter
maps to the founding mandate's Part III list.

| # | Invariant | Enforcement | Tests |
|---|---|---|---|
| O1 | **Ownership**: a block belongs to exactly one task or the free/held pool | arena ownership map + `verify()` re-derivation | test_rt_arena, test_rt_handover, every benchmark trial |
| O2 | **Lifetime**: no transfer or host access while an operation may access the block | DROP/RELEASE refuse in-flight reads and writes; invalid output pointers are hidden; staged extents are HELD until COMMIT | test_rt_executor, test_rt_handover, test_rt_adversarial |
| O3 | **Reservation**: outputs + scratch reserved before dispatch (physical) | `try_dispatch` allocates from owned blocks first; failure defers | test_rt_executor (stall tests) |
| O4 | **Dependency**: ops dispatch only with valid, resident inputs | `inputs_ready` + re-check at witness application | test_rt_executor |
| O5 | **Recomputation**: replayable producers only; identical bits on re-run | `recompute_tensor` refuses non-replayable; deterministic kernels | test_rt_executor (bit-exact recompute) |
| O6 | **Grant transition**: Bn effective only after preparation + real transfer | COMMIT transfers staged extents atomically, then lowers A's grant and raises an explicit recipient grant | test_rt_handover |
| O7 | **Failure atomicity**: bytes move only at COMMIT; abort/expiry leaves the task legal under its grant | staging discard on abort; truthful FAILED/EXPIRED reporting | test_rt_handover, test_rt_adversarial |
| O8 | **Completion**: required outputs resident at goal; checker enforces the goal on every accepted plan | abstract checker goal condition | all Phase 1 + runtime tests |
| O9 | **Numerical correctness**: constrained execution == independent reference bit-exact | separate plain-vector reference kernels, explicit accumulation order, hand-calculated regression | test_rt_executor, corrected benchmark rows |
| O10 | **Versioning**: planning against a moved state is refused | control-thread exclusivity + per-action re-checking + state_version guard | test_rt_adversarial |
| O11 | **Isolation**: B cannot touch A's blocks before transfer | separate task pools; ownership map; wrong-owner transfer refused | test_rt_arena, test_rt_adversarial |
| O12 | **No use-after-free**: dispatched ops hold reservations until completion; staged extents unallocatable | (O2, O3) + generational handles | test_rt_arena (stale handles) |
| O13 | **Per-action memory invariant**: usage ≤ ceiling after EVERY modeled action | model + independent checker (v1.1) | test_audit_repro, test_random_differential |
| O14 | **Accounting**: controlled bytes and owned blocks ≤ grant at all times, checked arithmetic | mutex-protected arena ownership; overflow-checked sums; no silent saturation | invariant scans in every test/bench trial |

## Known model↔reality boundary (documented, not hidden)

- **Fragmentation**: the abstract model accounts bytes; the arena requires
  contiguous runs. A post-release region can fragment so a later large
  scratch allocation stalls despite sufficient total bytes. Handover preflight
  rejects this before COMMIT; ordinary execution still reports a truthful
  stall. Allocation-aware release selection is future work.
- **Stall semantics**: a memory-pressure stall returns `run_to_completion() =
  false` — a legal, truthful outcome (no overcommit, no fabricated
  completion), never converted into a success status.
- **Expiry**: caller-driven cancellation is checked at pre-COMMIT boundaries and
  returns EXPIRED after discarding staging. Wall-clock asynchronous expiry is
  outside the supported synchronous protocol.
