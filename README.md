# KEEL — Live Memory Revocation with Executable Continuations

A C++20 runtime where a **running computation hands real, physical memory to
another computation** while a checked plan proves the first computation can
still finish under its reduced grant.

## The problem

A tensor can be safe to discard now yet impossible to give away later under a
smaller memory allocation. Eviction heuristics (DTR-style) reclaim greedily
and hope; static planners (Checkmate-style) plan before execution and cannot
react to a live state. KEEL captures a running task's exact frontier, plans a
legal memory-grant transition from *that* state under a preparation budget,
validates the plan with an independent checker, then physically transfers
arena blocks to the co-tenant — and the original task keeps running, bit-exact
against an unconstrained reference, inside its smaller grant.

## Status (honest labels)

- **Implemented + tested**: typed runtime IR; hybrid/keel/greedy-family
  planners; independent plan checker (model + concrete levels); real block
  arena (ownership, generations, held extents, physical transfer, debug
  verifier); worker-pool numerical executor (matmul, elementwise, reduction,
  scratch workspace); live two-task handover protocol with failure atomicity;
  seeded graph generators; exact oracle for small instances.
- **Experimentally established**: Phase 1 feasibility study (SPEC v1.0/v1.1,
  20k+ abstract instances); Phase 7 real-runtime benchmark (1,200 trials).
- **Not implemented / unsupported**: GPU execution, multi-device, dynamic
  control flow, mutation/aliases, wall-clock asynchronous expiry (the
  synchronous API supports caller-driven pre-COMMIT cancellation),
  arbitrary side-effecting operators, production-readiness claims of any
  kind. Nothing here is claimed superior to published systems.

## Build, test, demo

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure     # full suite (doctest)
./build/keel_demo                              # live two-task handover
./build/keel_bench --seed-lo 1 --seed-hi 6 \
    --out experiments/results/rt_bench_v2.jsonl
python3 experiments/rt_aggregate.py experiments/results/rt_bench_v2.jsonl
```

Sanitizers:
```
cmake -S . -B build-san -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-san && ./build-san/keel_tests && ./build-san/keel_demo
```

## The demonstration (`keel_demo`)

Prints and asserts the 14 defining properties: initial grants; the reduction
request; planning result; checker validation; preparation operations;
in-flight completion events; freed allocation ids; transferred block ids;
Task B's resulting allocations; Task A's new grant; peak managed memory;
output correctness (bit-exact vs reference); and a final ownership/invariant
scan. A deliberately over-sized request is shown REJECTED with zero transfer.

## What the planner guarantees

A committed handover satisfies: every action legal from its predecessor;
memory within the grant at every step (checked per action); the preparation
budget respected; ≥ Bo−Bn bytes physically transferred; and an independently
checked, complete continuation for the original task under the new grant.
A rejected, expired, or pre-COMMIT failed handover moves zero bytes and is
reported truthfully. A post-COMMIT continuation failure remains a distinct
FAILED result after the physical transfer has already occurred.

## How KEEL differs from prior work

- Checkmate plans statically before execution; KEEL plans from a live
  frontier under a prep budget with a checked continuation.
- DTR-style rematerialization is a heuristic with no completion guarantee;
  KEEL's accepted handovers carry a checker-validated continuation witness.
- BMS schedules under a memory bound from the start; KEEL revokes mid-flight
  for a co-tenant with physical ownership transfer.
- MOLT revokes activations with implicit recompute-on-demand; KEEL generalizes
  to arbitrary DAG live states with explicit, independently validated plans.
- Coop is single-job allocation+rematerialization co-optimization (the founding
  brief's framing was corrected; see docs/RESEARCH.md).

## Documentation

- `docs/SPEC.md` — formal contract (v1.1, versioned changelog)
- `docs/ARCHITECTURE.md` — layer map, control/data-plane split
- `docs/INVARIANTS.md` — every verified obligation + the model↔reality boundary
- `docs/EXPERIMENTS.md` — frozen methodology (v1.0 + v1.1 addendum)
- `docs/PHASE1_AUDIT_REMEDIATION.md` — audit findings and fixes
- `docs/DECISIONS.md` — decision log (D1–D12)
- `docs/ROADMAP.md` — phase gates and evidence
- `experiments/results/` — locally generated raw data and benchmark outputs
