# KEEL Architecture (full CPU runtime)

## Layer map

```
┌──────────────────────────────────────────────────────────────┐
│ keel_demo / keel_bench          (demonstration, measurement) │
├──────────────────────────────────────────────────────────────┤
│ rt::perform_handover            (Phase 5 protocol FSM:       │
│   capture → plan → validate →     REQUESTED…COMMITTED,       │
│   prepare → commit → continue     failure atomicity)         │
├──────────────────────────┬───────────────────────────────────┤
│ rt::TaskRuntime          │ abstract model (SPEC v1.1)        │
│  - worker-pool executor  │  Graph / State / Request / Act    │
│  - dispatch + completions│  hybrid planner (Phase 1,         │
│  - drop / recompute      │    audit-hardened)                │
│  - staging for handover  │  independent checker              │
│  - reference executor    │  exact oracle (≤64 ops)           │
├──────────────────────────┴───────────────────────────────────┤
│ rt::BlockArena                  (Phase 4A: real backing      │
│   blocks, ownership, extents,   storage, block grants,       │
│   generations, held extents,    physical transfer)           │
├──────────────────────────────────────────────────────────────┤
│ rt::RuntimeGraph                (Phase 2: typed IR,          │
│   dtypes, shapes, checked       guarded conversion to the    │
│   sizes, op semantics           abstract model)              │
└──────────────────────────────────────────────────────────────┘
```

## Control-plane / data-plane split

Each task's metadata mutation — dispatch decisions, completion bookkeeping,
drop, capture, and staging — happens on that task's **single control thread**.
Worker
threads only run op kernels on memory handed to them and push completion
events. A live A/B demonstration intentionally overlaps two task controllers,
so BlockArena ownership/allocation metadata is protected by an internal
mutex; task-local metadata remains controller-owned. Workers never change
arena metadata directly and synchronize only on the job/completion queues.

## The abstract↔concrete bridge

The planner and checker operate on the SPEC v1.1 abstract model. The runtime
binds the two levels with:

1. **Guarded conversion** (`RuntimeGraph::to_abstract`): refuses graphs above
   the 64-op/64-tensor caps instead of planning inexactly.
2. **Capture**: `tensor_valid` ↔ abstract residency; in-flight ops ↔ abstract
   in-flight set (their reserved outputs+scratch are covered by the abstract
   reservation term).
3. **Witness application**: every action is re-legality-checked concretely;
   forward drift (an in-flight op completing during planning) is tolerated by
   skip-if-satisfied logic; any refusal aborts with the task legal.
4. **Physical preflight and commit**: RELEASEd extents are HELD
   (unallocatable) during preparation; a cloned arena replays the continuation
   with real contiguous allocations; only then does COMMIT transfer all held
   extents atomically and lower the grant.

## Physical memory model

- Arena = one over-aligned backing buffer ÷ 256 B blocks.
- Task grant = claimed block count; allocations = contiguous block runs
  (first-fit); alignment is the block size.
- `alloc`/`free`/`hold`/`unhold`/`transfer_extent` are the only ownership
  primitives; a debug verifier re-derives all tables and checks overlap,
  ownership, and accounting after every interesting step in tests.

## Planning

The adaptive runtime policy is the **hybrid** (Phase 1): cheap guarded
reclamation first, bounded continuation-aware search on failure. The exact
oracle remains available for small instances (all benchmark workloads are
within its caps). Statuses: ACCEPT / REJECT / UNKNOWN — never a false
infeasibility claim. Work accounting separates search nodes (UCS pops) from
executor steps (constructive policies) — they are not comparable and are
never summed.
