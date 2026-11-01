# Research Ledger — relationship to prior work (verified 2026-11-01)

Summaries below were verified against primary sources on 2026-11-01 (arXiv
abstracts/pages; BMS via ACM DL/Semantic Scholar metadata). Page-level only —
full-paper re-reads are noted as future work where they would matter.

## Checkmate (Jain et al., MLSys 2020; arXiv:1910.02653)
Static MILP over the whole training graph: per-tensor keep/evict/recompute
decisions, peak-memory constraints, minimizes training time; exact when the
MILP finishes (up to ~1h), approximation for large graphs; amortized over
iterations. **Relation:** plans *before* execution over a static graph; no live
state, no mid-run grant reduction, no handover, no independent runtime check.
KEEL's problem starts where Checkmate's output ends: an arbitrary live frontier.

## Dynamic Tensor Rematerialization (Sudarshan et al., ICLR 2021)
Online tensor-cache: evict by a cost/staleness heuristic, recompute on demand.
**Relation:** greedy, no feasibility guarantee that work completes under the
bound; no witness; no multi-task transfer. KEEL's `greedy` baseline is a
deterministic stand-in for this class of policy; KEEL's `guarded_greedy` shows
what adding a checked continuation to such a policy buys.

## Bounded Memory Scheduling (Sbîrlea, Budimlić, Sarkar — PACT 2014)
Schedules dynamic task graphs under a memory bound (CnC runtime); memory as a
first-class scheduling resource. **Relation:** same "schedule under a bound"
instinct, but single-program scheduling from the start; no live-state revocation
for a *co-tenant*, no continuation witness, no ownership transfer protocol.

## Coop (NeurIPS 2023; arXiv:2311.00591) — SPEC CORRECTION
The founding mandate lists Coop as memory borrowing/commodity work. Verified:
Coop is a **single-job** co-optimization of tensor allocation and
rematerialization (contiguous eviction windows, partitioning, recomputable
in-place) to reduce fragmentation and recompute cost. There is no inter-job
borrowing. Recorded so KEEL does not claim novelty against a strawman or cite
it for something it does not do.

## MOLT (arXiv:2610.05748, 2026)
Closest to KEEL's defining capability: an inference replica **revokes
activations saved by a live fine-tuning step** mid-step; the step completes by
recomputing them during backward; revocation only touches memory no in-flight
GPU work can access. **Relation and honest delta:** MOLT revokes *activations*,
whose recompute-on-demand is structurally always available (backward knows how
to re-derive them) — it needs no search, no continuation witness, no prep
budget, and no independently validated transition. KEEL's candidate niche is
the general case: arbitrary DAG live states, bounded prep compute, an explicit
witness, and independent validation. **Whether that machinery earns its cost
over MOLT-style implicit recompute (approximated here by guarded greedy) is
precisely the Phase 1 question**, and the answer may be negative.

## Positioning (to be re-validated by experiments, not asserted)
- Known and not claimed: checkpointing, rematerialization, bounded-memory
  scheduling, memory borrowing as concepts.
- Candidate delta (unproven until Phase 1 reports): efficient bounded planning
  of a legal grant transition from an arbitrary live state, with an
  independently checked complete continuation, under a prep budget.
- Known risk: for activation-style graphs, implicit recompute (MOLT) or guarded
  greedy may dominate explicit planning at every measured point. If Phase 1
  shows that across held-out families, the honest recommendation is **no-go**.
