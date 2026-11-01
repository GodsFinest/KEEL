// KEEL Phase 1 — Exact small-instance oracle (docs/SPEC.md §7–§9).
//
// Uniform-cost search over the live-state space defined by the SPEC. Edge cost
// is the compute of RUN/LAUNCH; all control actions (DROP/RELEASE/WAIT/COMMIT)
// are free. The first goal state popped has minimal total compute (UCS
// completeness), and the reconstructed action sequence is a concrete witness.
//
// Statuses (SPEC §7, mandate §9): FEASIBLE carries a witness; INFEASIBLE is a
// proof (the reachable space within K was exhausted); UNKNOWN means the
// expansion budget ran out — it is never a feasibility claim.
//
// Dominance pruning (SPEC §9) is opt-in and sound by a drop-mimicking
// argument; it is validated by differential tests against the unpruned oracle
// before any experimental use (mandate: never label a heuristic safe without
// justification).

#pragma once

#include <keel/model.hpp>

#include <cstddef>
#include <cstdint>

namespace keel {

enum class OracleStatus {
  Feasible,
  Infeasible,
  Unknown,
};

const char* oracle_status_name(OracleStatus s);

struct OracleConfig {
  size_t max_expansions = 300000;  // node pops before giving up (-> Unknown)
  bool use_dominance = false;      // opt-in prune; differential-tested
  bool allow_async = false;        // generate LAUNCH/WAIT actions
};

struct OracleResult {
  OracleStatus status = OracleStatus::Unknown;
  Witness witness;           // meaningful iff Feasible
  int64_t total_compute = 0; // prep + continuation compute (iff Feasible)
  size_t expansions = 0;     // nodes popped (search work unit of record)
  size_t memo_size = 0;
  size_t dominance_skips = 0;
};

// Full transition problem: from live state s0 (either phase), find the
// min-compute (prep, COMMIT, continuation) witness or classify exactly.
OracleResult solve_transition(const Graph& g, const State& s0, const Request& r,
                              const OracleConfig& cfg);

// Convenience: continuation-only feasibility from an already-committed state
// (used by policies to validate candidate handovers under the same engine).
inline OracleResult solve_continuation(const Graph& g, const State& committed,
                                       const Request& r,
                                       const OracleConfig& cfg) {
  return solve_transition(g, committed, r, cfg);
}

}  // namespace keel
