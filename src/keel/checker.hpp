// KEEL Phase 1 — Independent plan checker.
//
// Independence contract (docs/DECISIONS.md D6, SPEC §1 note): this checker
// shares only IR *data* types with the planner stack. It re-derives every
// legality rule and its own memory accounting from docs/SPEC.md §4–§5 using a
// deliberately different internal representation (vectors of flags instead of
// bitmasks, direct loops instead of shared transition primitives). A bug in
// keel/model.cpp transitions therefore cannot mask itself here.
//
// The checker validates a concrete witness: every action legal in sequence,
// grant/accounting respected at every step, exactly one COMMIT, and the final
// state must be the goal (all required outputs resident, all in-flight work
// completed, handover committed).

#pragma once

#include <keel/model.hpp>

#include <string>

namespace keel {

struct CheckResult {
  bool ok = false;           // every action was legal
  bool reached_goal = false; // final state is the SPEC §5 goal
  int error_index = -1;      // index of the first illegal action (-1 if none)
  std::string error;         // human-readable failure reason
  // Metrics recomputed during validation (used by the experiment harness):
  int64_t prep_compute = 0;       // Σ cost of runs charged before COMMIT
  int64_t cont_compute = 0;       // Σ cost of runs after COMMIT
  int64_t recompute_compute = 0;  // cont runs of ops already done at COMMIT
  int64_t released_bytes = 0;
};

CheckResult check_witness(const Graph& g, const State& initial,
                          const Request& r, const Witness& w);

}  // namespace keel
