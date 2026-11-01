// KEEL Phase 1 — IR, live state, and action model.
// Normative source: docs/SPEC.md (v1.0). This header defines data and the
// single-action legality/transition primitives shared by the oracle, planner,
// baselines, and generators. The independent checker deliberately does NOT
// use the transition primitives below (docs/DECISIONS.md D6): it re-derives
// legality from the SPEC so that a shared bug cannot hide.
//
// Scope notes (SPEC §1–§4): Phase 1 models one lender task's controlled
// memory as a fixed region of Bo bytes that only shrinks via RELEASE.
// All byte arithmetic is overflow-checked; op/tensor counts are capped at 64
// (small-instance regime; the exact oracle is only meaningful there anyway).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace keel {

using OpId = int;       // 0 .. n-1
using TensorId = int;   // 0 .. m-1

inline constexpr int kMaxOps = 64;
inline constexpr int kMaxTensors = 64;

using OpMask = uint64_t;      // bit i = op i
using TensorMask = uint64_t;  // bit i = tensor i

// ---------------------------------------------------------------------------
// IR

struct TensorDef {
  int64_t size = 0;       // bytes, > 0
  OpId producer = -1;     // always valid: external inputs come from source ops
  bool required = false;  // member of Q (must be resident in the goal state)
  bool pinned = false;    // never droppable
};

struct OpDef {
  std::vector<TensorId> inputs;
  std::vector<TensorId> outputs;
  int64_t scratch = 0;    // bytes reserved while in flight
  int64_t cost = 1;       // work units; unit of K and of all compute metrics
  bool replayable = true; // false = captured external input, runs at most once
};

struct Graph {
  std::vector<OpDef> ops;
  std::vector<TensorDef> tensors;
  std::vector<OpId> topo;        // topological order (producer before consumer)
  std::vector<OpMask> readers;   // readers[t] = ops with t as input (precomputed)

  TensorMask outputs_mask(OpId v) const {
    TensorMask m = 0;
    for (TensorId t : ops[v].outputs) m |= (TensorMask(1) << t);
    return m;
  }
  TensorMask inputs_mask(OpId v) const {
    TensorMask m = 0;
    for (TensorId t : ops[v].inputs) m |= (TensorMask(1) << t);
    return m;
  }
  OpMask readers_of(TensorId t) const { return readers[t]; }
  bool is_dag() const;  // topo non-empty implies DAG by construction
};

// Builder validates invariants (single producer, sizes > 0, DAG, caps).
class GraphBuilder {
 public:
  TensorId add_tensor(int64_t size, OpId producer, bool required = false,
                      bool pinned = false);
  OpId add_op(std::vector<TensorId> inputs, std::vector<TensorId> outputs,
              int64_t scratch, int64_t cost, bool replayable = true);
  Graph finish();  // computes topo order; throws std::invalid_argument on error

 private:
  Graph g_;
};

// ---------------------------------------------------------------------------
// Live state (SPEC §3)

struct State {
  OpMask done = 0;
  TensorMask resident = 0;
  OpMask inflight = 0;
  int64_t released_bytes = 0;  // permanently transferred to the co-tenant
  int64_t prep_used = 0;       // compute charged during PREPARING (frozen later)
  bool committed = false;      // phase: PREPARING -> CONTINUING at COMMIT

  bool operator==(const State& o) const {
    return done == o.done && resident == o.resident && inflight == o.inflight &&
           released_bytes == o.released_bytes && prep_used == o.prep_used &&
           committed == o.committed;
  }
};

// A handover request (SPEC §5, §7).
struct Request {
  int64_t Bo = 0;
  int64_t Bn = 0;  // Bn < Bo
  int64_t K = 0;   // preparation compute budget
};

// ---------------------------------------------------------------------------
// Accounting (SPEC §4). Checked arithmetic; overflow is an abort-level error.

struct Memory {
  int64_t usage = 0;      // resident bytes + inflight scratch
  int64_t ceiling = 0;    // grant-in-force minus released bytes
  bool ok = true;         // false on overflow (planner must abort)
};

// Accounting (SPEC v1.1 §4): usage = resident bytes + in-flight reservations,
// where each in-flight op reserves its scratch AND its output bytes until
// completion. LAUNCH adds the reservation; WAIT converts it into residency
// (usage drops by scratch only). Prevents concurrent-launch oversubscription.
Memory memory_of(const Graph& g, const State& s, const Request& r);
int64_t tensor_bytes(const Graph& g, TensorMask m);
int64_t inflight_reserved(const Graph& g, OpMask m);  // scratch + outputs

// Structural validation of a captured state (SPEC v1.1 §3): residency only
// from completed producers, no op both done and in flight, budgets sane.
// Returns an empty string when the state is well-formed.
std::string validate_state(const Graph& g, const State& s, const Request& r);

// Validates request invariants (SPEC v1.1 §5): Bo > 0, 0 <= Bn < Bo, K >= 0.
std::string validate_request(const Request& r);

// ---------------------------------------------------------------------------
// Actions (SPEC §5). Legality is total (never assumes preconditions);
// apply_* are defined only for legal actions (asserted in debug builds).

enum class Err {
  OK = 0,
  OP_INFLIGHT,
  INPUTS_NOT_RESIDENT,
  OUTPUT_ALREADY_RESIDENT,
  NOT_REPLAYABLE,
  OVER_CEILING,
  PREP_BUDGET_EXCEEDED,
  NOT_RESIDENT,
  PROTECTED_BY_INFLIGHT,
  PINNED,
  NOT_INFLIGHT,
  RELEASE_OUTSIDE_PREPARING,
  COMMIT_SHORTFALL,   // released < Bo - Bn
  ALREADY_COMMITTED,
};

const char* err_name(Err e);

struct Legality {
  bool ok;
  Err err;
};

Legality can_run(const Graph& g, const State& s, const Request& r, OpId v);
Legality can_launch(const Graph& g, const State& s, const Request& r, OpId v);
Legality can_wait(const Graph& g, const State& s, OpId v);
Legality can_drop(const Graph& g, const State& s, TensorId t);
Legality can_release(const Graph& g, const State& s, TensorId t);
Legality can_commit(const Graph& g, const State& s, const Request& r);

State apply_run(const Graph& g, const State& s, const Request& r, OpId v);
State apply_launch(const Graph& g, const State& s, const Request& r, OpId v);
State apply_wait(const Graph& g, const State& s, OpId v);
State apply_drop(const Graph& g, const State& s, TensorId t);
State apply_release(const Graph& g, const State& s, TensorId t);
State apply_commit(const Graph& g, const State& s, const Request& r);

bool is_goal(const Graph& g, const State& s);

// Frontier capture (SPEC §5, K semantics): prep_used counts compute charged
// AFTER the handover request is captured. Work executed before the capture is
// sunk cost under whatever budget was in force then; the request's K bounds
// only the transition's own preparation. Capture therefore resets prep_used.
State capture_frontier(const State& s);

// ---------------------------------------------------------------------------
// Witnesses (SPEC §7): the concrete, checkable action sequence.

enum class ActKind { Run, Launch, Wait, Drop, Release, Commit };

struct Act {
  ActKind kind;
  int target = -1;  // op id or tensor id; unused for Commit
};

struct Witness {
  std::vector<Act> acts;
};

const char* act_name(ActKind k);
std::string witness_to_text(const Graph& g, const Witness& w);

}  // namespace keel
