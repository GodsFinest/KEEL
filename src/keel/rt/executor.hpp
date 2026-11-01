// KEEL Phase 4B — Numerical CPU execution engine.
//
// One TaskRuntime owns a typed runtime graph executing against the real
// BlockArena under a block grant. Design (mandate Part IV / Part III):
//
//  - Control-plane only: ALL metadata mutation (dispatch, completion
//    bookkeeping, drop, capture) happens on the scheduler/control thread.
//    Worker threads execute op bodies on memory handed to them and push
//    completion events; they never touch arena or task metadata. This is the
//    centralized-controller simplification the founding mandate permits.
//  - Reservation before dispatch: an op's output and scratch allocations are
//    made from the arena BEFORE dispatch, so an in-flight op's footprint is
//    physical from launch to completion (mirrors SPEC v1.1 §4).
//  - In-flight protection: DROP of a value an in-flight op reads is refused.
//  - Completion events: worker completions are drained by the control thread;
//    host-side dispatch is never treated as completion.
//  - Recomputation: a done op whose output was dropped may re-run (replayable
//    ops only) — values are immutable and deterministic, so recomputation
//    reproduces the same bits.
//  - Grant enforcement is physical: allocations come from the task's owned
//    blocks; oversubscription is an allocation failure, not a counter check.
//
// Numerical contract: every op is deterministic with a fixed internal order,
// so the parallel executor, a synchronous executor, and the plain-vector
// reference executor produce BIT-IDENTICAL results (exact comparison is
// therefore justified; no tolerance is needed or used).

#pragma once

#include <keel/rt/arena.hpp>
#include <keel/rt/ir.hpp>

#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace keel::rt {

struct TensorSlot {
  AllocHandle alloc;      // valid when resident
  bool valid = false;     // value materialized at least once
  bool live = false;      // becomes true when storage is allocated
  uint64_t version = 0;   // bumped on each (re)materialization
};

enum class OpState { Pending, InFlight, Done };

struct InFlightRecord {
  OpId op;
  std::vector<AllocHandle> outputs;
  AllocHandle scratch;  // invalid if none
};

enum class WorkerEvent { BeforeExecute, BeforeComplete, Completed };
using WorkerHook = std::function<void(OpId, WorkerEvent)>;

struct ExecutorStats {
  int64_t dispatches = 0;       // op executions started
  int64_t completions = 0;
  int64_t recomputations = 0;   // executions of already-done ops
  int64_t peak_live_bytes = 0;  // controlled-memory high-water mark
  int64_t alloc_failures = 0;   // dispatches deferred by memory pressure
};

class TaskRuntime {
 public:
  TaskRuntime(RuntimeGraph g, BlockArena& arena, TaskId task,
              int64_t grant_bytes, int workers);
  ~TaskRuntime();

  TaskRuntime(const TaskRuntime&) = delete;
  TaskRuntime& operator=(const TaskRuntime&) = delete;

  const RuntimeGraph& graph() const { return g_; }

  // Deterministic source data (seeded): sources must be loaded before use.
  void load_source(OpId source_op, uint64_t seed);

  // --- control-plane scheduling -------------------------------------------
  // Dispatch every op whose dependencies are satisfied and whose memory can
  // be reserved now. Returns the number dispatched.
  int dispatch_ready();
  // Block until `op` completes (draining completions meanwhile).
  void wait_op(OpId v);
  // Dispatch/drain until every op is done (or a stall is detected).
  // Returns false if the graph cannot complete or a required output is not
  // valid/resident after legal recovery is attempted.
  bool run_to_completion();
  // Terminal predicate used by handover continuation completion.  Unlike
  // run_to_completion(), this never dispatches or recomputes anything.
  bool completion_ready(std::string* why = nullptr) const;
  // Drain available completion events (non-blocking).
  void drain_completions();

  // --- memory actions (SPEC action language, concrete) ---------------------
  // DROP: frees the value's storage. Refused while an in-flight op reads it,
  // if pinned, or if already dropped. Returns false when refused.
  bool drop_tensor(TensorId t, std::string* why = nullptr);
  // RELEASE: drop + physically hand the freed blocks to the arena's
  // transfer path. Returns the moved block ids (empty when refused).
  std::vector<int32_t> release_tensor(TensorId t, TaskId to);
  // Re-materialize a dropped value by re-running its (replayable) producer:
  // synchronous dispatch + wait. Refused for non-replayable producers.
  bool recompute_tensor(TensorId t, std::string* why = nullptr);

  // --- capture (Phase 5) ---------------------------------------------------
  OpState op_state(OpId v) const;
  bool tensor_resident(TensorId t) const;
  bool tensor_valid(TensorId t) const;
  const void* tensor_ptr(TensorId t) const;
  AllocHandle tensor_allocation(TensorId t) const;
  int64_t controlled_bytes() const { return arena_.live_bytes(task_); }
  int64_t grant_bytes() const { return grant_bytes_; }
  const ExecutorStats& stats() const { return stats_; }
  uint64_t state_version() const { return state_version_; }
  const std::vector<OpId>& inflight() const { return inflight_list_; }
  // A coherent frontier: version, done/inflight masks, residency. The
  // scheduler must be quiescent (no dispatch in progress) when called.
  std::string frontier_error() const;  // empty when coherent
  // Validation hook used by tests and the handover: the controlled bytes
  // never exceed the grant and the arena invariants hold.
  bool invariants_hold(std::string* err) const;

  // --- handover staging (control thread only) ------------------------------
  TaskId task_id() const { return task_; }
  // During a handover, DROPped tensors have their extents STAGED: the physical
  // transfer happens only at COMMIT (failure atomicity — bytes move once).
  void begin_handover(TaskId to_task);
  // commit=true transfers staged extents to the co-tenant; false discards the
  // staging (freed blocks remain owned by this task — nothing ever moved).
  void end_handover(bool commit);
  bool staging_active() const { return staging_active_; }
  const std::vector<int32_t>& staged_blocks() const { return staged_blocks_; }
  size_t staged_block_count() const {
    size_t total = 0;
    for (const StagedExtent& e : staged_) total += static_cast<size_t>(e.nblocks);
    return total;
  }
  // Public dispatch entry for the handover's witness application (the plan
  // was validated; dispatch is still legality-checked by try_dispatch).
  bool try_dispatch_public(OpId v) { return try_dispatch(v); }
  // Grant adjustment at COMMIT: the new grant must cover the live footprint.
  void set_grant(int64_t new_bytes);
  bool grant_accepts(int64_t new_bytes) const;

  // A deterministic test/instrumentation hook. It is copied under the queue
  // mutex and invoked outside the mutex, so a hook may wait on a latch without
  // blocking metadata operations in another task.
  void set_worker_hook(WorkerHook hook);

  // Validate the post-COMMIT witness against a cloned arena layout. This is a
  // physical allocator preflight: it reserves every concrete output/scratch
  // extent and applies the witness's frees, waits, and drops without touching
  // the live task or arena.
  bool preflight_continuation(const std::vector<Act>& acts, size_t start,
                              int64_t new_grant_bytes, std::string* why) const;

 private:
  void worker_loop(int id);
  bool try_dispatch(OpId v, bool allow_uninitialized_source = false);
  // returns true if dispatched
  void execute_op(OpId v, const std::vector<void*>& ins, void* out,
                  void* scratch);
  void publish(OpId v);  // control-thread completion bookkeeping
  void discard_inflight(OpId v);  // reclaim reservations after worker failure
  bool inputs_ready(OpId v) const;
  int64_t inflight_reserved_bytes() const;
  bool required_outputs_ready(std::string* why) const;
  bool restore_tensor(TensorId t, std::vector<char>& visiting,
                      std::string* why);

  RuntimeGraph g_;
  BlockArena& arena_;
  TaskId task_;
  int64_t grant_bytes_;
  int workers_;

  std::vector<TensorSlot> slots_;
  std::vector<OpState> op_state_;
  std::vector<char> source_initialized_;
  std::vector<InFlightRecord> inflight_records_;
  std::vector<OpId> inflight_list_;

  // worker plumbing. Payload: [input pointers..., output pointer]
  // (+ scratch pointer when the op declares scratch).
  std::mutex mu_;
  std::condition_variable cv_;
  std::queue<std::pair<OpId, std::vector<double*>>> work_;
  struct Completion {
    OpId op;
    std::exception_ptr error;
  };
  std::queue<Completion> done_;
  std::vector<std::thread> threads_;
  bool stopping_ = false;
  std::exception_ptr worker_error_;
  WorkerHook worker_hook_;

  // In-flight input protection (DROP refuses while true).
  bool inflight_reads(TensorId t) const;
  bool inflight_writes(TensorId t) const;

  ExecutorStats stats_;
  uint64_t state_version_ = 0;

  // Handover staging state.
  bool staging_active_ = false;
  TaskId staging_to_ = -1;
  struct StagedExtent {
    TensorId tensor;
    int32_t first;
    int32_t nblocks;
  };
  std::vector<StagedExtent> staged_;
  std::vector<int32_t> staged_blocks_;
};

// Reference executor: plain vectors, single thread, no arena. Returns the
// value bytes of every tensor (row-major), computed in topo order.
std::vector<std::vector<double>> reference_execute_f64(
    const RuntimeGraph& g, const std::vector<uint64_t>& source_seeds);

}  // namespace keel::rt
