// KEEL Phase 3 + 5 — Live memory-grant handover: capture, adaptive planning
// (hybrid), concrete plan validation, the protocol FSM, and witness
// application against real arena storage.
//
// Protocol (mandate Part V):
//   REQUESTED -> CAPTURED -> PLANNING -> VALIDATED -> PREPARING -> READY
//   -> COMMITTED, with REJECTED / FAILED / EXPIRED / STALE outcomes.
//
// Correctness binding between the abstract model (SPEC v1.1) and physical
// storage:
//  - Capture converts the runtime frontier into the Phase 1 abstract state
//    (guarded: graphs above the 64-op/tensor caps refuse conversion rather
//    than planning inexact).
//  - The abstract planner (hybrid) emits a witness; the independent abstract
//    checker validates it; a CONCRETE validator additionally re-derives the
//    storage mapping (every released tensor's blocks must be free-owned and
//    transferable; the post-commit footprint must fit the new grant
//    physically, not just in bytes).
//  - Application replays the witness on the control thread; every action is
//    legality-checked again at runtime; any refusal aborts the handover with
//    the task left in a legal original-grant state (failure atomicity).
//  - Stale plans: the capture carries the task's state version; applying
//    against a moved state is refused.

#pragma once

#include <keel/rt/executor.hpp>

#include <functional>
#include <string>
#include <vector>

namespace keel::rt {

enum class HandoverPhase {
  Requested,
  Captured,
  Planning,
  Validated,
  Preparing,
  Ready,
  Committed,
  Rejected,   // proven infeasible / refused by the planner or checker
  Unknown,    // search budget exhausted: no claim either way
  Failed,     // refused mid-preparation/commit (task left legal under Bo)
  Stale,      // capture changed before preparation could begin
  Expired,    // caller cancelled before COMMIT; staging was rolled back
};

const char* handover_phase_name(HandoverPhase p);

struct HandoverLogEntry {
  std::string event;
  std::string detail;
};

struct HandoverResult {
  HandoverPhase phase = HandoverPhase::Requested;
  bool accepted = false;          // physical COMMIT completed (even if continuation later fails)
  // Planning is reported separately from physical acceptance so a rejected,
  // unknown, or checker-invalid plan is never confused with a transfer.
  std::string planning_status = "not_run";
  bool physical_preflight_ok = false;
  bool transfer_committed = false;
  int64_t requested_release_bytes = 0;  // Bo - Bn
  int64_t released_bytes = 0;           // physically transferred
  std::vector<int32_t> transferred_blocks;
  int64_t new_grant_bytes = 0;
  size_t outer_nodes = 0, cont_nodes = 0, candidates_checked = 0;
  size_t checker_invocations = 0;
  int64_t checker_us = 0, plan_us = 0, total_us = 0;
  int64_t capture_us = 0, preflight_us = 0, transfer_us = 0;
  int64_t commit_us = 0, wait_us = 0;
  int64_t prep_compute = 0, cont_compute = 0, recompute_compute = 0;
  // Runtime-id witness. Planner/checker witnesses are abstract-id values and
  // are translated once, after independent checking, before this is stored.
  Witness continuation;
  bool continuation_pending = false;
  std::string note;
  std::vector<HandoverLogEntry> log;
};

// Planning policies the handover can invoke (benchmark comparison axis).
enum class PlanPolicy {
  Conservative,
  GreedyUnguarded,
  GuardedGreedy,
  PrepGreedy,
  LimitedLookahead,
  Keel,
  Hybrid,
  Oracle,  // exact, small instances only (reference instrument)
};
const char* plan_policy_name(PlanPolicy p);
using HandoverCancel = std::function<bool()>;

// Performs one handover attempt synchronously on the control thread.
// `to_task` receives the transferred blocks. RELEASE stages extents during
// preparation; the physical transfer happens atomically at COMMIT. Returns
// the full log and outcome.
HandoverResult perform_handover(TaskRuntime& task, BlockArena& arena,
                                TaskId to_task, int64_t new_grant_bytes,
                                int64_t prep_budget, PlanPolicy policy,
                                bool stop_after_commit = false,
                                TaskRuntime* recipient = nullptr,
                                HandoverCancel cancel = {});

// Execute a continuation left pending by perform_handover(...,
// stop_after_commit=true). This is intentionally separate from COMMIT: the
// caller can start the recipient on the transferred physical blocks while the
// donor still has unfinished work.
bool continue_handover(TaskRuntime& task, HandoverResult& result,
                       std::string* why = nullptr);


}  // namespace keel::rt
