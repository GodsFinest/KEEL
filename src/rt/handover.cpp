#include <keel/rt/handover.hpp>

#include <keel/checker.hpp>
#include <keel/policy.hpp>

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace keel::rt {

const char* handover_phase_name(HandoverPhase p) {
  switch (p) {
    case HandoverPhase::Requested: return "REQUESTED";
    case HandoverPhase::Captured: return "CAPTURED";
    case HandoverPhase::Planning: return "PLANNING";
    case HandoverPhase::Validated: return "VALIDATED";
    case HandoverPhase::Preparing: return "PREPARING";
    case HandoverPhase::Ready: return "READY";
    case HandoverPhase::Committed: return "COMMITTED";
    case HandoverPhase::Rejected: return "REJECTED";
    case HandoverPhase::Unknown: return "UNKNOWN";
    case HandoverPhase::Failed: return "FAILED";
    case HandoverPhase::Stale: return "STALE";
    case HandoverPhase::Expired: return "EXPIRED";
  }
  return "?";
}

namespace {

struct TotalTimingGuard {
  HandoverResult& result;
  std::chrono::steady_clock::time_point start;
  ~TotalTimingGuard() {
    result.total_us = std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::steady_clock::now() - start)
                          .count();
  }
};

struct WaitTimingGuard {
  HandoverResult& result;
  const int64_t& wait_us;
  ~WaitTimingGuard() { result.wait_us = wait_us; }
};

struct Capture {
  RuntimeGraph::AbstractProjection projection;
  State state;
  Request request;
  uint64_t version;
};

// Planner and checker witnesses are expressed in the abstract graph's id
// space.  The executor has a different, stable runtime id space.  Translate
// exactly once after independent checking and keep all subsequent concrete
// application/preflight/continuation paths in runtime ids.
bool translate_witness_to_runtime(
    const Witness& abstract_witness,
    const RuntimeGraph::AbstractProjection& projection, Witness* runtime,
    std::string* why) {
  auto refuse = [&](const std::string& message) {
    if (why) *why = message;
    return false;
  };
  const size_t nops = projection.graph.ops.size();
  const size_t nts = projection.graph.tensors.size();
  if (projection.runtime_to_abstract_op.size() != nops ||
      projection.abstract_to_runtime_op.size() != nops ||
      projection.runtime_to_abstract_tensor.size() != nts ||
      projection.abstract_to_runtime_tensor.size() != nts)
    return refuse("abstract projection dimensions do not match its graph");
  for (size_t v = 0; v < nops; ++v) {
    const OpId a = projection.runtime_to_abstract_op[v];
    if (a < 0 || a >= (OpId)nops ||
        projection.abstract_to_runtime_op[(size_t)a] != (OpId)v)
      return refuse("abstract operation projection is not bijective");
  }
  for (size_t t = 0; t < nts; ++t) {
    const TensorId a = projection.runtime_to_abstract_tensor[t];
    if (a < 0 || a >= (TensorId)nts ||
        projection.abstract_to_runtime_tensor[(size_t)a] != (TensorId)t)
      return refuse("abstract tensor projection is not bijective");
  }
  runtime->acts.clear();
  runtime->acts.reserve(abstract_witness.acts.size());
  for (const Act& a : abstract_witness.acts) {
    Act translated = a;
    switch (a.kind) {
      case ActKind::Run:
      case ActKind::Launch:
      case ActKind::Wait:
        if (a.target < 0 || a.target >= (OpId)nops)
          return refuse("witness operation id is out of range");
        translated.target =
            projection.abstract_to_runtime_op[(size_t)a.target];
        break;
      case ActKind::Drop:
      case ActKind::Release:
        if (a.target < 0 || a.target >= (TensorId)nts)
          return refuse("witness tensor id is out of range");
        translated.target =
            projection.abstract_to_runtime_tensor[(size_t)a.target];
        break;
      case ActKind::Commit:
        break;
    }
    runtime->acts.push_back(translated);
  }
  return true;
}

// Convert the runtime frontier into the abstract model state. Control-thread
// exclusive: no dispatch can interleave. In-flight ops are represented as
// such; the witness's WAIT/LAUNCH actions drive them to completion.
Capture capture_state(TaskRuntime& task, int64_t new_grant_bytes,
                      int64_t prep_budget) {
  Capture c;
  c.projection = task.graph().to_abstract_projection();
  const size_t nops = task.graph().ops().size();
  const size_t nts = task.graph().tensors().size();
  for (size_t v = 0; v < nops; ++v) {
    const OpId abstract_v = c.projection.runtime_to_abstract_op[v];
    if (task.op_state((OpId)v) == OpState::Done)
      c.state.done |= (OpMask(1) << abstract_v);
    if (task.op_state((OpId)v) == OpState::InFlight)
      c.state.inflight |= (OpMask(1) << abstract_v);
  }
  // Abstract "resident" = the value is present (valid). In-flight outputs are
  // covered by the abstract in-flight reservation term instead.
  for (size_t t = 0; t < nts; ++t) {
    if (task.tensor_valid((TensorId)t)) {
      const TensorId abstract_t =
          c.projection.runtime_to_abstract_tensor[t];
      c.state.resident |= (TensorMask(1) << abstract_t);
    }
  }
  c.state.released_bytes = 0;
  c.state.prep_used = 0;
  c.state.committed = false;
  c.request = Request{task.grant_bytes(), new_grant_bytes, prep_budget};
  c.version = task.state_version();
  if (const std::string error =
          validate_state(c.projection.graph, c.state, c.request);
      !error.empty())
    throw std::invalid_argument("invalid captured frontier: " + error);
  return c;
}

// Apply one witness action concretely. Returns false + reason on refusal
// (caller aborts; the task remains legal under its grant — bytes move only
// at COMMIT). The state may have moved forward along the plan (in-flight ops
// completed): each action re-checks reality and skips what is satisfied.
bool apply_act(TaskRuntime& task, const Act& a, bool committed,
               [[maybe_unused]] int64_t new_grant_bytes,
               int64_t* prep_compute, int64_t* cont_compute,
               int64_t* recompute_compute, int64_t* wait_us,
               std::string* why,
               HandoverLogEntry* log_entry) {
  auto wait_and_measure = [&](OpId v) {
    const auto start = std::chrono::steady_clock::now();
    task.wait_op(v);
    *wait_us += std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - start)
                    .count();
  };
  switch (a.kind) {
    case ActKind::Run:
    case ActKind::Launch: {
      const OpId v = a.target;
      const OpSpec& op = task.graph().ops()[v];
      if (task.op_state(v) == OpState::InFlight) {
        // Already launched by this witness: WAIT semantics apply.
        wait_and_measure(v);
        if (log_entry)
          *log_entry = {"wait", "v" + std::to_string(v) + " completed"};
        return true;
      }
      bool all_resident = true;
      for (TensorId t : op.outputs)
        if (!task.tensor_resident((TensorId)t) ||
            !task.tensor_valid((TensorId)t))
          all_resident = false;
      if (task.op_state(v) == OpState::Done && all_resident) {
        if (log_entry)
          *log_entry = {"skip", std::string(op_kind_name(op.kind)) + " v" +
                                    std::to_string(v) + " already satisfied"};
        return true;
      }
      // Legality re-check: inputs resident and valid.
      for (TensorId t : op.inputs) {
        if (!task.tensor_valid((TensorId)t) || !task.tensor_resident((TensorId)t)) {
          *why = std::string(act_name(a.kind)) + " v" + std::to_string(v) +
                 ": input t" + std::to_string(t) + " missing";
          return false;
        }
      }
      if (committed) {
        *cont_compute += op.cost;
        if (task.op_state(v) == OpState::Done) *recompute_compute += op.cost;
      } else {
        *prep_compute += op.cost;
      }
      if (a.kind == ActKind::Launch) {
        if (!task.try_dispatch_public(v)) {
          *why = std::string("LAUNCH v") + std::to_string(v) +
                 ": dispatch failed (memory pressure)";
          return false;
        }
        if (log_entry)
          *log_entry = {"launch", "v" + std::to_string(v)};
        return true;
      }
      // RUN: synchronous. A done op whose outputs are gone recomputes.
      if (task.op_state(v) == OpState::Done) {
        const auto start = std::chrono::steady_clock::now();
        if (!task.recompute_tensor(op.outputs[0], why)) return false;
        *wait_us += std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - start)
                        .count();
      } else {
        if (!task.try_dispatch_public(v)) {
          *why = std::string("RUN v") + std::to_string(v) +
                 ": dispatch failed (memory pressure)";
          return false;
        }
        wait_and_measure(v);
      }
      if (log_entry)
        *log_entry = {"exec", std::string(act_name(a.kind)) + " v" +
                                  std::to_string(v)};
      return true;
    }
    case ActKind::Wait: {
      wait_and_measure(a.target);
      if (log_entry)
        *log_entry = {"wait", "v" + std::to_string(a.target) + " completed"};
      return true;
    }
    case ActKind::Drop: {
      const TensorId t = a.target;
      if (!task.tensor_resident(t)) {
        if (log_entry)
          *log_entry = {"skip", "drop t" + std::to_string(t) + " (gone)"};
        return true;  // moved forward along the plan
      }
      std::string why2;
      if (!task.drop_tensor(t, &why2)) {
        *why = std::string("DROP t") + std::to_string(t) + ": " + why2;
        return false;
      }
      if (log_entry)
        *log_entry = {"drop",
                      "t" + std::to_string(t) + " (" +
                          std::to_string(task.graph().tensor_bytes(t)) + " B)"};
      return true;
    }
    case ActKind::Release: {
      // RELEASE is deliberately distinct from DROP. It drops the value and
      // reserves its exact physical extent in the handover's held set; only
      // COMMIT changes the owner bits.
      const TensorId t = a.target;
      if (!task.tensor_resident(t)) {
        *why = "RELEASE t" + std::to_string(t) + ": tensor is not resident";
        return false;
      }
      *why = "RELEASE must be bound to the handover recipient";
      return false;
    }
    case ActKind::Commit: {
      // COMMIT is handled by perform_handover after the concrete continuation
      // preflight. It is not legal for a generic action applicator to mutate
      // ownership without that gate.
      if (committed) {
        *why = "duplicate COMMIT";
        return false;
      }
      if (log_entry)
        *log_entry = {"ready", "COMMIT awaiting physical preflight"};
      return true;
    }
  }
  *why = "unknown action";
  return false;
}

}  // namespace

const char* plan_policy_name(PlanPolicy p) {
  switch (p) {
    case PlanPolicy::Conservative: return "conservative";
    case PlanPolicy::GreedyUnguarded: return "greedy";
    case PlanPolicy::GuardedGreedy: return "guarded_greedy";
    case PlanPolicy::PrepGreedy: return "prep_greedy";
    case PlanPolicy::LimitedLookahead: return "limited_lookahead";
    case PlanPolicy::Keel: return "keel";
    case PlanPolicy::Hybrid: return "hybrid";
    case PlanPolicy::Oracle: return "oracle";
  }
  return "?";
}


PolicyOutcome run_planner(PlanPolicy policy, const Graph& abs,
                          const State& s0, const Request& r,
                          const PolicyConfig& cfg, OracleConfig& oc) {
  switch (policy) {
    case PlanPolicy::Conservative: return conservative(abs, s0, r, cfg);
    case PlanPolicy::GreedyUnguarded: return greedy(abs, s0, r, cfg);
    case PlanPolicy::GuardedGreedy: return guarded_greedy(abs, s0, r, cfg);
    case PlanPolicy::PrepGreedy: return prep_greedy(abs, s0, r, cfg);
    case PlanPolicy::LimitedLookahead: return limited_lookahead(abs, s0, r, cfg);
    case PlanPolicy::Keel: return keel_plan(abs, s0, r, cfg);
    case PlanPolicy::Hybrid: return hybrid(abs, s0, r, cfg);
    case PlanPolicy::Oracle: {
      OracleResult o = solve_transition(abs, s0, r, oc);
      PolicyOutcome p;
      p.status = policy_status_from_oracle(o.status);
      p.witness = o.witness;
      p.outer_expansions = o.expansions;
      p.planning_nodes = o.expansions;
      p.note = oracle_status_name(o.status);
      return p;
    }
  }
  return guarded_greedy(abs, s0, r, cfg);
}

HandoverResult perform_handover(TaskRuntime& task, BlockArena& arena,
                                TaskId to_task, int64_t new_grant_bytes,
                                int64_t prep_budget, PlanPolicy policy,
                                bool stop_after_commit,
                                TaskRuntime* recipient,
                                HandoverCancel cancel) {
  HandoverResult out;
  if (__builtin_sub_overflow(task.grant_bytes(), new_grant_bytes,
                             &out.requested_release_bytes)) {
    out.phase = HandoverPhase::Rejected;
    out.note = "request arithmetic overflows";
    return out;
  }
  const auto t0 = std::chrono::steady_clock::now();
  TotalTimingGuard total_guard{out, t0};
  auto log = [&](const std::string& event, const std::string& detail) {
    out.log.push_back({event, detail});
  };
  auto expired = [&](const char* point) {
    if (!cancel || !cancel()) return false;
    if (task.staging_active()) task.end_handover(false);
    out.phase = HandoverPhase::Expired;
    out.note = std::string("handover cancelled before COMMIT at ") + point;
    log("expired", out.note);
    out.total_us = std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::steady_clock::now() - t0)
                       .count();
    return true;
  };

  // 1. CAPTURED (control-thread exclusive: coherent frontier).
  Capture cap;
  const auto capture_start = std::chrono::steady_clock::now();
  try {
    cap = capture_state(task, new_grant_bytes, prep_budget);
    out.capture_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::steady_clock::now() - capture_start)
                         .count();
  } catch (const std::exception& e) {
    out.capture_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::steady_clock::now() - capture_start)
                         .count();
    out.phase = HandoverPhase::Rejected;
    out.note = std::string("capture failed: ") + e.what();
    log("capture_failed", out.note);
    return out;
  }
  out.phase = HandoverPhase::Captured;
  log("captured", "done=" + std::to_string(cap.state.done) +
                      " inflight=" + std::to_string(cap.state.inflight) +
                      " resident=" + std::to_string(cap.state.resident));
  if (expired("capture")) return out;

  // 2. PLANNING: adaptive policy = hybrid (guarded fast path, then bounded
  //    continuation-aware search). Pure computation on the snapshot.
  out.phase = HandoverPhase::Planning;
  PolicyConfig cfg;
  OracleConfig ocfg;
  ocfg.allow_async = cap.state.inflight != 0;
  const auto t_plan = std::chrono::steady_clock::now();
  PolicyOutcome plan = run_planner(policy, cap.projection.graph, cap.state,
                                   cap.request,
                                   cfg, ocfg);
  out.plan_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t_plan)
                    .count();
  out.outer_nodes = plan.outer_expansions;
  out.cont_nodes = plan.cont_expansions_total;
  out.candidates_checked = plan.candidates_checked;
  out.checker_invocations = plan.checker_invocations;
  out.checker_us = plan.checker_us;
  out.prep_compute = plan.prep_compute;
  out.cont_compute = plan.cont_compute;
  out.recompute_compute = plan.recompute_compute;

  if (plan.status != PolicyStatus::Accept) {
    out.planning_status = plan.status == PolicyStatus::Unknown ? "unknown"
                                                               : "rejected";
    out.phase = plan.status == PolicyStatus::Unknown ? HandoverPhase::Unknown
                                                     : HandoverPhase::Rejected;
    out.note = "planner: " + plan.note;
    log(out.phase == HandoverPhase::Unknown ? "planner_unknown" : "rejected",
        out.note);
    return out;
  }

  // 3. VALIDATED: the independent abstract checker must accept the witness.
  CheckResult check = check_witness(cap.projection.graph, cap.state,
                                    cap.request, plan.witness);
  if (!check.ok || !check.reached_goal) {
    out.planning_status = "checker_rejected";
    out.phase = HandoverPhase::Rejected;
    out.note = "checker rejected planner witness: " +
               (check.error.empty() ? "goal not reached" : check.error);
    log("checker_rejected", out.note);
    return out;
  }
  out.planning_status = "accepted";
  out.phase = HandoverPhase::Validated;
  log("validated", "witness: " + std::to_string(plan.witness.acts.size()) +
                       " actions");
  if (expired("validation")) return out;

  Witness runtime_witness;
  std::string translation_error;
  if (!translate_witness_to_runtime(plan.witness, cap.projection,
                                    &runtime_witness, &translation_error)) {
    out.planning_status = "translation_rejected";
    out.phase = HandoverPhase::Rejected;
    out.note = "witness translation rejected: " + translation_error;
    log("translation_rejected", out.note);
    return out;
  }

  // 4. PREPARING + 5. COMMIT + 6. CONTINUATION: apply the witness concretely.
  //    Refusal at any action aborts; bytes move only at COMMIT (failure
  //    atomicity), so an abort leaves the task legal under its grant.
  // Planning is pure snapshot work. Do not apply a plan captured before a
  // control-plane state transition was observed.
  if (task.state_version() != cap.version) {
    out.phase = HandoverPhase::Stale;
    out.note = "captured frontier changed before preparation";
    log("stale", out.note);
    return out;
  }
  if (recipient && recipient->task_id() != to_task) {
    out.phase = HandoverPhase::Failed;
    out.note = "recipient task id does not match handover target";
    log("recipient_mismatch", out.note);
    return out;
  }

  out.phase = HandoverPhase::Preparing;
  task.begin_handover(to_task);
  int64_t prep_compute = 0, cont_compute = 0, recompute_compute = 0;
  int64_t wait_us = 0;
  WaitTimingGuard wait_guard{out, wait_us};
  bool committed = false;
  try {
    for (size_t i = 0; i < runtime_witness.acts.size(); ++i) {
    const Act& a = runtime_witness.acts[i];
    if (expired("preparation")) return out;
    if (a.kind == ActKind::Commit) out.phase = HandoverPhase::Ready;
    if (a.kind == ActKind::Release) {
      // Bind RELEASE to the concrete recipient here rather than allowing the
      // generic DROP path to silently stage it.
      const TensorId t = a.target;
      if (!task.tensor_resident(t)) {
        task.end_handover(false);
        out.phase = HandoverPhase::Failed;
        out.note = "RELEASE target is not resident";
        log("aborted", out.note);
        return out;
      }
      if (task.release_tensor(t, to_task).empty()) {
        task.end_handover(false);
        out.phase = HandoverPhase::Failed;
        out.note = "RELEASE could not stage its physical extent";
        log("aborted", out.note);
        return out;
      }
      std::string invariant_error;
      if (!task.invariants_hold(&invariant_error)) {
        task.end_handover(false);
        out.phase = HandoverPhase::Failed;
        out.note = "RELEASE violated a concrete invariant: " + invariant_error;
        log("aborted", out.note);
        return out;
      }
      out.log.push_back({"stage", "t" + std::to_string(t) + " staged"});
      if (expired("release staging")) return out;
      continue;
    }
    if (a.kind == ActKind::Commit) {
      if (expired("commit gate")) return out;
      std::string why;
      const auto preflight_start = std::chrono::steady_clock::now();
      const bool preflight_ok = task.preflight_continuation(
          runtime_witness.acts, i + 1, new_grant_bytes, &why);
      out.preflight_us += std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now() - preflight_start)
                              .count();
      if (!preflight_ok) {
        task.end_handover(false);
        out.phase = HandoverPhase::Failed;
        out.note = "physical continuation preflight rejected COMMIT: " + why;
        log("preflight_rejected", out.note);
        out.total_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::steady_clock::now() - t0)
                           .count();
        return out;
      }
      out.physical_preflight_ok = true;
      if (expired("post-preflight gate")) return out;
      const int64_t released_bytes =
          static_cast<int64_t>(task.staged_block_count()) * arena.block_size();
      int64_t recipient_grant = 0;
      if (recipient) {
        if (__builtin_add_overflow(recipient->grant_bytes(), released_bytes,
                                   &recipient_grant)) {
          task.end_handover(false);
          out.phase = HandoverPhase::Failed;
          out.note = "recipient grant arithmetic overflows";
          log("recipient_rejected", out.note);
          return out;
        }
        if (arena.bytes_owned(recipient->task_id()) > recipient->grant_bytes() ||
            !recipient->grant_accepts(recipient_grant)) {
          task.end_handover(false);
          out.phase = HandoverPhase::Failed;
          out.note = "recipient cannot accept the transferred ownership";
          log("recipient_rejected", out.note);
          return out;
        }
      }
      HandoverLogEntry entry;
      const bool ok = apply_act(task, a, committed,
                                new_grant_bytes, &prep_compute, &cont_compute,
                                &recompute_compute, &wait_us, nullptr, &entry);
      if (!ok) {
        task.end_handover(false);
        out.phase = HandoverPhase::Failed;
        out.note = "COMMIT action refused before transfer";
        log("aborted", out.note);
        return out;
      }
      const auto transfer_start = std::chrono::steady_clock::now();
      try {
        // transfer_held_extents is transactional: an exception occurs before
        // owner bits change, so the staging can still be safely discarded.
        task.end_handover(/*commit=*/true);
      } catch (const std::exception& e) {
        try {
          task.end_handover(/*commit=*/false);
        } catch (...) {
          // The arena transaction is required to be non-mutating on failure;
          // preserve the original exception as the diagnostic if rollback
          // bookkeeping itself cannot be allocated.
        }
        out.phase = HandoverPhase::Failed;
        out.note = std::string("commit transaction failed before transfer: ") +
                   e.what();
        log("aborted", out.note);
        return out;
      }
      task.set_grant(new_grant_bytes);
      if (recipient)
        recipient->set_grant(recipient_grant);
      out.transfer_us += std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - transfer_start)
                             .count();
      out.commit_us += std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::steady_clock::now() - transfer_start)
                           .count();
      committed = true;
      out.phase = HandoverPhase::Committed;
      out.accepted = true;
      out.transfer_committed = true;
      out.new_grant_bytes = new_grant_bytes;
      out.released_bytes = released_bytes;
      out.transferred_blocks = task.staged_blocks();
      log("commit", "transferred " +
                        std::to_string(out.transferred_blocks.size()) +
                        " blocks (" + std::to_string(out.released_bytes) +
                        " B) to task " + std::to_string(to_task));
      std::string commit_error;
      if (!task.invariants_hold(&commit_error) ||
          (recipient && !recipient->invariants_hold(&commit_error))) {
        out.phase = HandoverPhase::Failed;
        out.note = "post-COMMIT concrete invariant failure: " + commit_error;
        log("commit_invariant_failure", out.note);
        out.accepted = true;
        return out;
      }
      if (stop_after_commit) {
        out.continuation.acts.assign(runtime_witness.acts.begin() +
                                         static_cast<std::ptrdiff_t>(i + 1),
                                     runtime_witness.acts.end());
        out.continuation_pending = !out.continuation.acts.empty();
        break;
      }
      continue;
    }
    std::string why;
    HandoverLogEntry entry;
    const bool ok = apply_act(task, a, committed,
                              new_grant_bytes, &prep_compute, &cont_compute,
                              &recompute_compute, &wait_us, &why, &entry);
    if (!entry.event.empty()) out.log.push_back(entry);
    if (!ok) {
      task.end_handover(false);  // discard staging: nothing ever moved
      out.phase = HandoverPhase::Failed;
      out.note = "aborted at " + std::string(act_name(a.kind)) + ": " + why;
      log("aborted", out.note);
      out.total_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::steady_clock::now() - t0)
                         .count();
      return out;
    }
    std::string invariant_error;
    if (!task.invariants_hold(&invariant_error)) {
      task.end_handover(false);
      out.phase = HandoverPhase::Failed;
      out.note = "preparation violated a concrete invariant: " + invariant_error;
      log("aborted", out.note);
      return out;
    }
    }
  } catch (const std::exception& e) {
    if (task.staging_active()) {
      try {
        task.end_handover(false);
      } catch (...) {
      }
    }
    out.phase = HandoverPhase::Failed;
    out.note = std::string("handover action raised: ") + e.what();
    log("aborted", out.note);
    return out;
  } catch (...) {
    if (task.staging_active()) {
      try {
        task.end_handover(false);
      } catch (...) {
      }
    }
    out.phase = HandoverPhase::Failed;
    out.note = "handover action raised an unknown exception";
    log("aborted", out.note);
    return out;
  }
  out.prep_compute = prep_compute;
  out.cont_compute = cont_compute;
  out.recompute_compute = recompute_compute;
  out.wait_us = wait_us;
  if (!committed) task.end_handover(false);
  if (committed && (!stop_after_commit || !out.continuation_pending)) {
    std::string terminal_error;
    if (!task.completion_ready(&terminal_error)) {
      out.phase = HandoverPhase::Failed;
      out.note = "concrete witness did not reach a valid terminal state: " +
                 terminal_error;
      log("terminal_state_failed", out.note);
    }
  }
  out.total_us = std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - t0)
                     .count();
  if (committed)
    log("done", "task continues under " + std::to_string(new_grant_bytes) +
                    " B; co-tenant received " +
                    std::to_string(out.released_bytes) + " B");
  return out;
}

bool continue_handover(TaskRuntime& task, HandoverResult& result,
                       std::string* why) {
  if (!result.accepted || !result.continuation_pending) {
    if (why) *why = "handover has no pending continuation";
    return !result.continuation_pending;
  }
  int64_t prep_compute = result.prep_compute;
  int64_t cont_compute = result.cont_compute;
  int64_t recompute_compute = result.recompute_compute;
  int64_t wait_us = result.wait_us;
  WaitTimingGuard wait_guard{result, wait_us};
  try {
    for (const Act& a : result.continuation.acts) {
    std::string local;
    HandoverLogEntry entry;
    if (!apply_act(task, a, /*committed=*/true, result.new_grant_bytes,
                   &prep_compute,
                   &cont_compute, &recompute_compute, &wait_us, &local,
                   &entry)) {
      if (why) *why = local;
      result.phase = HandoverPhase::Failed;
      result.note = "continuation failed: " + local;
      return false;
    }
    std::string invariant_error;
    if (!task.invariants_hold(&invariant_error)) {
      if (why) *why = invariant_error;
      result.phase = HandoverPhase::Failed;
      result.note = "continuation violated a concrete invariant: " +
                    invariant_error;
      return false;
    }
    if (!entry.event.empty()) result.log.push_back(entry);
    }
  } catch (const std::exception& e) {
    if (why) *why = std::string("continuation raised: ") + e.what();
    result.phase = HandoverPhase::Failed;
    result.note = why ? *why : "continuation raised an exception";
    return false;
  } catch (...) {
    if (why) *why = "continuation raised an unknown exception";
    result.phase = HandoverPhase::Failed;
    result.note = why ? *why : "continuation raised an exception";
    return false;
  }
  std::string terminal_error;
  if (!task.completion_ready(&terminal_error)) {
    if (why) *why = terminal_error;
    result.phase = HandoverPhase::Failed;
    result.note = "continuation did not reach a valid terminal state: " +
                  terminal_error;
    return false;
  }
  result.continuation_pending = false;
  result.prep_compute = prep_compute;
  result.cont_compute = cont_compute;
  result.recompute_compute = recompute_compute;
  result.wait_us = wait_us;
  result.log.push_back({"continuation_done", "post-COMMIT witness completed"});
  return true;
}

}  // namespace keel::rt
