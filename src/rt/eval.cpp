// Reproducible CPU-runtime evaluation driver.
//
// This is deliberately separate from the historical keel_bench executable:
// old rows remain immutable evidence, while this driver makes paired
// frontiers, no-handover baselines, outcome categories, timing boundaries,
// allocator shape, and physical reuse explicit for the research study.

#include <keel/rt/bench_gen.hpp>
#include <keel/rt/handover.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace keel;
using namespace keel::rt;

namespace {

using Clock = std::chrono::steady_clock;

int64_t elapsed_us(Clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start)
      .count();
}

std::string json_string(const std::string& value) {
  std::string out = "\"";
  for (char c : value) {
    if (c == '\\' || c == '"') out += '\\';
    out += c;
  }
  out += '"';
  return out;
}

const char* bool_text(bool value) { return value ? "true" : "false"; }

std::vector<std::string> split(const char* text) {
  std::vector<std::string> out;
  std::string current;
  for (const char* p = text; ; ++p) {
    if (*p == ',' || *p == '\0') {
      if (!current.empty()) out.push_back(current);
      current.clear();
      if (*p == '\0') break;
    } else {
      current += *p;
    }
  }
  return out;
}

std::vector<double> parse_doubles(const char* text) {
  std::vector<double> out;
  for (const std::string& value : split(text))
    out.push_back(std::atof(value.c_str()));
  return out;
}

BenchFamily parse_family(const std::string& name) {
  for (int i = 0; i < bench_family_count(); ++i) {
    const BenchFamily family = static_cast<BenchFamily>(i);
    if (name == bench_family_name(family)) return family;
  }
  throw std::invalid_argument("unknown benchmark family: " + name);
}

PlanPolicy parse_policy(const std::string& name) {
  const std::pair<const char*, PlanPolicy> names[] = {
      {"conservative", PlanPolicy::Conservative},
      {"greedy", PlanPolicy::GreedyUnguarded},
      {"guarded_greedy", PlanPolicy::GuardedGreedy},
      {"prep_greedy", PlanPolicy::PrepGreedy},
      {"limited_lookahead", PlanPolicy::LimitedLookahead},
      {"keel", PlanPolicy::Keel},
      {"hybrid", PlanPolicy::Hybrid},
      {"oracle", PlanPolicy::Oracle}};
  for (const auto& entry : names)
    if (name == entry.first) return entry.second;
  throw std::invalid_argument("unknown benchmark policy: " + name);
}

int64_t graph_bytes(const RuntimeGraph& graph) {
  int64_t total = 0;
  for (int64_t bytes : graph.tensor_bytes_all())
    if (__builtin_add_overflow(total, bytes, &total))
      throw std::overflow_error("evaluation graph byte sum overflows");
  return total;
}

int64_t graph_scratch_bytes(const RuntimeGraph& graph) {
  int64_t total = 0;
  for (const OpSpec& op : graph.ops())
    if (__builtin_add_overflow(total, op.scratch, &total))
      throw std::overflow_error("evaluation scratch byte sum overflows");
  return total;
}

int64_t graph_peak_scratch_bytes(const RuntimeGraph& graph) {
  int64_t peak = 0;
  for (const OpSpec& op : graph.ops()) peak = std::max(peak, op.scratch);
  return peak;
}

int64_t required_output_bytes(const RuntimeGraph& graph) {
  int64_t total = 0;
  for (TensorId t = 0; t < static_cast<TensorId>(graph.tensors().size()); ++t) {
    if (!graph.tensors()[t].required) continue;
    if (__builtin_add_overflow(total, graph.tensor_bytes(t), &total))
      throw std::overflow_error("evaluation required-output byte sum overflows");
  }
  return total;
}

const char* size_class(const RuntimeGraph& graph) {
  const int64_t bytes = graph_bytes(graph);
  if (bytes <= 16 * 1024) return "small";
  if (bytes <= 64 * 1024) return "medium";
  return "large";
}

int64_t remaining_work(const RuntimeGraph& graph, const TaskRuntime& task) {
  int64_t total = 0;
  for (OpId v = 0; v < static_cast<OpId>(graph.ops().size()); ++v) {
    if (task.op_state(v) != OpState::Pending) continue;
    if (__builtin_add_overflow(total, graph.ops()[v].cost, &total))
      throw std::overflow_error("evaluation remaining-work sum overflows");
  }
  return total;
}

void load_sources(TaskRuntime& task, const RuntimeGraph& graph,
                  const std::vector<uint64_t>& seeds) {
  for (OpId v = 0; v < static_cast<OpId>(graph.ops().size()); ++v)
    if (graph.ops()[v].kind == OpKind::Source)
      task.load_source(v, seeds[static_cast<size_t>(v)]);
}

bool prepare_frontier(TaskRuntime& task, const RuntimeGraph& graph,
                      double fraction) {
  const double bounded = std::max(0.0, std::min(1.0, fraction));
  const size_t count = static_cast<size_t>(bounded * graph.topo().size());
  for (size_t i = 0; i < count; ++i) {
    const OpId v = graph.topo()[i];
    if (task.op_state(v) == OpState::Done) continue;
    if (task.op_state(v) == OpState::InFlight) {
      task.wait_op(v);
      continue;
    }
    if (!task.try_dispatch_public(v)) return false;
    task.wait_op(v);
  }
  return true;
}

double fragmentation_ratio(const BlockArena& arena, TaskId task) {
  const int64_t free = arena.free_blocks(task);
  if (free <= 0) return 0.0;
  return 1.0 - static_cast<double>(arena.largest_free_run_blocks(task)) /
                 static_cast<double>(free);
}

bool required_outputs_match(TaskRuntime& task, const RuntimeGraph& graph,
                            const std::vector<std::vector<double>>& reference) {
  for (TensorId t = 0; t < static_cast<TensorId>(graph.tensors().size()); ++t) {
    if (!graph.tensors()[t].required) continue;
    if (!task.tensor_valid(t)) return false;
    const double* got = static_cast<const double*>(task.tensor_ptr(t));
    for (size_t i = 0; i < reference[static_cast<size_t>(t)].size(); ++i)
      if (got[i] != reference[static_cast<size_t>(t)][i]) return false;
  }
  return true;
}

struct Baseline {
  bool a_ok = false, b_ok = false, a_correct = false, b_correct = false;
  int64_t a_us = 0, b_us = 0;
};

struct SharedCounterfactual {
  bool a_ok = false, a_correct = false;
  bool b_admitted = false, b_ok = false, b_correct = false;
  bool invariants = false;
  int64_t a_completion_us = 0, b_admission_us = 0;
  int64_t b_first_useful_us = 0, b_completion_us = 0, b_deadline_us = 0;
  int64_t joint_makespan_us = 0, transferred_bytes = 0;
  int64_t reused_bytes = 0;
};

void install_reuse_observer(TaskRuntime& task, const RuntimeGraph& graph,
                            BlockArena& arena,
                            const std::vector<int32_t>& transferred,
                            Clock::time_point request_start,
                            int64_t* first_useful_us,
                            int64_t* reused_bytes) {
  const auto metrics_mu = std::make_shared<std::mutex>();
  task.set_worker_hook([&, transferred, request_start, first_useful_us,
                        reused_bytes, metrics_mu](OpId op, WorkerEvent event) {
    if (event != WorkerEvent::BeforeExecute ||
        graph.ops()[op].kind == OpKind::Source)
      return;
    std::set<int32_t> seen;
    for (TensorId output : graph.ops()[op].outputs) {
      const AllocRecord allocation =
          arena.record(task.tensor_allocation(output));
      for (int32_t block = allocation.first_block;
           block < allocation.first_block + allocation.nblocks; ++block)
        if (std::find(transferred.begin(), transferred.end(), block) !=
            transferred.end())
          seen.insert(block);
    }
    std::lock_guard<std::mutex> lk(*metrics_mu);
    if (*first_useful_us == 0) *first_useful_us = elapsed_us(request_start);
    for (size_t i = 0; i < seen.size(); ++i)
      *reused_bytes += arena.block_size();
  });
}

bool release_reclaimable(TaskRuntime& task, const RuntimeGraph& graph,
                         TaskId recipient, int64_t block_size,
                         int64_t* transferred_bytes,
                         std::vector<int32_t>* transferred_blocks) {
  int64_t bytes = 0;
  for (TensorId t = 0; t < static_cast<TensorId>(graph.tensors().size()); ++t) {
    if (graph.tensors()[t].required || graph.tensors()[t].pinned ||
        !task.tensor_resident(t))
      continue;
    const std::vector<int32_t> moved = task.release_tensor(t, recipient);
    transferred_blocks->insert(transferred_blocks->end(), moved.begin(),
                               moved.end());
    int64_t moved_bytes = 0;
    if (__builtin_mul_overflow(static_cast<int64_t>(moved.size()), block_size,
                               &moved_bytes) ||
        __builtin_add_overflow(bytes, moved_bytes, &bytes))
      throw std::overflow_error("counterfactual transfer size overflows");
  }
  *transferred_bytes = bytes;
  return true;
}

SharedCounterfactual run_shared_wait_baseline(
    const RuntimeGraph& graph, const std::vector<uint64_t>& a_seeds,
    const std::vector<uint64_t>& b_seeds, double frontier_fraction,
    int64_t old_grant, int64_t b_grant, int64_t capacity_bytes,
    const std::vector<std::vector<double>>& a_reference,
    const std::vector<std::vector<double>>& b_reference) {
  SharedCounterfactual out;
  BlockArena arena(capacity_bytes, 256);
  TaskRuntime task_a(graph, arena, 0, old_grant, 2);
  TaskRuntime task_b(graph, arena, 1, b_grant, 2);
  load_sources(task_a, graph, a_seeds);
  load_sources(task_b, graph, b_seeds);
  if (!prepare_frontier(task_a, graph, frontier_fraction)) return out;
  const auto request_start = Clock::now();
  std::atomic<bool> a_finished{false};
  std::mutex observation_mu;
  task_b.set_worker_hook([&](OpId op, WorkerEvent event) {
    if (event != WorkerEvent::BeforeExecute ||
        graph.ops()[op].kind == OpKind::Source)
      return;
    std::lock_guard<std::mutex> lk(observation_mu);
    if (!out.b_admitted) {
      out.b_admitted = true;
      out.b_admission_us = elapsed_us(request_start);
    }
    if (out.b_first_useful_us == 0)
      out.b_first_useful_us = elapsed_us(request_start);
  });

  // Work-conserving no-handover control: both controllers are released from
  // the same timing origin. B uses only its original grant and begins as soon
  // as its own allocations permit. If it stalls, retry only while A is still
  // live; after A finishes, the control is censored at its deadline rather
  // than converting an unfinished B into a zero-latency observation. No A
  // blocks are transferred in this control; that is the separate drain row.
  std::thread b_thread([&] {
    for (;;) {
      out.b_ok = task_b.run_to_completion();
      if (out.b_ok || a_finished.load(std::memory_order_acquire)) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (out.b_ok) out.b_completion_us = elapsed_us(request_start);
    else out.b_deadline_us = elapsed_us(request_start);
  });
  std::thread a_thread([&] {
    out.a_ok = task_a.run_to_completion();
    out.a_completion_us = elapsed_us(request_start);
    out.a_correct = out.a_ok &&
                    required_outputs_match(task_a, graph, a_reference);
    a_finished.store(true, std::memory_order_release);
  });
  a_thread.join();
  b_thread.join();
  if (out.b_ok) {
    out.b_deadline_us = out.b_completion_us;
  } else if (out.b_deadline_us == 0) {
    out.b_deadline_us = elapsed_us(request_start);
  }
  out.b_correct = out.b_ok &&
                  required_outputs_match(task_b, graph, b_reference);
  out.joint_makespan_us = std::max(out.a_completion_us, out.b_deadline_us);
  std::string error;
  out.invariants = arena.verify(&error) && task_a.invariants_hold(&error) &&
                   task_b.invariants_hold(&error);
  return out;
}

SharedCounterfactual run_shared_drain_baseline(
    const RuntimeGraph& graph, const std::vector<uint64_t>& a_seeds,
    const std::vector<uint64_t>& b_seeds, double frontier_fraction,
    int64_t old_grant, int64_t b_grant, int64_t capacity_bytes,
    const std::vector<std::vector<double>>& a_reference,
    const std::vector<std::vector<double>>& b_reference) {
  SharedCounterfactual out;
  BlockArena arena(capacity_bytes, 256);
  TaskRuntime task_a(graph, arena, 0, old_grant, 2);
  TaskRuntime task_b(graph, arena, 1, b_grant, 2);
  load_sources(task_a, graph, a_seeds);
  load_sources(task_b, graph, b_seeds);
  if (!prepare_frontier(task_a, graph, frontier_fraction)) return out;
  const auto request_start = Clock::now();
  out.a_ok = task_a.run_to_completion();
  out.a_completion_us = elapsed_us(request_start);
  out.a_correct = out.a_ok &&
                  required_outputs_match(task_a, graph, a_reference);
  if (!out.a_ok) return out;

  // Conservative drain is intentionally post-completion: A keeps its
  // required output, then gives B only reclaimable extents. No live planner
  // or pre-COMMIT overlap is credited to this baseline.
  std::vector<int32_t> transferred;
  release_reclaimable(task_a, graph, 1, arena.block_size(),
                      &out.transferred_bytes, &transferred);
  int64_t recipient_grant = 0;
  if (__builtin_add_overflow(b_grant, out.transferred_bytes,
                             &recipient_grant))
    throw std::overflow_error("drain recipient grant overflows");
  task_b.set_grant(recipient_grant);
  out.b_admitted = true;
  out.b_admission_us = elapsed_us(request_start);
  install_reuse_observer(task_b, graph, arena, transferred, request_start,
                         &out.b_first_useful_us, &out.reused_bytes);
  out.b_ok = task_b.run_to_completion();
  out.b_completion_us = out.b_ok ? elapsed_us(request_start) : 0;
  out.b_deadline_us = elapsed_us(request_start);
  out.b_correct = out.b_ok &&
                  required_outputs_match(task_b, graph, b_reference);
  out.joint_makespan_us = std::max(out.a_completion_us, out.b_deadline_us);
  std::string error;
  out.invariants = arena.verify(&error) && task_a.invariants_hold(&error) &&
                   task_b.invariants_hold(&error);
  return out;
}

Baseline measure_baseline(const RuntimeGraph& graph,
                          const std::vector<uint64_t>& a_seeds,
                          const std::vector<uint64_t>& b_seeds,
                          double frontier_fraction, int64_t old_grant,
                          int64_t b_grant) {
  Baseline baseline;
  {
    BlockArena arena(2 * 1024 * 1024, 256);
    TaskRuntime task(graph, arena, 0, old_grant, 2);
    load_sources(task, graph, a_seeds);
    if (!prepare_frontier(task, graph, frontier_fraction)) return baseline;
    const auto start = Clock::now();
    baseline.a_ok = task.run_to_completion();
    baseline.a_us = elapsed_us(start);
    const auto reference = reference_execute_f64(graph, a_seeds);
    baseline.a_correct = baseline.a_ok &&
                         required_outputs_match(task, graph, reference);
  }
  {
    BlockArena arena(2 * 1024 * 1024, 256);
    TaskRuntime task(graph, arena, 1, b_grant, 2);
    load_sources(task, graph, b_seeds);
    const auto start = Clock::now();
    baseline.b_ok = task.run_to_completion();
    baseline.b_us = elapsed_us(start);
    const auto reference = reference_execute_f64(graph, b_seeds);
    baseline.b_correct = baseline.b_ok &&
                         required_outputs_match(task, graph, reference);
  }
  return baseline;
}

struct Observation {
  int64_t first_use_us = 0, a_end_us = 0, b_end_us = 0;
  bool a_finished = false, useful_started = false, overlap = false;
  std::set<int32_t> reused_blocks;
  std::mutex mu;
};

std::string failure_stage(const HandoverResult& result, const Baseline& baseline,
                          bool a_ok, bool b_ok, bool a_correct, bool b_correct,
                          bool invariants) {
  if (!result.transfer_committed) {
    if (result.phase == HandoverPhase::Unknown) return "planner_unknown";
    if (result.planning_status == "checker_rejected") return "checker_rejected";
    if (result.planning_status == "translation_rejected")
      return "translation_rejected";
    if (result.note.find("preflight") != std::string::npos)
      return "physical_preflight_rejected";
    if (result.phase == HandoverPhase::Expired) return "expired";
    if (result.phase == HandoverPhase::Stale) return "stale_capture";
    if (result.phase == HandoverPhase::Failed) return "precommit_execution";
    return "planning_rejected";
  }
  if (!invariants) return "invariant_failure";
  if (!a_ok || !a_correct) return "task_a_failure";
  if (!b_ok || !b_correct)
    return baseline.b_ok ? "task_b_failure" : "task_b_base_infeasible";
  return "none";
}

void emit_row(FILE* output, const std::string& tier, BenchFamily family, int seed,
              int repeat, double pressure, double budget_fraction,
              double frontier_fraction, PlanPolicy policy,
              const RuntimeGraph& graph, const HandoverResult& result,
              const Baseline& baseline, int64_t old_grant, int64_t b_grant,
              const std::string& grant_profile, int64_t target_grant,
              const SharedCounterfactual& wait_baseline,
              const SharedCounterfactual& drain_baseline,
              int64_t prep_budget, int64_t remaining_request,
              int64_t remaining_commit, int64_t request_to_commit_us,
              int64_t a_us, int64_t b_us, int64_t first_use_us,
              int64_t reused_bytes, int64_t free_before,
              int64_t largest_before, double frag_before, int64_t free_after,
              int64_t b_free_after, int64_t b_largest_after,
              int64_t a_owned_before, int64_t b_owned_before,
              int64_t a_owned_after, int64_t b_owned_after,
              int64_t graph_scratch, int64_t graph_peak_scratch,
              int64_t required_outputs, int64_t a_live_before,
              int64_t b_live_before, int64_t a_peak_live,
              int64_t b_peak_live, int64_t b_free_before,
              int64_t b_largest_free_before,
              int64_t recomputations, bool a_ok, bool b_ok, bool a_correct,
              bool b_correct, bool invariants, bool overlap) {
  const bool joint = result.transfer_committed && a_ok && b_ok && a_correct &&
                     b_correct && invariants;
  const double slowdown = baseline.a_us > 0
                              ? static_cast<double>(a_us) / baseline.a_us
                              : 0.0;
  const double b_benefit = baseline.b_ok && b_ok
                               ? static_cast<double>(baseline.b_us - b_us)
                               : 0.0;
  const bool wait_b_benefit_valid = result.transfer_committed &&
                                    wait_baseline.b_ok && b_ok;
  const bool drain_b_benefit_valid = result.transfer_committed &&
                                     drain_baseline.b_ok && b_ok;
  const int64_t wait_b_benefit =
      wait_b_benefit_valid ? wait_baseline.b_completion_us - b_us : 0;
  const int64_t drain_b_benefit =
      drain_b_benefit_valid ? drain_baseline.b_completion_us - b_us : 0;
  const std::string fingerprint = bench_graph_fingerprint(graph);
  const std::string workload_id =
      std::string(bench_family_name(family)) + ":" + std::to_string(seed) +
      ":" + fingerprint;
  const auto optional_us = [](bool valid, int64_t value) {
    return valid ? std::to_string(value) : std::string("null");
  };
  const auto optional_double = [](bool valid, double value) {
    return valid ? std::to_string(value) : std::string("null");
  };
  const std::string stage = failure_stage(result, baseline, a_ok, b_ok,
                                          a_correct, b_correct, invariants);
  const std::string wait_admission =
      optional_us(wait_baseline.b_admitted, wait_baseline.b_admission_us);
  const std::string wait_first_useful = optional_us(
      wait_baseline.b_first_useful_us > 0, wait_baseline.b_first_useful_us);
  const std::string wait_completion =
      optional_us(wait_baseline.b_ok, wait_baseline.b_completion_us);
  const std::string wait_deadline =
      optional_us(wait_baseline.b_deadline_us > 0, wait_baseline.b_deadline_us);
  const std::string drain_admission =
      optional_us(drain_baseline.b_admitted, drain_baseline.b_admission_us);
  const std::string drain_first_useful = optional_us(
      drain_baseline.b_first_useful_us > 0, drain_baseline.b_first_useful_us);
  const std::string drain_completion =
      optional_us(drain_baseline.b_ok, drain_baseline.b_completion_us);
  const std::string drain_deadline = optional_us(
      drain_baseline.b_deadline_us > 0, drain_baseline.b_deadline_us);
  const std::string wait_a_slowdown = optional_double(
      baseline.a_us > 0 && wait_baseline.a_ok,
      static_cast<double>(wait_baseline.a_completion_us) / baseline.a_us);
  const std::string drain_a_slowdown = optional_double(
      baseline.a_us > 0 && drain_baseline.a_ok,
      static_cast<double>(drain_baseline.a_completion_us) / baseline.a_us);
  std::fprintf(
      output,
      "{\"tier\":%s,\"family\":%s,\"workload_instance_id\":%s,"
      "\"graph_fingerprint\":%s,\"size_class\":%s,\"seed\":%d,"
      "\"repeat\":%d,\"policy\":%s,\"pressure\":%.4f,"
      "\"budget_fraction\":%.4f,\"frontier_fraction\":%.4f,"
      "\"grant_profile\":%s,"
      "\"shared_capacity_bytes\":%lld,\"old_grant_bytes\":%lld,"
      "\"new_grant_bytes\":%lld,"
      "\"b_grant_bytes\":%lld,\"prep_budget\":%lld,"
      "\"graph_ops\":%zu,\"graph_tensors\":%zu,\"graph_bytes\":%lld,"
      "\"graph_scratch_bytes\":%lld,\"graph_peak_scratch_bytes\":%lld,"
      "\"required_output_bytes\":%lld,\"a_live_bytes_before\":%lld,"
      "\"b_live_bytes_before\":%lld,\"a_peak_live_bytes\":%lld,"
      "\"b_peak_live_bytes\":%lld,\"b_free_blocks_before\":%lld,"
      "\"b_largest_free_run_before\":%lld,"
      "\"remaining_work_request\":%lld,\"remaining_work_commit\":%lld,"
      "\"planning_status\":%s,\"phase\":%s,\"failure_stage\":%s,"
      "\"accepted\":%s,\"physical_preflight_ok\":%s,"
      "\"committed\":%s,\"joint_success\":%s,\"a_ok\":%s,"
      "\"b_ok\":%s,\"a_correct\":%s,\"b_correct\":%s,"
      "\"invariants\":%s,\"requested_bytes\":%lld,"
      "\"transferred_bytes\":%lld,\"reused_bytes\":%lld,"
      "\"b_grant_after_bytes\":%lld,"
      "\"baseline_a_ok\":%s,\"baseline_a_correct\":%s,"
      "\"baseline_b_ok\":%s,\"baseline_b_correct\":%s,"
      "\"standalone_b_us\":%lld,\"standalone_b_ok\":%s,"
      "\"wait_b_admitted\":%s,\"wait_b_ok\":%s,"
      "\"wait_b_correct\":%s,\"wait_b_admission_us\":%s,"
      "\"wait_b_first_useful_us\":%s,\"wait_b_completion_us\":%s,"
      "\"wait_b_deadline_us\":%s,\"wait_a_slowdown_ratio\":%s,"
      "\"wait_a_completion_us\":%lld,\"wait_joint_makespan_us\":%lld,"
      "\"wait_joint_completion\":%s,\"wait_b_benefit_us\":%s,"
      "\"drain_b_admitted\":%s,\"drain_b_ok\":%s,"
      "\"drain_b_correct\":%s,\"drain_b_admission_us\":%s,"
      "\"drain_b_first_useful_us\":%s,\"drain_b_completion_us\":%s,"
      "\"drain_b_deadline_us\":%s,\"drain_a_slowdown_ratio\":%s,"
      "\"drain_a_completion_us\":%lld,\"drain_joint_makespan_us\":%lld,"
      "\"drain_joint_completion\":%s,\"drain_transferred_bytes\":%lld,"
      "\"drain_reused_bytes\":%lld,\"drain_b_benefit_us\":%s,"
      "\"transferred_blocks\":%zu,\"overlap_observed\":%s,"
      "\"capture_us\":%lld,\"plan_us\":%lld,\"checker_us\":%lld,"
      "\"preflight_us\":%lld,\"transfer_us\":%lld,\"commit_us\":%lld,"
      "\"handover_total_us\":%lld,"
      "\"wait_us\":%lld,\"request_to_commit_us\":%lld,"
      "\"request_to_b_useful_us\":%lld,\"request_to_b_completion_us\":%lld,"
      "\"request_to_a_completion_us\":%lld,\"end_to_end_us\":%lld,"
      "\"baseline_a_us\":%lld,\"baseline_b_us\":%lld,"
      "\"a_slowdown_ratio\":%.6f,\"standalone_lower_bound_delta_us\":%.1f,"
      "\"prep_compute\":%lld,\"cont_compute\":%lld,"
      "\"recompute_compute\":%lld,\"recomputations\":%lld,"
      "\"planner_nodes\":%zu,\"candidates_checked\":%zu,"
      "\"checker_calls\":%zu,\"a_free_blocks_before\":%lld,"
      "\"a_largest_free_run_before\":%lld,\"fragmentation_before\":%.6f,"
      "\"a_free_blocks_after\":%lld,\"b_free_blocks_after\":%lld,"
      "\"b_largest_free_run_after\":%lld,\"a_owned_before\":%lld,"
      "\"b_owned_before\":%lld,\"a_owned_after\":%lld,"
      "\"b_owned_after\":%lld}\n",
      json_string(tier).c_str(), json_string(bench_family_name(family)).c_str(),
      json_string(workload_id).c_str(), json_string(fingerprint).c_str(),
      json_string(size_class(graph)).c_str(), seed, repeat,
      json_string(plan_policy_name(policy)).c_str(), pressure, budget_fraction,
      frontier_fraction, json_string(grant_profile).c_str(),
      (long long)2 * 1024 * 1024, (long long)old_grant,
      (long long)target_grant,
      (long long)b_grant, (long long)prep_budget, graph.ops().size(),
      graph.tensors().size(), (long long)graph_bytes(graph),
      (long long)graph_scratch, (long long)graph_peak_scratch,
      (long long)required_outputs, (long long)a_live_before,
      (long long)b_live_before, (long long)a_peak_live,
      (long long)b_peak_live, (long long)b_free_before,
      (long long)b_largest_free_before,
      (long long)remaining_request, (long long)remaining_commit,
      json_string(result.planning_status).c_str(),
      json_string(handover_phase_name(result.phase)).c_str(),
      json_string(stage).c_str(),
      bool_text(result.planning_status == "accepted"),
      bool_text(result.physical_preflight_ok), bool_text(result.transfer_committed),
      bool_text(joint), bool_text(a_ok), bool_text(b_ok), bool_text(a_correct),
      bool_text(b_correct), bool_text(invariants),
      (long long)result.requested_release_bytes, (long long)result.released_bytes,
      (long long)reused_bytes,
      (long long)(b_grant + result.released_bytes), bool_text(baseline.a_ok),
      bool_text(baseline.a_correct), bool_text(baseline.b_ok),
      bool_text(baseline.b_correct), (long long)baseline.b_us,
      bool_text(baseline.b_ok),
      bool_text(wait_baseline.b_admitted), bool_text(wait_baseline.b_ok),
      bool_text(wait_baseline.b_correct), wait_admission.c_str(),
      wait_first_useful.c_str(), wait_completion.c_str(),
      wait_deadline.c_str(), wait_a_slowdown.c_str(),
      (long long)wait_baseline.a_completion_us,
      (long long)wait_baseline.joint_makespan_us,
      bool_text(wait_baseline.a_ok && wait_baseline.a_correct &&
                wait_baseline.b_ok && wait_baseline.b_correct &&
                wait_baseline.invariants),
      optional_us(wait_b_benefit_valid, wait_b_benefit).c_str(),
      bool_text(drain_baseline.b_admitted), bool_text(drain_baseline.b_ok),
      bool_text(drain_baseline.b_correct),
      drain_admission.c_str(), drain_first_useful.c_str(),
      drain_completion.c_str(), drain_deadline.c_str(),
      drain_a_slowdown.c_str(),
      (long long)drain_baseline.a_completion_us,
      (long long)drain_baseline.joint_makespan_us,
      bool_text(drain_baseline.a_ok && drain_baseline.a_correct &&
                drain_baseline.b_ok && drain_baseline.b_correct &&
                drain_baseline.invariants),
      (long long)drain_baseline.transferred_bytes,
      (long long)drain_baseline.reused_bytes,
      optional_us(drain_b_benefit_valid, drain_b_benefit).c_str(),
      result.transferred_blocks.size(), bool_text(overlap),
      (long long)result.capture_us, (long long)result.plan_us,
      (long long)result.checker_us, (long long)result.preflight_us,
      (long long)result.transfer_us, (long long)result.commit_us,
      (long long)result.total_us,
      (long long)result.wait_us, (long long)request_to_commit_us,
      (long long)first_use_us, (long long)b_us, (long long)a_us,
      (long long)std::max(a_us, b_us), (long long)baseline.a_us,
      (long long)baseline.b_us, slowdown, b_benefit,
      (long long)result.prep_compute, (long long)result.cont_compute,
      (long long)result.recompute_compute, (long long)recomputations,
      result.outer_nodes + result.cont_nodes, result.candidates_checked,
      result.checker_invocations, (long long)free_before,
      (long long)largest_before, frag_before, (long long)free_after,
      (long long)b_free_after, (long long)b_largest_after,
      (long long)a_owned_before, (long long)b_owned_before,
      (long long)a_owned_after, (long long)b_owned_after);
}

struct Options {
  std::string tier = "custom", out_path;
  std::string grant_profile = "custom";
  int seed_lo = 1, seed_hi = 1, repeats = 1;
  int64_t old_grant = 512 * 1024, b_grant = 64 * 1024;
  std::vector<std::string> family_names, policy_names;
  std::vector<double> pressures, budgets, frontiers;
};

void set_defaults(Options& options) {
  if (options.family_names.empty())
    for (int i = 0; i < bench_family_count(); ++i)
      options.family_names.push_back(
          bench_family_name(static_cast<BenchFamily>(i)));
  if (options.policy_names.empty())
    options.policy_names = {"conservative", "greedy", "guarded_greedy",
                            "prep_greedy", "limited_lookahead", "keel",
                            "hybrid"};
  if (options.pressures.empty()) options.pressures = {0.60, 0.75, 0.90};
  if (options.budgets.empty())
    options.budgets = {0.0, 0.10, 0.25, 0.50, 1.0};
  if (options.frontiers.empty()) options.frontiers = {0.50};
}

void parse_args(int argc, char** argv, Options& options) {
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    auto next = [&](const char* name) -> const char* {
      if (i + 1 >= argc)
        throw std::invalid_argument(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (!std::strcmp(arg, "--tier")) options.tier = next(arg);
    else if (!std::strcmp(arg, "--out")) options.out_path = next(arg);
    else if (!std::strcmp(arg, "--seed-lo")) options.seed_lo = std::atoi(next(arg));
    else if (!std::strcmp(arg, "--seed-hi")) options.seed_hi = std::atoi(next(arg));
    else if (!std::strcmp(arg, "--repeats")) options.repeats = std::atoi(next(arg));
    else if (!std::strcmp(arg, "--grant-profile"))
      options.grant_profile = next(arg);
    else if (!std::strcmp(arg, "--old-grant-kb"))
      options.old_grant = std::atoll(next(arg)) * 1024;
    else if (!std::strcmp(arg, "--b-grant-kb"))
      options.b_grant = std::atoll(next(arg)) * 1024;
    else if (!std::strcmp(arg, "--families"))
      options.family_names = split(next(arg));
    else if (!std::strcmp(arg, "--policies"))
      options.policy_names = split(next(arg));
    else if (!std::strcmp(arg, "--pressures"))
      options.pressures = parse_doubles(next(arg));
    else if (!std::strcmp(arg, "--budgets"))
      options.budgets = parse_doubles(next(arg));
    else if (!std::strcmp(arg, "--frontiers"))
      options.frontiers = parse_doubles(next(arg));
    else throw std::invalid_argument(std::string("unknown argument: ") + arg);
  }
  if (options.seed_lo > options.seed_hi || options.repeats <= 0)
    throw std::invalid_argument("invalid seed or repeat range");
  set_defaults(options);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options options;
    parse_args(argc, argv, options);
    if (options.out_path.empty()) throw std::invalid_argument("--out is required");
    FILE* output = std::fopen(options.out_path.c_str(), "w");
    if (!output) throw std::runtime_error("cannot open evaluation output");

    const int64_t block_size = 256;
    for (const std::string& family_name : options.family_names) {
      const BenchFamily family = parse_family(family_name);
      for (int seed = options.seed_lo; seed <= options.seed_hi; ++seed) {
        RuntimeGraph graph = make_bench_graph(family, static_cast<uint64_t>(seed));
        const std::vector<uint64_t> a_seeds =
            bench_source_seeds(graph, static_cast<uint64_t>(seed) * 1009 + 7);
        const std::vector<uint64_t> b_seeds =
            bench_source_seeds(graph, static_cast<uint64_t>(seed) * 1009 + 10007);
        const auto a_reference = reference_execute_f64(graph, a_seeds);
        const auto b_reference = reference_execute_f64(graph, b_seeds);
        const int64_t graph_scratch = graph_scratch_bytes(graph);
        const int64_t graph_peak_scratch = graph_peak_scratch_bytes(graph);
        const int64_t required_outputs = required_output_bytes(graph);
        for (int repeat = 0; repeat < options.repeats; ++repeat) {
          for (double frontier : options.frontiers) {
            const Baseline baseline = measure_baseline(
                graph, a_seeds, b_seeds, frontier, options.old_grant,
                options.b_grant);
            const SharedCounterfactual wait_baseline =
                run_shared_wait_baseline(
                    graph, a_seeds, b_seeds, frontier, options.old_grant,
                    options.b_grant, 2 * 1024 * 1024, a_reference, b_reference);
            const SharedCounterfactual drain_baseline =
                run_shared_drain_baseline(
                    graph, a_seeds, b_seeds, frontier, options.old_grant,
                    options.b_grant, 2 * 1024 * 1024, a_reference, b_reference);
            for (double pressure : options.pressures) {
              int64_t new_grant =
                  static_cast<int64_t>(options.old_grant * pressure);
              new_grant = (new_grant / block_size) * block_size;
              for (double budget_fraction : options.budgets) {
                for (const std::string& policy_name : options.policy_names) {
                  const PlanPolicy policy = parse_policy(policy_name);
                  if (policy == PlanPolicy::Oracle && graph.ops().size() > 10)
                    continue;
                  BlockArena arena(2 * 1024 * 1024, block_size);
                  TaskRuntime task_a(graph, arena, 0, options.old_grant, 2);
                  TaskRuntime task_b(graph, arena, 1, options.b_grant, 2);
                  load_sources(task_a, graph, a_seeds);
                  load_sources(task_b, graph, b_seeds);
                  if (!prepare_frontier(task_a, graph, frontier)) continue;
                  const int64_t remaining_request = remaining_work(graph, task_a);
                  const int64_t prep_budget = static_cast<int64_t>(
                      static_cast<double>(remaining_request) * budget_fraction);
                  const int64_t free_before = arena.free_blocks(0);
                  const int64_t largest_before =
                      arena.largest_free_run_blocks(0);
                  const double frag_before = fragmentation_ratio(arena, 0);
                  const int64_t a_owned_before = arena.blocks_owned(0);
                  const int64_t b_owned_before = arena.blocks_owned(1);
                  const int64_t a_live_before = task_a.controlled_bytes();
                  const int64_t b_live_before = task_b.controlled_bytes();
                  const int64_t b_free_before = arena.free_blocks(1);
                  const int64_t b_largest_free_before =
                      arena.largest_free_run_blocks(1);
                  const auto request_start = Clock::now();
                  HandoverResult result = perform_handover(
                      task_a, arena, 1, new_grant, prep_budget, policy,
                      /*stop_after_commit=*/true, &task_b);
                  const int64_t request_to_commit_us =
                      result.transfer_committed ? elapsed_us(request_start) : 0;
                  const int64_t remaining_commit = remaining_work(graph, task_a);
                  Observation observation;
                  bool a_ok = false, b_ok = false;
                  int64_t a_us = 0, b_us = 0;
                  if (result.transfer_committed) {
                    task_b.set_worker_hook([&](OpId op, WorkerEvent event) {
                      if (event != WorkerEvent::BeforeExecute ||
                          graph.ops()[op].kind == OpKind::Source)
                        return;
                      std::lock_guard<std::mutex> lk(observation.mu);
                      for (TensorId output : graph.ops()[op].outputs) {
                        const AllocRecord allocation =
                            arena.record(task_b.tensor_allocation(output));
                        for (int32_t block = allocation.first_block;
                             block < allocation.first_block + allocation.nblocks;
                             ++block) {
                          if (std::find(result.transferred_blocks.begin(),
                                        result.transferred_blocks.end(),
                                        block) != result.transferred_blocks.end())
                            observation.reused_blocks.insert(block);
                        }
                      }
                      if (!observation.useful_started) {
                        observation.useful_started = true;
                        observation.first_use_us = elapsed_us(request_start);
                        observation.overlap = remaining_commit > 0 &&
                                              !observation.a_finished;
                      }
                    });
                    std::thread a_thread([&] {
                      std::string why;
                      a_ok = continue_handover(task_a, result, &why);
                      const int64_t elapsed = elapsed_us(request_start);
                      std::lock_guard<std::mutex> lk(observation.mu);
                      observation.a_end_us = elapsed;
                      observation.a_finished = true;
                    });
                    std::thread b_thread([&] {
                      b_ok = task_b.run_to_completion();
                      const int64_t elapsed = elapsed_us(request_start);
                      std::lock_guard<std::mutex> lk(observation.mu);
                      observation.b_end_us = elapsed;
                    });
                    a_thread.join();
                    b_thread.join();
                  } else {
                    a_ok = task_a.run_to_completion();
                    a_us = elapsed_us(request_start);
                    const auto b_start = Clock::now();
                    b_ok = task_b.run_to_completion();
                    b_us = elapsed_us(b_start);
                  }
                  {
                    std::lock_guard<std::mutex> lk(observation.mu);
                    if (result.transfer_committed) {
                      a_us = observation.a_end_us;
                      b_us = observation.b_end_us;
                    }
                  }
                  const bool a_correct = a_ok &&
                                         required_outputs_match(task_a, graph,
                                                                a_reference);
                  const bool b_correct = b_ok &&
                                         required_outputs_match(task_b, graph,
                                                                b_reference);
                  std::string error;
                  const bool invariants = arena.verify(&error) &&
                                          task_a.invariants_hold(&error) &&
                                          task_b.invariants_hold(&error);
                  const int64_t free_after = arena.free_blocks(0);
                  const int64_t b_free_after = arena.free_blocks(1);
                  const int64_t b_largest_after =
                      arena.largest_free_run_blocks(1);
                  const int64_t a_owned_after = arena.blocks_owned(0);
                  const int64_t b_owned_after = arena.blocks_owned(1);
                  const int64_t reused_bytes = static_cast<int64_t>(
                      observation.reused_blocks.size()) * block_size;
                  emit_row(output, options.tier, family, seed, repeat, pressure,
                           budget_fraction, frontier, policy, graph, result,
                           baseline, options.old_grant, options.b_grant,
                           options.grant_profile, new_grant, wait_baseline,
                           drain_baseline,
                           prep_budget, remaining_request,
                           remaining_commit,
                           request_to_commit_us, a_us, b_us,
                           observation.first_use_us, reused_bytes, free_before,
                           largest_before, frag_before, free_after, b_free_after,
                           b_largest_after, a_owned_before, b_owned_before,
                           a_owned_after, b_owned_after,
                           graph_scratch, graph_peak_scratch, required_outputs,
                           a_live_before, b_live_before,
                           task_a.stats().peak_live_bytes,
                           task_b.stats().peak_live_bytes, b_free_before,
                           b_largest_free_before,
                           task_a.stats().recomputations, a_ok, b_ok, a_correct,
                           b_correct, invariants, observation.overlap);
                }
              }
            }
          }
        }
      }
    }
    std::fclose(output);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "keel_eval: %s\n", e.what());
    return 2;
  }
}
