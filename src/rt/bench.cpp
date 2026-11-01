// Phase 7 — real-runtime benchmark: physical handover, joint completion,
// independent numerical checks, and honest overlap/timing accounting.

#include <keel/rt/bench_gen.hpp>
#include <keel/rt/handover.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace keel;
using namespace keel::rt;

namespace {

int64_t now_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

const PlanPolicy kPolicies[] = {
    PlanPolicy::Conservative, PlanPolicy::GreedyUnguarded,
    PlanPolicy::GuardedGreedy, PlanPolicy::PrepGreedy,
    PlanPolicy::LimitedLookahead, PlanPolicy::Keel, PlanPolicy::Hybrid,
    PlanPolicy::Oracle};

bool required_outputs_match(TaskRuntime& task, const RuntimeGraph& graph,
                            const std::vector<std::vector<double>>& reference) {
  for (TensorId t = 0; t < (TensorId)graph.tensors().size(); ++t) {
    if (!graph.tensors()[t].required) continue;
    if (!task.tensor_valid(t)) return false;
    const double* got = static_cast<const double*>(task.tensor_ptr(t));
    for (size_t i = 0; i < reference[t].size(); ++i)
      if (got[i] != reference[t][i]) return false;
  }
  return true;
}

void emit(const std::string& line) { std::printf("%s\n", line.c_str()); }

}  // namespace

int main(int argc, char** argv) {
  int seed_lo = 1, seed_hi = 6;
  const char* out_path = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--seed-lo") && i + 1 < argc)
      seed_lo = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seed-hi") && i + 1 < argc)
      seed_hi = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--out") && i + 1 < argc)
      out_path = argv[++i];
  }
  if (out_path) freopen(out_path, "w", stdout);

  const double pressures[] = {0.60, 0.75, 0.90};  // Bn / Bo
  const int64_t bo = 512 * 1024;

  for (int fi = 0; fi < bench_family_count(); ++fi) {
    const BenchFamily family = (BenchFamily)fi;
    for (int seed = seed_lo; seed <= seed_hi; ++seed) {
      RuntimeGraph ga_graph = make_fixed_bench_graph(family, (uint64_t)seed);
      const auto a_seeds = bench_source_seeds(ga_graph, (uint64_t)seed);
      const auto a_reference = reference_execute_f64(ga_graph, a_seeds);

      // Unconstrained A-only baseline. It is explicitly not a joint row.
      {
        BlockArena arena(1 << 20, 256);
        TaskRuntime task(ga_graph, arena, 0, bo, 2);
        const int64_t t0 = now_us();
        for (size_t v = 0; v < ga_graph.ops().size(); ++v)
          if (ga_graph.ops()[v].kind == OpKind::Source)
            task.load_source((OpId)v, a_seeds[v]);
        const bool ok = task.run_to_completion();
        const bool correct = ok &&
                             required_outputs_match(task, ga_graph, a_reference);
        std::string error;
        const bool inv = arena.verify(&error) && task.invariants_hold(&error);
        const int64_t elapsed = now_us() - t0;
        char buf[2048];
        std::snprintf(
            buf, sizeof buf,
            "{\"family\":\"%s\",\"seed\":%d,\"policy\":\"none\","
            "\"pressure\":0,\"accepted\":false,\"committed\":false,"
            "\"phase\":\"BASELINE\",\"a_us\":%lld,\"b_us\":0,"
            "\"total_us\":%lld,\"a_ok\":%s,\"b_ok\":true,"
            "\"a_correct\":%s,\"b_correct\":true,\"correct\":%s,"
            "\"invariants\":%s,\"joint_success\":%s,"
            "\"overlap_observed\":false,\"b_used_transferred\":false}",
            bench_family_name(family), seed, (long long)elapsed,
            (long long)elapsed, ok ? "true" : "false",
            correct ? "true" : "false", correct ? "true" : "false",
            inv ? "true" : "false", "false");
        emit(buf);
      }

      for (double pressure : pressures) {
        int64_t bn = (int64_t)(bo * pressure);
        bn = (bn / 256) * 256;
        for (PlanPolicy policy : kPolicies) {
          BlockArena arena(1 << 20, 256);
          TaskRuntime ta(ga_graph, arena, 0, bo, 2);
          RuntimeGraph gb_graph = make_fixed_bench_graph(family, (uint64_t)seed + 1000);
          const auto b_seeds =
              bench_source_seeds(gb_graph, (uint64_t)seed + 1000);
          const auto b_reference = reference_execute_f64(gb_graph, b_seeds);
          TaskRuntime tb(gb_graph, arena, 1, 64 * 1024, 2);

          for (size_t v = 0; v < ga_graph.ops().size(); ++v)
            if (ga_graph.ops()[v].kind == OpKind::Source)
              ta.load_source((OpId)v, a_seeds[v]);
          for (size_t v = 0; v < gb_graph.ops().size(); ++v)
            if (gb_graph.ops()[v].kind == OpKind::Source)
              tb.load_source((OpId)v, b_seeds[v]);

          const size_t mid = ga_graph.ops().size() / 2;
          bool partial_ok = true;
          for (size_t i = 0; i < mid && partial_ok; ++i) {
            const OpId v = ga_graph.topo()[i];
            if (ta.op_state(v) == OpState::Done) continue;
            if (ta.op_state(v) == OpState::InFlight) {
              ta.wait_op(v);
              continue;
            }
            partial_ok = ta.try_dispatch_public(v);
            if (partial_ok) ta.wait_op(v);
          }

          const int64_t t0 = now_us();
          HandoverResult result = perform_handover(
              ta, arena, 1, bn, 100, policy, /*stop_after_commit=*/true, &tb);
          const int64_t t1 = now_us();

          bool a_ok = false, b_ok = false;
          int64_t a_end = 0, b_end = 0;
          int64_t b_first_use_us = 0;
          bool overlap_observed = false;
          bool b_used_transferred = false;
          if (result.accepted) {
            std::mutex mu;
            std::condition_variable cv;
            bool a_at_boundary = false;
            bool b_started = false;
            bool release_a = false;
            bool a_finished = false;
            ta.set_worker_hook([&](OpId, WorkerEvent event) {
              if (event != WorkerEvent::BeforeComplete) return;
              std::unique_lock<std::mutex> lk(mu);
              if (a_at_boundary) return;
              a_at_boundary = true;
              cv.notify_all();
              cv.wait(lk, [&] { return release_a; });
            });
            tb.set_worker_hook([&](OpId op, WorkerEvent event) {
              if (event != WorkerEvent::BeforeExecute ||
                  gb_graph.ops()[op].kind == OpKind::Source)
                return;
              const AllocRecord rec = arena.record(tb.tensor_allocation(
                  gb_graph.ops()[op].outputs[0]));
              for (int32_t block : result.transferred_blocks)
                if (rec.first_block <= block &&
                    block < rec.first_block + rec.nblocks)
                  b_used_transferred = true;
              if (b_used_transferred && b_first_use_us == 0)
                b_first_use_us = now_us() - t0;
              std::lock_guard<std::mutex> lk(mu);
              if (!b_started) {
                b_started = true;
                overlap_observed = a_at_boundary && !a_finished;
              }
              release_a = true;
              cv.notify_all();
            });
            std::thread a_thread([&] {
              std::string why;
              a_ok = continue_handover(ta, result, &why);
              a_end = now_us();
              std::lock_guard<std::mutex> lk(mu);
              a_finished = true;
              cv.notify_all();
            });
            std::thread b_thread([&] {
              b_ok = tb.run_to_completion();
              b_end = now_us();
            });
            {
              std::unique_lock<std::mutex> lk(mu);
              if (!a_at_boundary)
                cv.wait_for(lk, std::chrono::seconds(5),
                            [&] { return a_at_boundary || a_finished; });
              if (a_at_boundary && !b_started)
                cv.wait_for(lk, std::chrono::seconds(5),
                            [&] { return b_started; });
              release_a = true;
            }
            cv.notify_all();
            a_thread.join();
            b_thread.join();
          } else {
            a_ok = ta.run_to_completion();
            a_end = now_us();
            b_ok = tb.run_to_completion();
            b_end = now_us();
          }
          const bool a_correct = a_ok &&
                                 required_outputs_match(ta, ga_graph, a_reference);
          const bool b_correct = b_ok &&
                                 required_outputs_match(tb, gb_graph, b_reference);
          std::string error;
          const bool invariants = arena.verify(&error) &&
                                  ta.invariants_hold(&error) &&
                                  tb.invariants_hold(&error);
          const bool joint_success = result.accepted && a_ok && b_ok &&
                                     a_correct && b_correct && invariants;
          const int64_t a_us = a_end > t0 ? a_end - t0 : 0;
          const int64_t b_us = b_end > t0 ? b_end - t0 : 0;
          const int64_t total_us = now_us() - t0;
          char buf[4096];
          std::snprintf(
              buf, sizeof buf,
              "{\"family\":\"%s\",\"seed\":%d,\"policy\":\"%s\","
              "\"pressure\":%.2f,\"accepted\":%s,\"committed\":%s,"
              "\"phase\":\"%s\",\"released\":%lld,\"requested\":%lld,"
              "\"plan_us\":%lld,\"checker_us\":%lld,\"handover_us\":%lld,"
              "\"a_us\":%lld,\"b_us\":%lld,\"total_us\":%lld,"
              "\"a_ok\":%s,\"b_ok\":%s,\"a_correct\":%s,"
              "\"b_correct\":%s,\"correct\":%s,\"invariants\":%s,"
              "\"joint_success\":%s,\"overlap_observed\":%s,"
              "\"b_used_transferred\":%s,\"continuation_actions\":%zu,"
              "\"b_first_use_us\":%lld,\"partial_frontier_ok\":%s,"
              "\"prep_compute\":%lld,\"cont_compute\":%lld,"
              "\"nodes\":%zu,\"candidates\":%zu,\"checker_calls\":%zu}",
              bench_family_name(family), seed, plan_policy_name(policy), pressure,
              result.accepted ? "true" : "false",
              result.accepted ? "true" : "false",
              handover_phase_name(result.phase), (long long)result.released_bytes,
              (long long)result.requested_release_bytes, (long long)result.plan_us,
              (long long)result.checker_us, (long long)(t1 - t0),
              (long long)a_us, (long long)b_us, (long long)total_us,
              a_ok ? "true" : "false", b_ok ? "true" : "false",
              a_correct ? "true" : "false", b_correct ? "true" : "false",
              (a_correct && b_correct) ? "true" : "false",
              invariants ? "true" : "false", joint_success ? "true" : "false",
              overlap_observed ? "true" : "false",
              b_used_transferred ? "true" : "false",
              result.continuation.acts.size(), (long long)b_first_use_us,
              partial_ok ? "true" : "false", (long long)result.prep_compute,
              (long long)result.cont_compute, result.outer_nodes + result.cont_nodes,
              result.candidates_checked, result.checker_invocations);
          emit(buf);
        }
      }
    }
  }
  fprintf(stderr, "done\n");
  return 0;
}
