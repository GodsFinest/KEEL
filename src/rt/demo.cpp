// KEEL Phase 5 — Live two-task memory-grant handover demonstration.
//
// A real runtime execution (not a transcript): Task A runs an actual
// numerical DAG on real arena storage; Task B requests part of Task A's
// allocation; KEEL captures, plans, validates, prepares, commits, and both
// tasks complete with reference-correct outputs. A deliberately over-sized
// request is also shown being REJECTED with zero transfer.
//
// The 13 observable properties (mandate Part V) are printed and asserted.

#include <keel/rt/handover.hpp>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

using namespace keel;
using namespace keel::rt;

namespace {

RuntimeGraph make_a_graph() {
  RuntimeGraph g;
  g.add_tensor({DType::F64, {32, 24}});
  g.add_tensor({DType::F64, {24, 16}});
  g.add_tensor({DType::F64, {32, 16}});
  g.add_tensor({DType::F64, {32, 1}});
  g.add_tensor({DType::F64, {32, 16}});
  g.add_tensor({DType::F64, {32, 16}, /*required=*/true});
  g.add_op({OpKind::Source, {}, {0}, 0, 1, false});
  g.add_op({OpKind::Source, {}, {1}, 0, 1, false});
  g.add_op({OpKind::ScratchGemm, {0, 1}, {2}, 24 * 16 * 8, 6, true});
  g.add_op({OpKind::ReduceSum, {2}, {3}, 0, 3, true});
  g.add_op({OpKind::Add, {3, 2}, {4}, 0, 2, true});
  g.add_op({OpKind::Mul, {4, 2}, {5}, 0, 2, true});
  g.finish();
  return g;
}

RuntimeGraph make_b_graph() {
  RuntimeGraph g;
  g.add_tensor({DType::F64, {16, 16}});
  g.add_tensor({DType::F64, {16, 16}});
  g.add_tensor({DType::F64, {16, 16}, /*required=*/true});
  g.add_op({OpKind::Source, {}, {0}, 0, 1, false});
  g.add_op({OpKind::Source, {}, {1}, 0, 1, false});
  g.add_op({OpKind::MatMul, {0, 1}, {2}, 0, 4, true});
  g.finish();
  return g;
}

int check(bool ok, const char* what) {
  std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
  return ok ? 0 : 1;
}

}  // namespace

int main() {
  int failures = 0;
  std::printf("=== KEEL live two-task memory-grant handover ===\n\n");

  // (1) Task A executes an actual numerical computation on real arena storage.
  BlockArena arena(1 << 20, 256);  // 1 MiB physical backing store
  RuntimeGraph ga = make_a_graph();
  RuntimeGraph gb = make_b_graph();
  auto ref_a = reference_execute_f64(ga, {7, 8});
  auto ref_b = reference_execute_f64(gb, {21, 22});

  TaskRuntime task_a(ga, arena, /*task=*/0, /*grant=*/256 * 1024, /*workers=*/2);
  TaskRuntime task_b(gb, arena, /*task=*/1, /*grant=*/20 * 256, /*workers=*/2);
  std::printf("initial grants: A = %lld B, B = %lld B (arena %lld B, block %lld)\n",
              (long long)task_a.grant_bytes(), (long long)task_b.grant_bytes(),
              (long long)arena.capacity_bytes(), (long long)arena.block_size());
  failures += check(task_a.grant_bytes() == 256 * 1024 &&
                        task_b.grant_bytes() == 20 * 256,
                    "1. Task A executing a real computation on real arena storage");

  // Task A: run sources + gemm, then hold C resident mid-execution.
  task_a.load_source(0, 7);
  task_a.load_source(1, 8);
  task_a.dispatch_ready();
  task_a.wait_op(2);
  const int64_t a_before = arena.blocks_owned(0);
  const int64_t b_before = arena.blocks_owned(1);
  std::printf("mid-execution: A holds %lld B live (gemm done, C resident)\n",
              (long long)task_a.controlled_bytes());

  // B preloads its sources. Its small grant leaves four free blocks, so the
  // eight-block output must use the transferred extent.
  task_b.load_source(0, 21);
  task_b.load_source(1, 22);

  // (3) Task B requests part of Task A's current allocation.
  const int64_t new_grant_a = 256 * 1024 - 8 * 1024;  // Bn
  std::printf("request: reduce A's grant by 8192 B (Bn = %lld B)\n",
              (long long)new_grant_a);

  // (4-7) capture, plan, validate, prepare.
  HandoverResult res =
      perform_handover(task_a, arena, /*to_task=*/1, new_grant_a, /*K=*/50,
                       PlanPolicy::Hybrid, /*stop_after_commit=*/true,
                       &task_b);

  // Print the protocol trace.
  std::printf("protocol outcome: %s (%s)\n", handover_phase_name(res.phase),
              res.note.c_str());
  for (const auto& e : res.log)
    std::printf("  %-17s %s\n", e.event.c_str(), e.detail.c_str());

  failures += check(res.accepted, "2-4. capture, planning, validation succeeded");
  failures += check(res.checker_invocations >= 1,
                    "5. independent checker validated the plan");
  failures += check(!res.transferred_blocks.empty(),
                    "8. concrete storage blocks became transferable");

  // (9) Ownership of the exact blocks changed A -> B.
  bool ownership_moved = arena.blocks_owned(1) > b_before &&
                         arena.blocks_owned(0) < a_before;
  failures += check(ownership_moved, "9. block ownership changed A -> B");
  std::printf("transferred blocks: %zu (bytes: %lld); A owned %lld -> %lld, "
              "B owned %lld -> %lld\n",
              res.transferred_blocks.size(), (long long)res.released_bytes,
              (long long)a_before, (long long)arena.blocks_owned(0),
              (long long)b_before, (long long)arena.blocks_owned(1));

  // (10-11) Start A's post-COMMIT continuation and deliberately stop it
  // before publication. B then runs concurrently and its output allocation is
  // inspected at the worker boundary, proving real overlap and real reuse.
  std::mutex overlap_mu;
  std::condition_variable overlap_cv;
  bool a_blocked = false, release_a = false, a_finished = false;
  bool b_started = false, b_used_transferred = false;
  task_a.set_worker_hook([&](OpId, WorkerEvent event) {
    if (event != WorkerEvent::BeforeComplete) return;
    std::unique_lock<std::mutex> lk(overlap_mu);
    if (a_blocked) return;
    a_blocked = true;
    overlap_cv.notify_all();
    overlap_cv.wait(lk, [&] { return release_a; });
  });
  task_b.set_worker_hook([&](OpId op, WorkerEvent event) {
    if (op != 2 || event != WorkerEvent::BeforeExecute) return;
    const AllocRecord& rec = arena.record(task_b.tensor_allocation(2));
    for (int32_t b : res.transferred_blocks)
      if (rec.first_block <= b && b < rec.first_block + rec.nblocks)
        b_used_transferred = true;
    std::lock_guard<std::mutex> lk(overlap_mu);
    b_started = true;
    overlap_cv.notify_all();
  });
  bool a_ok = false, b_ok = false;
  std::thread a_thread([&] {
    std::string why;
    a_ok = continue_handover(task_a, res, &why);
    std::lock_guard<std::mutex> lk(overlap_mu);
    a_finished = true;
    overlap_cv.notify_all();
  });
  {
    std::unique_lock<std::mutex> lk(overlap_mu);
    const bool reached = overlap_cv.wait_for(
        lk, std::chrono::seconds(5), [&] { return a_blocked; });
    failures += check(reached, "10. Task A has unfinished post-COMMIT work");
  }
  std::thread b_thread([&] { b_ok = task_b.run_to_completion(); });
  bool b_observed = false;
  {
    std::unique_lock<std::mutex> lk(overlap_mu);
    b_observed = overlap_cv.wait_for(
        lk, std::chrono::seconds(5), [&] { return b_started; });
    failures += check(b_observed && !a_finished,
                      "11. Task B used transferred blocks while A was unfinished");
    release_a = true;
  }
  overlap_cv.notify_all();
  b_thread.join();
  a_thread.join();
  failures += check(a_ok && task_a.controlled_bytes() <= task_a.grant_bytes(),
                    "12. Task A completed within the reduced grant");
  failures += check(b_ok && b_used_transferred,
                    "    Task B completed from the transferred physical extent");
  std::printf("new Task A grant: %lld B (peak live %lld B)\n",
              (long long)task_a.grant_bytes(),
              (long long)task_a.stats().peak_live_bytes);

  // (13) Both computations produce independently checked reference-correct outputs.
  {
    auto* got = (const double*)task_a.tensor_ptr(5);
    bool same = true;
    for (size_t i = 0; i < ref_a[5].size(); ++i)
      if (got[i] != ref_a[5][i]) same = false;
    failures += check(same, "13a. Task A output == independent reference (bit-exact)");
  }
  {
    auto* got = (const double*)task_b.tensor_ptr(2);
    bool same = true;
    for (size_t i = 0; i < ref_b[2].size(); ++i)
      if (got[i] != ref_b[2][i]) same = false;
    failures += check(same, "13b. Task B output == independent reference (bit-exact)");
  }

  // (14) Ownership/invariant verification over the whole arena.
  {
    std::string err;
    const bool ok = arena.verify(&err) && task_a.invariants_hold(&err) &&
                    task_b.invariants_hold(&err);
    failures += check(ok, ("14. final ownership/invariant scan: " +
                           (ok ? std::string("clean") : err))
                              .c_str());
  }

  // Rejected request: no illegal transfer, A unaffected.
  std::printf("\n=== rejected request demonstration ===\n");
  TaskRuntime task_c(make_a_graph(), arena, /*task=*/2, /*grant=*/192 * 1024,
                     /*workers=*/2);
  task_c.load_source(0, 31);
  task_c.load_source(1, 32);
  task_c.run_to_completion();
  // K=0: no prep compute allowed -> the prep space is finite and the planner
  // can prove the over-request impossible instead of exhausting its budget.
  HandoverResult bad = perform_handover(task_c, arena, 1,
                                        192 * 1024 - 160 * 1024, 0, PlanPolicy::Hybrid);
  std::printf("over-request outcome: %s (%s)\n",
              handover_phase_name(bad.phase), bad.note.c_str());
  failures += check(!bad.accepted && bad.released_bytes == 0 &&
                        bad.transferred_blocks.empty(),
                    "R1. over-request REJECTED with zero transfer");
  std::string err2;
  failures += check(task_c.invariants_hold(&err2) && arena.verify(&err2),
                    "R2. task legal under original grant after rejection");

  std::printf("\n%s\n", failures == 0
                             ? "ALL 14 PROPERTIES OBSERVED AND VERIFIED"
                             : "FAILURES PRESENT");
  return failures == 0 ? 0 : 1;
}
