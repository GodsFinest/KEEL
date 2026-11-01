// Phase 6: adversarial and hardening tests (mandate Part VI). Every case is
// deterministic — no reliance on scheduler luck.

#include <doctest/doctest.h>

#include <keel/policy.hpp>
#include <keel/rt/executor.hpp>
#include <keel/rt/handover.hpp>

#include <cstring>

using namespace keel;
using namespace keel::rt;

namespace {

RuntimeGraph make_dag() {
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

}  // namespace

TEST_CASE("adversary: opposite completion orders give identical results") {
  RuntimeGraph g = make_dag();
  auto ref = reference_execute_f64(g, {7, 8});
  for (int order = 0; order < 2; ++order) {
    BlockArena arena(1 << 20, 256);
    TaskRuntime rt(g, arena, 0, 1 << 20, 2);
    if (order == 0) {
      rt.load_source(0, 7);
      rt.load_source(1, 8);
    } else {
      rt.load_source(1, 8);
      rt.load_source(0, 7);
    }
    rt.dispatch_ready();
    rt.wait_op(1);
    rt.wait_op(0);
    CHECK(rt.run_to_completion());
    auto* got = (const double*)rt.tensor_ptr(5);
    bool same = true;
    for (size_t i = 0; i < ref[5].size(); ++i)
      if (got[i] != ref[5][i]) same = false;
    CHECK(same);
  }
}

TEST_CASE("adversary: handover request while an operation is in flight") {
  RuntimeGraph g = make_dag();
  BlockArena arena(1 << 20, 256);
  TaskRuntime task_a(g, arena, 0, 256 * 1024, 2);
  TaskRuntime task_b(make_dag(), arena, 1, 64 * 1024, 2);
  task_a.load_source(0, 7);
  task_a.load_source(1, 8);
  task_a.dispatch_ready();  // gemm in flight
  CHECK(task_a.op_state(2) == OpState::InFlight);

  // The capture sees the in-flight op; the witness WAITs it before any action
  // that touches its outputs.
  HandoverResult res = perform_handover(task_a, arena, 1,
                                        256 * 1024 - 8 * 1024, 100,
                                        PlanPolicy::Hybrid, false, &task_b);
  const std::string imsg =
      std::string("handover with in-flight op failed: ") + res.note;
  CHECK_MESSAGE(res.accepted, imsg);
  CHECK(task_a.tensor_valid(2));  // the in-flight gemm completed
  CHECK(task_a.run_to_completion());
  std::string err;
  CHECK(task_a.invariants_hold(&err));
  CHECK(arena.verify(&err));
  (void)task_b;
}

TEST_CASE("adversary: post-commit grant violation is reported, not hidden") {
  // Request Bn BELOW the required output's footprint. If a plan is accepted
  // anyway, the grant must still hold the required output; otherwise the
  // outcome must be Failed/Rejected/Unknown with any moved bytes reported.
  RuntimeGraph g = make_dag();  // required output F = 4096 B
  BlockArena arena(1 << 20, 256);
  TaskRuntime task_a(g, arena, 0, 256 * 1024, 2);
  task_a.load_source(0, 7);
  task_a.load_source(1, 8);
  task_a.run_to_completion();

  const int64_t bn = 2048;  // below F's 4096 B footprint
  HandoverResult res = perform_handover(task_a, arena, 1, bn, 100, PlanPolicy::Hybrid);
  if (res.accepted) {
    CHECK(task_a.grant_bytes() >= 4096);
    CHECK(task_a.controlled_bytes() <= task_a.grant_bytes());
  } else {
    const bool bad_phase = res.phase == HandoverPhase::Failed ||
                           res.phase == HandoverPhase::Rejected ||
                           res.phase == HandoverPhase::Unknown;
    CHECK(bad_phase);
    CHECK(res.released_bytes >= 0);  // truthfully reported (may be 0)
    std::string err;
    CHECK(task_a.invariants_hold(&err));
    CHECK(arena.verify(&err));
  }
}

TEST_CASE("adversary: successive grant reductions") {
  RuntimeGraph g = make_dag();
  BlockArena arena(1 << 20, 256);
  TaskRuntime task_a(g, arena, 0, 256 * 1024, 2);
  task_a.load_source(0, 7);
  task_a.load_source(1, 8);
  task_a.run_to_completion();

  HandoverResult r1 = perform_handover(task_a, arena, 1,
                                       256 * 1024 - 8 * 1024, 100,
                                       PlanPolicy::Hybrid);
  CHECK(r1.accepted);
  HandoverResult r2 = perform_handover(task_a, arena, 1,
                                       256 * 1024 - 12 * 1024, 100,
                                       PlanPolicy::Hybrid);
  if (r2.accepted) {
    CHECK(task_a.grant_bytes() == 256 * 1024 - 12 * 1024);
  } else {
    CHECK(task_a.grant_bytes() == 256 * 1024 - 8 * 1024);
  }
  std::string err;
  CHECK(task_a.invariants_hold(&err));
  CHECK(arena.verify(&err));
}

TEST_CASE("adversary: planner budget exhaustion yields UNKNOWN, never INFEASIBLE") {
  RuntimeGraph g = make_dag();
  Graph abs = g.to_abstract();
  State s;
  for (int v = 0; v < 6; ++v) s.done |= (OpMask(1) << v);
  for (int t = 0; t < 6; ++t) s.resident |= (TensorMask(1) << t);
  Request r{256 * 1024, 1024, 0};  // absurd target
  PolicyConfig cfg;
  cfg.keel_expansion_budget = 8;  // force exhaustion
  PolicyOutcome o = keel_plan(abs, s, r, cfg);
  CHECK(o.status == PolicyStatus::Unknown);
  OracleResult oc = solve_transition(abs, s, r, OracleConfig{});
  CHECK(oc.status == OracleStatus::Infeasible);  // the oracle PROVES it
}

TEST_CASE("adversary: arena rejects malformed requests") {
  SUBCASE("capacity not a block multiple") {
    CHECK_THROWS(BlockArena(1000, 256));
  }
  SUBCASE("zero/negative block size") {
    CHECK_THROWS(BlockArena(1024, 0));
    CHECK_THROWS(BlockArena(1024, -256));
  }
  SUBCASE("allocation size overflow") {
    BlockArena a(4096, 256);
    a.claim(0, 16);
    CHECK_THROWS(a.alloc(0, INT64_MAX));
    CHECK_THROWS(a.alloc(0, -1));
    CHECK_THROWS(a.alloc(0, 0));
  }
  SUBCASE("wrong-owner transfer extent") {
    BlockArena a(2048, 256);
    a.claim(0, 4);
    a.claim(1, 4);
    AllocHandle h = a.alloc(1, 512);
    const AllocRecord& rec = a.record(h);
    // Task 0 tries to transfer task 1's LIVE allocation: refused.
    CHECK_THROWS(a.transfer_extent(0, 1, rec.first_block, rec.nblocks));
    // Even after free, the extent belongs to task 1's free pool.
    a.free(h);
    CHECK_THROWS(a.transfer_extent(0, 1, rec.first_block, rec.nblocks));
    // The owner can transfer it.
    CHECK_NOTHROW(a.transfer_extent(1, 0, rec.first_block, rec.nblocks));
  }
  SUBCASE("negative task id") {
    BlockArena a(1024, 256);
    CHECK_THROWS(a.claim(-1, 4));
  }
}

TEST_CASE("adversary: shutdown with outstanding worker operations") {
  RuntimeGraph g = make_dag();
  BlockArena arena(1 << 20, 256);
  {
    TaskRuntime rt(g, arena, 0, 1 << 20, 2);
    rt.load_source(0, 7);
    rt.load_source(1, 8);
    rt.dispatch_ready();  // gemm in flight at scope exit
    // Destructor runs here: workers must shut down cleanly (no hang, no
    // use-after-free). ASan validates the lifetime.
  }
  MESSAGE("clean shutdown with in-flight work");
}

TEST_CASE("adversary: repeated recomputation of shared ancestors") {
  RuntimeGraph g = make_dag();
  auto ref = reference_execute_f64(g, {7, 8});
  BlockArena arena(1 << 20, 256);
  TaskRuntime rt(g, arena, 0, 1 << 20, 2);
  rt.load_source(0, 7);
  rt.load_source(1, 8);
  REQUIRE(rt.run_to_completion());
  // C (t2) feeds D, E, and F: drop and recompute it repeatedly.
  for (int round = 0; round < 3; ++round) {
    std::string why;
    REQUIRE(rt.drop_tensor(2, &why));
    CHECK(rt.recompute_tensor(2, &why));
    auto* got = (const double*)rt.tensor_ptr(2);
    bool same = true;
    for (size_t i = 0; i < ref[2].size(); ++i)
      if (got[i] != ref[2][i]) same = false;
    CHECK(same);
  }
  CHECK(rt.stats().recomputations == 3);
}

TEST_CASE("adversary: scratch-exhaustion stalls the graph") {
  RuntimeGraph g = make_dag();
  BlockArena arena(1 << 20, 256);
  TaskRuntime rt(g, arena, 0, 12 * 1024, 1);  // sources fit; gemm's C+scratch don't
  rt.load_source(0, 7);
  rt.load_source(1, 8);
  // The gemm cannot reserve C + scratch within the remainder.
  CHECK_FALSE(rt.run_to_completion());
  std::string err;
  CHECK(rt.invariants_hold(&err));
  CHECK(rt.stats().alloc_failures > 0);
}

TEST_CASE("adversary: state version advances with execution (stale-plan guard)") {
  RuntimeGraph g = make_dag();
  BlockArena arena(1 << 20, 256);
  TaskRuntime rt(g, arena, 0, 1 << 20, 2);
  const uint64_t v0 = rt.state_version();
  rt.load_source(0, 7);
  const uint64_t v1 = rt.state_version();
  CHECK(v1 > v0);
  rt.load_source(1, 8);
  rt.run_to_completion();
  CHECK(rt.state_version() > v1);
}
