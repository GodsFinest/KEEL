// Phase 4B: numerical executor tests. Bit-exact reference agreement (all ops
// deterministic), physical grant enforcement, in-flight protection,
// recomputation, memory-pressure stalls, and worker execution.

#include <doctest/doctest.h>

#include <keel/rt/executor.hpp>

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <initializer_list>
#include <mutex>

using namespace keel;
using namespace keel::rt;

namespace {

// Mixed numerical DAG with real shapes:
//   src0 -> A [32x24]      src1 -> B [24x16]
//   G  = ScratchGemm(A,B) -> C [32x16]   (scratch 24*16*8 bytes)
//   R  = ReduceSum(C)     -> D [32x1]
//   E1 = Add(D broadcast, C) -> E [32x16]  (elementwise broadcast)
//   M  = Mul(E, C)        -> F [32x16]     (required output)
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
  g.add_op({OpKind::Mul, {4, 2}, {5}, 0, 2, true});  // E * C (distinct inputs)
  g.finish();
  return g;
}

std::vector<uint64_t> two_seeds() { return {7, 8}; }

}  // namespace

TEST_CASE("executor: real DAG executes and matches the reference bit-exactly") {
  RuntimeGraph g = make_dag();
  auto ref = reference_execute_f64(g, two_seeds());

  BlockArena arena(1 << 20, 256);  // 1 MiB — ample
  TaskRuntime rt(g, arena, /*task=*/0, /*grant=*/1 << 20, /*workers=*/4);
  rt.load_source(0, 7);
  rt.load_source(1, 8);
  REQUIRE(rt.run_to_completion());

  // Required output F (tensor 5) must match the reference bit-for-bit.
  auto* got = (const double*)rt.tensor_ptr(5);
  const std::vector<double>& want = ref[5];
  REQUIRE(want.size() == 32 * 16);
  for (size_t i = 0; i < want.size(); ++i) CHECK(got[i] == want[i]);

  // Intermediate C (ScratchGemm output) also matches: the scratch kernel has
  // the same summation order as the plain kernel.
  auto* gc = (const double*)rt.tensor_ptr(2);
  for (size_t i = 0; i < ref[2].size(); ++i) CHECK(gc[i] == ref[2][i]);

  std::string err;
  CHECK(rt.invariants_hold(&err));
  CHECK(rt.stats().completions == 6);
}

TEST_CASE("executor: physical grant enforcement — oversubscription stalls") {
  RuntimeGraph g = make_dag();
  // Sources fit in a 12 KB grant (9216 B); the gemm's C + scratch need 7168 B
  // more than the 3072 B left, and nothing is reclaimable mid-flight -> the
  // run must stall under real memory pressure.
  BlockArena arena(64 * 1024, 256);
  TaskRuntime rt(g, arena, 0, 12288, 2);
  rt.load_source(0, 7);
  rt.load_source(1, 8);
  CHECK(rt.controlled_bytes() == 6144 + 3072);
  CHECK_FALSE(rt.run_to_completion());  // stall, no illegal overcommit
  std::string err;
  CHECK(rt.invariants_hold(&err));
  CHECK(rt.controlled_bytes() <= rt.grant_bytes());
}

TEST_CASE("executor: drop + recompute reproduces identical bits") {
  RuntimeGraph g = make_dag();
  auto ref = reference_execute_f64(g, two_seeds());
  BlockArena arena(1 << 20, 256);
  TaskRuntime rt(g, arena, 0, 1 << 20, 2);
  rt.load_source(0, 7);
  rt.load_source(1, 8);
  REQUIRE(rt.run_to_completion());

  // Drop intermediate E (tensor 4), then recompute it via its producer.
  std::string why;
  REQUIRE(rt.drop_tensor(4, &why));
  CHECK_FALSE(rt.tensor_resident(4));
  const int64_t after_drop = rt.controlled_bytes();
  CHECK(rt.recompute_tensor(4, &why));
  CHECK(rt.tensor_resident(4));
  CHECK(rt.tensor_valid(4));
  CHECK(rt.controlled_bytes() >= after_drop);
  auto* got = (const double*)rt.tensor_ptr(4);
  for (size_t i = 0; i < ref[4].size(); ++i) CHECK(got[i] == ref[4][i]);

  // Recompute is refused for non-replayable producers (sources) once their
  // value is actually gone.
  REQUIRE(rt.drop_tensor(0, &why));
  CHECK_FALSE(rt.recompute_tensor(0, &why));

  std::string err;
  CHECK(rt.invariants_hold(&err));
  CHECK(rt.stats().recomputations == 1);
}

TEST_CASE("executor: in-flight protection refuses drops during execution") {
  RuntimeGraph g = make_dag();
  BlockArena arena(1 << 20, 256);
  // Single worker: dispatch A->in flight deterministically before it runs.
  TaskRuntime rt(g, arena, 0, 1 << 20, 1);
  rt.load_source(0, 7);
  rt.load_source(1, 8);
  // Sources were loaded (done); the gemm is the only dispatchable op.
  CHECK(rt.dispatch_ready() == 1);
  CHECK(rt.op_state(2) == OpState::InFlight);
  // The gemm is in flight and reads A and B: their drops must be refused.
  std::string why;
  CHECK_FALSE(rt.drop_tensor(0, &why));
  CHECK_FALSE(rt.drop_tensor(1, &why));
  // C (its output) is reserved but not yet valid: not droppable either
  // (not valid yet, and written by the in-flight op).
  CHECK(rt.op_state(2) == OpState::InFlight);
  rt.wait_op(2);
  CHECK(rt.tensor_valid(2));
}

TEST_CASE("executor: drop-after-last-use reclaims physical storage") {
  RuntimeGraph g = make_dag();
  BlockArena arena(1 << 20, 256);
  TaskRuntime rt(g, arena, 0, 1 << 20, 2);
  rt.load_source(0, 7);
  rt.load_source(1, 8);
  REQUIRE(rt.run_to_completion());
  const int64_t peak = rt.stats().peak_live_bytes;
  CHECK(peak > 0);
  // Drop every intermediate; only the required output stays.
  for (TensorId t = 0; t < 5; ++t) {
    std::string why;
    const bool dropped = rt.drop_tensor(t, &why);
    const std::string dmsg = std::string("drop t") + std::to_string(t) +
                             std::string(" refused: ") + why;
    CHECK_MESSAGE(dropped, dmsg);
  }
  CHECK(rt.controlled_bytes() == rt.graph().tensor_bytes(5));
  std::string err;
  CHECK(rt.invariants_hold(&err));
}

TEST_CASE("executor: completion requires valid required outputs") {
  RuntimeGraph g;
  g.add_tensor({DType::F64, {1}, true});
  g.add_op({OpKind::Source, {}, {0}, 0, 1, false});
  g.finish();

  BlockArena arena(16 * 1024, 256);
  TaskRuntime rt(g, arena, 0, 16 * 1024, 1);
  rt.load_source(0, 7);
  REQUIRE(rt.run_to_completion());
  std::string why;
  REQUIRE(rt.drop_tensor(0, &why));
  CHECK_FALSE(rt.run_to_completion());
  CHECK_FALSE(rt.completion_ready(&why));
  CHECK(why.find("required tensor") != std::string::npos);
}

TEST_CASE("executor: replayable required outputs are restored, sources are immutable") {
  RuntimeGraph g;
  g.add_tensor({DType::F64, {1}});
  g.add_tensor({DType::F64, {1}, true});
  g.add_op({OpKind::Source, {}, {0}, 0, 1, false});
  g.add_op({OpKind::Add, {0}, {1}, 0, 1, true});
  g.finish();

  BlockArena arena(16 * 1024, 256);
  TaskRuntime rt(g, arena, 0, 16 * 1024, 1);
  rt.load_source(0, 7);
  REQUIRE(rt.run_to_completion());
  const double source_before =
      static_cast<const double*>(rt.tensor_ptr(0))[0];
  const double output_before =
      static_cast<const double*>(rt.tensor_ptr(1))[0];
  CHECK_THROWS(rt.load_source(0, 99));
  CHECK(static_cast<const double*>(rt.tensor_ptr(0))[0] == source_before);
  CHECK(static_cast<const double*>(rt.tensor_ptr(1))[0] == output_before);

  std::string why;
  REQUIRE(rt.drop_tensor(1, &why));
  CHECK(rt.run_to_completion());
  CHECK(static_cast<const double*>(rt.tensor_ptr(1))[0] == output_before);
}

TEST_CASE("executor: second source load is rejected during dependent execution") {
  RuntimeGraph g;
  g.add_tensor({DType::F64, {1}});
  g.add_tensor({DType::F64, {1}, true});
  g.add_op({OpKind::Source, {}, {0}, 0, 1, false});
  g.add_op({OpKind::Add, {0}, {1}, 0, 1, true});
  g.finish();

  BlockArena arena(16 * 1024, 256);
  TaskRuntime rt(g, arena, 0, 16 * 1024, 1);
  rt.load_source(0, 7);
  std::mutex mu;
  std::condition_variable cv;
  bool before_complete = false;
  bool release_worker = false;
  rt.set_worker_hook([&](OpId op, WorkerEvent event) {
    if (op != 1 || event != WorkerEvent::BeforeComplete) return;
    std::unique_lock<std::mutex> lk(mu);
    before_complete = true;
    cv.notify_all();
    cv.wait(lk, [&] { return release_worker; });
  });
  REQUIRE(rt.dispatch_ready() == 1);
  {
    std::unique_lock<std::mutex> lk(mu);
    REQUIRE(cv.wait_for(lk, std::chrono::seconds(5),
                        [&] { return before_complete; }));
  }
  CHECK_THROWS(rt.load_source(0, 99));
  {
    std::lock_guard<std::mutex> lk(mu);
    release_worker = true;
  }
  cv.notify_all();
  rt.wait_op(1);
  CHECK(rt.completion_ready());
}

TEST_CASE("executor: completion ordering does not affect results") {
  // Same graph, workers=1 vs workers=4: bit-identical outputs.
  RuntimeGraph g = make_dag();
  auto ref = reference_execute_f64(g, two_seeds());
  for (int workers : {1, 2, 4}) {
    BlockArena arena(1 << 20, 256);
    TaskRuntime rt(g, arena, 0, 1 << 20, workers);
    rt.load_source(0, 7);
    rt.load_source(1, 8);
    REQUIRE(rt.run_to_completion());
    auto* got = (const double*)rt.tensor_ptr(5);
    bool same = true;
    for (size_t i = 0; i < ref[5].size(); ++i)
      if (got[i] != ref[5][i]) same = false;
    CHECK(same);
  }
}

TEST_CASE("executor: independent hand-calculated kernels cover broadcast and matmul") {
  RuntimeGraph g;
  g.add_tensor({DType::F64, {2, 3}});  // 0: A
  g.add_tensor({DType::F64, {3}});     // 1: b, broadcast over rows
  g.add_tensor({DType::F64, {3, 2}});  // 2: B
  g.add_tensor({DType::F64, {2, 3}});  // 3: A+b
  g.add_tensor({DType::F64, {2, 2}});  // 4: A*B
  g.add_tensor({DType::F64, {2, 1}});  // 5: row sum
  g.add_tensor({DType::F64, {2, 3}, true});  // 6: elementwise product
  g.add_op({OpKind::Source, {}, {0}, 0, 1, false});
  g.add_op({OpKind::Source, {}, {1}, 0, 1, false});
  g.add_op({OpKind::Source, {}, {2}, 0, 1, false});
  g.add_op({OpKind::Add, {0, 1}, {3}, 0, 1, true});
  g.add_op({OpKind::MatMul, {0, 2}, {4}, 0, 1, true});
  g.add_op({OpKind::ReduceSum, {3}, {5}, 0, 1, true});
  g.add_op({OpKind::Mul, {3, 0}, {6}, 0, 1, true});
  g.finish();

  BlockArena arena(64 * 1024, 256);
  TaskRuntime rt(g, arena, 0, 64 * 1024, 2);
  rt.load_source(0, 1);
  rt.load_source(1, 2);
  rt.load_source(2, 3);
  auto overwrite = [&](TensorId t, std::initializer_list<double> values) {
    double* p = static_cast<double*>(const_cast<void*>(rt.tensor_ptr(t)));
    size_t i = 0;
    for (double value : values) p[i++] = value;
  };
  overwrite(0, {1, 2, 3, 4, 5, 6});
  overwrite(1, {10, 20, 30});
  overwrite(2, {1, 0, 2, 1, 3, 1});
  REQUIRE(rt.run_to_completion());

  const double* add = static_cast<const double*>(rt.tensor_ptr(3));
  const double* mm = static_cast<const double*>(rt.tensor_ptr(4));
  const double* red = static_cast<const double*>(rt.tensor_ptr(5));
  const double* mul = static_cast<const double*>(rt.tensor_ptr(6));
  for (size_t i = 0; i < 6; ++i)
    CHECK(add[i] == std::array<double, 6>{11, 22, 33, 14, 25, 36}[i]);
  for (size_t i = 0; i < 4; ++i)
    CHECK(mm[i] == std::array<double, 4>{14, 5, 32, 11}[i]);
  CHECK(red[0] == 66);
  CHECK(red[1] == 75);
  for (size_t i = 0; i < 6; ++i)
    CHECK(mul[i] == std::array<double, 6>{11, 44, 99, 56, 125, 216}[i]);
}

TEST_CASE("executor: multidimensional broadcast uses trailing dimensions") {
  RuntimeGraph g;
  g.add_tensor({DType::F64, {2, 1}});
  g.add_tensor({DType::F64, {2, 2}});
  g.add_tensor({DType::F64, {2, 2}, true});
  g.add_op({OpKind::Source, {}, {0}, 0, 1, false});
  g.add_op({OpKind::Source, {}, {1}, 0, 1, false});
  g.add_op({OpKind::Add, {0, 1}, {2}, 0, 1, true});
  g.finish();

  BlockArena arena(16 * 1024, 256);
  TaskRuntime rt(g, arena, 0, 16 * 1024, 1);
  rt.load_source(0, 1);
  rt.load_source(1, 2);
  auto write = [&](TensorId t, std::initializer_list<double> values) {
    double* p = static_cast<double*>(const_cast<void*>(rt.tensor_ptr(t)));
    size_t i = 0;
    for (double value : values) p[i++] = value;
  };
  write(0, {1, 2});
  write(1, {10, 20, 30, 40});
  REQUIRE(rt.run_to_completion());
  const double* out = static_cast<const double*>(rt.tensor_ptr(2));
  CHECK(out[0] == 11);
  CHECK(out[1] == 21);
  CHECK(out[2] == 32);
  CHECK(out[3] == 42);
}

TEST_CASE("executor: in-flight writer lifetime is protected") {
  RuntimeGraph g = make_dag();
  BlockArena arena(1 << 20, 256);
  TaskRuntime rt(g, arena, 0, 1 << 20, 1);
  rt.load_source(0, 7);
  rt.load_source(1, 8);

  std::mutex mu;
  std::condition_variable cv;
  bool before_complete = false;
  bool release_worker = false;
  rt.set_worker_hook([&](OpId op, WorkerEvent event) {
    if (op != 2 || event != WorkerEvent::BeforeComplete) return;
    std::unique_lock<std::mutex> lk(mu);
    before_complete = true;
    cv.notify_all();
    cv.wait(lk, [&] { return release_worker; });
  });
  REQUIRE(rt.dispatch_ready() == 1);
  {
    std::unique_lock<std::mutex> lk(mu);
    REQUIRE(cv.wait_for(lk, std::chrono::seconds(5),
                        [&] { return before_complete; }));
  }
  CHECK_FALSE(rt.drop_tensor(2));
  CHECK(rt.release_tensor(2, 1).empty());
  CHECK(rt.tensor_resident(2));
  CHECK_THROWS(rt.tensor_ptr(2));
  {
    std::lock_guard<std::mutex> lk(mu);
    release_worker = true;
  }
  cv.notify_all();
  rt.wait_op(2);
  CHECK(rt.tensor_valid(2));
}
