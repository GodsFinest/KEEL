// Generator tests: determinism, frontier legality, and the four DECLARED
// adversarial traps with oracle-pinned classifications (docs/EXPERIMENTS.md).
// Trap pins double as design verification: each trap's sizes were hand-checked
// against the SPEC accounting, and these tests lock the analysis in.

#include <doctest/doctest.h>

#include <keel/checker.hpp>
#include <keel/gen.hpp>
#include <keel/oracle.hpp>

using namespace keel;

namespace {

State run_prefix(const Graph& g, int64_t bo, const std::vector<OpId>& ops) {
  Request exec{bo, bo, (int64_t)1 << 40};
  State s;
  for (OpId v : ops) {
    REQUIRE(can_run(g, s, exec, v).ok);
    s = apply_run(g, s, exec, v);
  }
  return capture_frontier(s);
}

}  // namespace

TEST_CASE("generators are deterministic and structurally valid") {
  for (int fi = 0; fi < keel::kFamilyCount; ++fi) {
    Family f = (Family)fi;
    for (int di = 0; di < 3; ++di) {
      CostDist d = (CostDist)di;
      Generated a = generate(f, d, 7);
      Generated b = generate(f, d, 7);
      CHECK(a.bo == b.bo);
      REQUIRE(a.graph.ops.size() == b.graph.ops.size());
      bool same = true;
      for (size_t i = 0; i < a.graph.tensors.size(); ++i)
        if (a.graph.tensors[i].size != b.graph.tensors[i].size) same = false;
      CHECK(same);
      CHECK(a.graph.is_dag());
      CHECK(a.bo > 0);
    }
  }
}

TEST_CASE("sampled frontiers are legal states under Bo") {
  for (int fi = 0; fi < 7; ++fi) {  // organic families only
    Family f = (Family)fi;
    for (uint64_t seed = 1; seed <= 5; ++seed) {
      Generated gen = generate(f, CostDist::Uniform, seed);
      for (uint64_t fseed = 0; fseed < 4; ++fseed) {
        Rng rng(1000 + fseed);
        State s = sample_frontier(gen.graph, gen.bo, rng, 1, 5, fseed >= 2, 2);
        Memory m = memory_of(gen.graph, s, Request{gen.bo, gen.bo, 0});
        CHECK(m.usage <= gen.bo);          // invariant (released = 0 here)
        CHECK(m.ok);
        for (int t = 0; t < (int)gen.graph.tensors.size(); ++t) {
          if (s.resident & (TensorMask(1) << t)) {
            OpId p = gen.graph.tensors[t].producer;
            CHECK((s.done & (OpMask(1) << p)) != 0);
          }
        }
      }
    }
  }
}

TEST_CASE("trap_recompute: greedy overshoot kills s; oracle re-derives") {
  Generated gen = generate(Family::TrapRecompute, CostDist::Uniform, 1);
  const Graph& g = gen.graph;  // Bo = 160; frontier s(25), x(50), y(20)
  State s0 = run_prefix(g, gen.bo, {0, 1, 2});
  CHECK(gen.bo == 160);

  SUBCASE("release {x,y} + re-derive chain from s: feasible at K=0") {
    Request r{160, 105, 0};  // target 55; greedy's {x,s} is fatal
    OracleResult res = solve_transition(g, s0, r, OracleConfig{});
    REQUIRE(res.status == OracleStatus::Feasible);
    CheckResult c = check_witness(g, s0, r, res.witness);
    CHECK(c.ok);
    CHECK(c.reached_goal);
    // s must never be released: nothing re-derives it (v0 non-replayable).
    for (const Act& a : res.witness.acts)
      if (a.kind == ActKind::Release) CHECK(a.target != 0);
  }
  SUBCASE("target 105 is feasible via RELEASE AMPLIFICATION") {
    // Discovery: the oracle surrenders x's block, re-derives x from s into a
    // fresh block, and can surrender the NEW block too — released bytes are
    // not bounded by frontier residency. At target 105 (Bn=55) some
    // amplifying plan must exist; pin released >= 105.
    Request r{160, 55, 1000};
    OracleResult res = solve_transition(g, s0, r, OracleConfig{});
    REQUIRE(res.status == OracleStatus::Feasible);
    CheckResult c = check_witness(g, s0, r, res.witness);
    CHECK(c.ok);
    CHECK(c.reached_goal);
    CHECK(c.released_bytes >= 105);
  }
  SUBCASE("target 141 violates the region invariant -> provably infeasible") {
    // At the goal z(20) must be resident, so released <= Bo - 20 = 140
    // always; 141 is impossible for ANY plan at ANY K.
    Request r{160, 19, 1000};
    OracleResult res = solve_transition(g, s0, r, OracleConfig{});
    CHECK(res.status == OracleStatus::Infeasible);
  }
}

TEST_CASE("trap_nonreplayable: keep the source, release the derivations") {
  Generated gen = generate(Family::TrapNonreplayable, CostDist::Uniform, 1);
  const Graph& g = gen.graph;  // Bo = 240
  State s0 = run_prefix(g, gen.bo, {0, 1, 2});  // s, a, b resident
  CHECK(gen.bo == 240);

  SUBCASE("release {a,b}, re-run v1/v2 post-commit: feasible at K=0") {
    Request r{240, 170, 0};  // target 70 = a + b
    OracleResult res = solve_transition(g, s0, r, OracleConfig{});
    REQUIRE(res.status == OracleStatus::Feasible);
    CheckResult c = check_witness(g, s0, r, res.witness);
    CHECK(c.ok);
    CHECK(c.reached_goal);
    // The witness must NOT release s: without it nothing downstream reproduces.
    for (const Act& a : res.witness.acts)
      if (a.kind == ActKind::Release) CHECK(a.target != 0);
  }
  SUBCASE("target 80 forces sacrificing s (K=0 blocks v3-in-prep)") {
    Request r{240, 160, 0};
    OracleResult res = solve_transition(g, s0, r, OracleConfig{});
    CHECK(res.status == OracleStatus::Infeasible);
  }
}

TEST_CASE("trap_memory: greedy's {big,c1} overshoot is fatal") {
  Generated gen = generate(Family::TrapMemory, CostDist::Uniform, 1);
  const Graph& g = gen.graph;  // Bo = 320
  State s0 = run_prefix(g, gen.bo, {0, 1, 3, 4});  // big, c1, d1, d2 resident
  CHECK(gen.bo == 320);

  Request r{320, 170, 0};  // target 150: viable sets keep or re-derive c1
  OracleResult res = solve_transition(g, s0, r, OracleConfig{});
  REQUIRE(res.status == OracleStatus::Feasible);
  CheckResult c = check_witness(g, s0, r, res.witness);
  CHECK(c.ok);
  CHECK(c.reached_goal);

  SUBCASE("target 200 with K=0 is provably infeasible") {
    Request r2{320, 120, 0};  // v2-in-prep blocked; all >=200 sets fatal
    OracleResult res2 = solve_transition(g, s0, r2, OracleConfig{});
    CHECK(res2.status == OracleStatus::Infeasible);
  }
}

TEST_CASE("prep_unlock: prep compute is the unlock") {
  Generated gen = generate(Family::PrepUnlock, CostDist::Uniform, 1);
  const Graph& g = gen.graph;  // Bo = 200
  State s0 = run_prefix(g, gen.bo, {0});  // x resident
  CHECK(gen.bo == 200);

  SUBCASE("K=0: no drop-only plan can work") {
    Request r{200, 120, 0};  // target 80 = x
    OracleResult res = solve_transition(g, s0, r, OracleConfig{});
    CHECK(res.status == OracleStatus::Infeasible);
  }
  SUBCASE("K=3: one prep RUN converts x to y, unlocking the handover") {
    Request r{200, 120, 3};
    OracleResult res = solve_transition(g, s0, r, OracleConfig{});
    REQUIRE(res.status == OracleStatus::Feasible);
    CHECK(res.total_compute >= 3);  // at least the v1 run
    CheckResult c = check_witness(g, s0, r, res.witness);
    CHECK(c.ok);
    CHECK(c.reached_goal);
  }
  SUBCASE("target 170 exceeds max releasable 160 -> infeasible at any K") {
    Request r{200, 30, 1000};
    OracleResult res = solve_transition(g, s0, r, OracleConfig{});
    CHECK(res.status == OracleStatus::Infeasible);
  }
}

TEST_CASE("trap_concurrent_launch: serialization is the only legal plan") {
  // Audit finding 6 pin: under v1.0 the concurrent trace (LAUNCH both, WAIT
  // both, DROP the disposable output) validated despite 6 > 4 in flight;
  // under v1.1 the second concurrent launch is illegal and only the
  // serialized plan survives. Grid-aligned: Bo=5, Bn=4 (the 0.8 bucket),
  // target 1 = the init block.
  using namespace keel;
  Generated gen = generate(Family::TrapConcurrentLaunch, CostDist::Uniform, 1);
  const Graph& g = gen.graph;  // Bo = 5; init(1), x(3, disposable), y(3, req)
  CHECK(gen.bo == 5);
  State s0 = run_prefix(g, gen.bo, {0});  // init resident

  SUBCASE("concurrent launches are illegal at the second LAUNCH") {
    State s = s0;
    REQUIRE(can_release(g, s, 0).ok);
    s = apply_release(g, s, 0);
    REQUIRE(can_commit(g, s, Request{5, 4, 100}).ok);
    s = apply_commit(g, s, Request{5, 4, 100});
    CHECK(can_launch(g, s, Request{5, 4, 100}, 1).ok);
    s = apply_launch(g, s, Request{5, 4, 100}, 1);
    CHECK_FALSE(can_launch(g, s, Request{5, 4, 100}, 2).ok);
  }
  SUBCASE("oracle finds the serialized plan; concurrent trace rejected") {
    Request r{5, 4, 100};
    OracleConfig cfg;
    cfg.allow_async = true;
    OracleResult res = solve_transition(g, s0, r, cfg);
    REQUIRE(res.status == OracleStatus::Feasible);
    CheckResult c = check_witness(g, s0, r, res.witness);
    CHECK(c.ok);
    CHECK(c.reached_goal);
    // The witness must never hold both launches in flight at once.
    int inflight_count = 0;
    for (const Act& a : res.witness.acts) {
      if (a.kind == ActKind::Launch) ++inflight_count;
      if (a.kind == ActKind::Wait) --inflight_count;
      CHECK(inflight_count <= 1);
    }
  }
}
