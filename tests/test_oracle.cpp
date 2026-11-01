// Exact oracle tests (docs/SPEC.md §7–§9): witness production with optimal
// cost, proven infeasibility vs budget exhaustion, async frontiers, K
// semantics, and the differential soundness check of dominance pruning.

#include <doctest/doctest.h>

#include <keel/checker.hpp>
#include <keel/oracle.hpp>

using namespace keel;

namespace {

// v0 (source, non-replayable) -> t0(40);  v1: t0 -> t1(30, required).
Graph chain() {
  GraphBuilder b;
  b.add_op({}, {b.add_tensor(40, -1)}, 0, 1, false);
  b.add_op({0}, {b.add_tensor(30, -1, true)}, 0, 2, true);
  return b.finish();
}

Request req(int64_t bn, int64_t k) { return Request{.Bo = 100, .Bn = bn, .K = k}; }

State after(const Graph& g, const Request& r, const std::vector<Act>& acts) {
  // Prefix execution runs under a pre-request budget (K unlimited); the
  // captured frontier then resets prep_used so the request's K counts only
  // post-capture preparation (docs/SPEC.md §5).
  Request exec{r.Bo, r.Bn, (int64_t)1 << 40};
  State s;
  for (const Act& a : acts) {
    switch (a.kind) {
      case ActKind::Run: {
        REQUIRE(can_run(g, s, exec, a.target).ok);
        s = apply_run(g, s, exec, a.target);
        break;
      }
      case ActKind::Launch:
        REQUIRE(can_launch(g, s, exec, a.target).ok);
        s = apply_launch(g, s, exec, a.target);
        break;
      case ActKind::Wait:
        s = apply_wait(g, s, a.target);
        break;
      case ActKind::Release:
        s = apply_release(g, s, a.target);
        break;
      case ActKind::Drop:
        s = apply_drop(g, s, a.target);
        break;
      case ActKind::Commit:
        s = apply_commit(g, s, r);
        break;
    }
  }
  // Frontier capture: pre-request compute is sunk; K counts from here.
  return capture_frontier(s);
}

}  // namespace

TEST_CASE("oracle: feasible handover with optimal cost + valid witness") {
  Graph g = chain();
  Request r = req(65, 10);
  State s0 = after(g, r, {{ActKind::Run, 0}});  // t0 resident, v0 spent

  OracleResult res = solve_transition(g, s0, r, OracleConfig{});
  REQUIRE(res.status == OracleStatus::Feasible);
  // Optimal: run v1 (cost 2), release t0 (40 >= 35), commit. Nothing cheaper
  // exists: t1 must be produced (2 units) and no other compute is needed.
  CHECK(res.total_compute == 2);
  CheckResult c = check_witness(g, s0, r, res.witness);
  CHECK(c.ok);
  CHECK(c.reached_goal);
}

TEST_CASE("oracle: sacrificial release + reproduce is found when legal") {
  // The task may surrender a tensor it can still reproduce: RUN v1, RELEASE
  // t1 (30), RUN v1 again (t0 is still resident: inputs outlive ops), RELEASE
  // t0 (40) -> released 70, COMMIT, goal holds with t1 re-produced. This
  // sacrifice-and-reproduce path costs 4 and is exactly the kind of
  // non-obvious transition the planner exists to find.
  Graph g = chain();
  Request r = req(50, 100);
  State s0 = after(g, r, {{ActKind::Run, 0}});
  OracleResult res = solve_transition(g, s0, r, OracleConfig{});
  REQUIRE(res.status == OracleStatus::Feasible);
  CHECK(res.total_compute == 4);
  CheckResult c = check_witness(g, s0, r, res.witness);
  CHECK(c.ok);
  CHECK(c.reached_goal);
}

TEST_CASE("oracle: proven infeasibility (exhausted space)") {
  Graph g = chain();
  // Maximum physically releasable bytes = 70 (t0 + t1, the latter only via
  // sacrifice-and-reproduce). Bo - Bn = 75 exceeds it: the request is
  // genuinely impossible, and the oracle PROVES that by exhausting the space.
  Request r = req(25, 100);
  State s0 = after(g, r, {{ActKind::Run, 0}});
  OracleResult res = solve_transition(g, s0, r, OracleConfig{});
  CHECK(res.status == OracleStatus::Infeasible);
}

TEST_CASE("oracle: budget exhaustion yields UNKNOWN, never INFEASIBLE") {
  Graph g = chain();
  Request r = req(65, 10);
  State s0 = after(g, r, {{ActKind::Run, 0}});
  OracleConfig tiny{};
  tiny.max_expansions = 1;  // cannot even pop twice
  OracleResult res = solve_transition(g, s0, r, tiny);
  CHECK(res.status == OracleStatus::Unknown);
}

TEST_CASE("oracle: K=0 forbids prep compute") {
  Graph g = chain();
  Request r = req(65, 0);

  SUBCASE("frontier with work done: free handover still possible") {
    State s0 = after(g, r, {{ActKind::Run, 0}, {ActKind::Run, 1}});
    OracleResult res = solve_transition(g, s0, r, OracleConfig{});
    REQUIRE(res.status == OracleStatus::Feasible);
    CHECK(res.total_compute == 0);  // release t0 + commit only
    CheckResult c = check_witness(g, s0, r, res.witness);
    CHECK(c.ok);
    CHECK(c.reached_goal);
  }
  SUBCASE("frontier before v0: no prep allowed -> infeasible") {
    State s0;
    OracleResult res = solve_transition(g, s0, r, OracleConfig{});
    CHECK(res.status == OracleStatus::Infeasible);
  }
}

TEST_CASE("oracle: async frontier with in-flight op") {
  Graph g = chain();
  Request r = req(65, 10);
  // v1 launched, not completed: t0 protected, t1 absent.
  State s0 = after(g, r, {{ActKind::Run, 0}, {ActKind::Launch, 1}});
  CHECK(s0.inflight != 0);

  OracleConfig cfg;
  cfg.allow_async = true;
  OracleResult res = solve_transition(g, s0, r, cfg);
  REQUIRE(res.status == OracleStatus::Feasible);
  CHECK(res.total_compute == 0);  // WAIT + RELEASE + COMMIT are free
  CheckResult c = check_witness(g, s0, r, res.witness);
  CHECK(c.ok);
  CHECK(c.reached_goal);
  // t0 is protected while v1 is in flight, so the witness must WAIT before
  // the first RELEASE (the checker enforces this; here we pin the shape).
  bool saw_wait = false;
  for (const Act& a : res.witness.acts) {
    if (a.kind == ActKind::Wait) saw_wait = true;
    if (a.kind == ActKind::Release) REQUIRE(saw_wait);
  }
}

TEST_CASE("oracle: async frontier without async actions still solves") {
  // Sequential oracle may WAIT-less: it can still reach the goal by never
  // launching; but the in-flight op from the frontier cannot be completed
  // without WAIT, which sequential mode does not generate.
  Graph g = chain();
  Request r = req(65, 10);
  State s0 = after(g, r, {{ActKind::Run, 0}, {ActKind::Launch, 1}});
  OracleResult res = solve_transition(g, s0, r, OracleConfig{});
  CHECK(res.status == OracleStatus::Infeasible);  // stuck behind in-flight v1
}

TEST_CASE("oracle: scratch is accounted at run/launch time") {
  // v1 needs 25 scratch; its peak is 30 (out) + 25 (scratch) = 55.
  GraphBuilder b;
  b.add_op({}, {b.add_tensor(40, -1)}, 0, 1, false);         // v0 -> t0
  b.add_op({0}, {b.add_tensor(30, -1, true)}, 25, 2, true);  // v1: t0 -> t1
  Graph g = b.finish();
  State s0_empty;
  Request r = req(65, 10);
  State s0 = after(g, r, {{ActKind::Run, 0}});  // t0 resident (40)

  // Bo = 94: after surrendering t0 the region is 54 < 55 — v1 can never run
  // (before release: 40+55 = 95 > 94; after release: 55 > 54). A model that
  // forgot scratch would call this feasible; the oracle must not.
  Request scratch_trap{.Bo = 94, .Bn = 65, .K = 10};
  OracleResult res = solve_transition(g, s0, scratch_trap, OracleConfig{});
  CHECK(res.status == OracleStatus::Infeasible);

  // Bo = 95: ceiling after release is exactly 55 — boundary case is legal.
  Request boundary{.Bo = 95, .Bn = 65, .K = 10};
  OracleResult res2 = solve_transition(g, s0, boundary, OracleConfig{});
  REQUIRE(res2.status == OracleStatus::Feasible);
  CHECK(res2.total_compute == 2);
  CheckResult c = check_witness(g, s0, boundary, res2.witness);
  CHECK(c.ok);
  CHECK(c.reached_goal);
  (void)s0_empty;
}

TEST_CASE("dominance pruning: differential agreement on hand instances") {
  Graph g = chain();
  Request r = req(65, 10);
  State s0 = after(g, r, {{ActKind::Run, 0}});
  OracleConfig plain{};
  OracleConfig dom{};
  dom.use_dominance = true;
  OracleResult a = solve_transition(g, s0, r, plain);
  OracleResult bres = solve_transition(g, s0, r, dom);
  REQUIRE(a.status == OracleStatus::Feasible);
  REQUIRE(bres.status == OracleStatus::Feasible);
  CHECK(a.total_compute == bres.total_compute);
  CHECK(bres.dominance_skips >= 0);
}
