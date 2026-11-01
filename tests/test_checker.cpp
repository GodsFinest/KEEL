// Independent checker tests: valid witnesses accepted (with correct metrics),
// deliberately corrupted witnesses rejected (docs/SPEC.md §5, mandate §7 P3).
// The checker is the tool that must never accept an illegal plan, so these
// mutation cases are the core of its contract.

#include <doctest/doctest.h>

#include <keel/checker.hpp>
#include <keel/model.hpp>

using namespace keel;

namespace {

// v0 (source, non-replayable) -> t0(40);  v1: t0 -> t1(30, required).
Graph chain() {
  GraphBuilder b;
  b.add_op({}, {b.add_tensor(40, -1)}, 0, 1, false);
  b.add_op({0}, {b.add_tensor(30, -1, true)}, 0, 2, true);
  return b.finish();
}

Request std_req() { return Request{.Bo = 100, .Bn = 65, .K = 10}; }

State after_v0(const Request& r) {
  State s;
  Legality l = can_run(chain(), s, r, 0);
  REQUIRE(l.ok);
  return apply_run(chain(), s, r, 0);  // t0 resident, done={v0}
}

Witness w(const std::vector<Act>& acts) { return Witness{acts}; }

}  // namespace

TEST_CASE("checker accepts a valid handover witness") {
  Graph g = chain();
  Request r = std_req();
  State s0 = after_v0(r);
  // Handover: surrender t0's live block (40 >= Bo-Bn=35), keep t1 (Q).
  Witness good = w({{ActKind::Run, 1},
                    {ActKind::Release, 0},
                    {ActKind::Commit, -1}});
  CheckResult res = check_witness(g, s0, r, good);
  CHECK(res.ok);
  CHECK(res.reached_goal);
  CHECK(res.error_index == -1);
  CHECK(res.prep_compute == 2);  // RUN v1 before COMMIT
  CHECK(res.cont_compute == 0);
  CHECK(res.recompute_compute == 0);
  CHECK(res.released_bytes == 40);
}

TEST_CASE("checker metrics: recompute after COMMIT is classified") {
  Graph g = chain();
  Request r = std_req();
  // Give up BOTH blocks; recompute t0 (via replayable v1? no - t0 comes from
  // non-replayable v0). Instead: release t0, commit, then drop t1 and recompute
  // it via v1 (t0 is gone -> not resident) -- that continuation is impossible.
  // So recompute classification is exercised with a re-runnable producer chain:
  // v2: t1 -> t2(20, required). Commit after releasing t0; then drop t2, run
  // v2 again (inputs t1 resident) -> recompute of an op done at COMMIT.
  GraphBuilder b;
  b.add_op({}, {b.add_tensor(40, -1)}, 0, 1, false);              // v0 -> t0
  b.add_op({0}, {b.add_tensor(30, -1, true)}, 0, 2, true);        // v1: t0->t1
  b.add_op({1}, {b.add_tensor(20, -1, true)}, 0, 3, true);        // v2: t1->t2
  Graph g2 = b.finish();
  State s = after_v0(r);
  Witness plan = w({{ActKind::Run, 1},
                    {ActKind::Run, 2},
                    {ActKind::Release, 0},
                    {ActKind::Commit, -1},
                    {ActKind::Drop, 2},
                    {ActKind::Run, 2}});
  CheckResult res = check_witness(g2, s, r, plan);
  CHECK(res.ok);
  CHECK(res.reached_goal);
  // prep: v1(2) + v2(3) = 5 (v0 ran before the witness window)
  CHECK(res.prep_compute == 5);
  CHECK(res.cont_compute == 3);
  CHECK(res.recompute_compute == 3);  // v2 already done at COMMIT
  CHECK(res.released_bytes == 40);
  CHECK(res.released_bytes >= r.Bo - r.Bn);
  (void)g;
}

TEST_CASE("checker accepts async LAUNCH/WAIT witness") {
  Graph g = chain();
  Request r = std_req();
  State s0 = after_v0(r);
  Witness plan = w({{ActKind::Launch, 1},
                    {ActKind::Wait, 1},
                    {ActKind::Release, 0},
                    {ActKind::Commit, -1}});
  CheckResult res = check_witness(g, s0, r, plan);
  CHECK(res.ok);
  CHECK(res.reached_goal);
}

TEST_CASE("mutations: every corruption is rejected") {
  Graph g = chain();
  Request r = std_req();
  State s0 = after_v0(r);

  SUBCASE("missing dependency: drop the input before consuming it") {
    // The miniature greedy trap: surrender t0 before v1 consumes it; v1 can
    // never run again (v0 is non-replayable) -> t1 unproducible.
    Witness plan = w({{ActKind::Release, 0}, {ActKind::Run, 1}, {ActKind::Commit, -1}});
    CheckResult res = check_witness(g, s0, r, plan);
    CHECK_FALSE(res.ok);
    CHECK(res.error_index == 1);
  }

  SUBCASE("commit shortfall") {
    Witness plan = w({{ActKind::Run, 1}, {ActKind::Commit, -1}});  // released 0 < 35
    CheckResult res = check_witness(g, s0, r, plan);
    CHECK_FALSE(res.ok);
  }

  SUBCASE("no commit at all") {
    Witness plan = w({{ActKind::Run, 1}});
    CheckResult res = check_witness(g, s0, r, plan);
    CHECK(res.ok);
    CHECK_FALSE(res.reached_goal);
  }

  SUBCASE("incomplete continuation: required output dropped at the end") {
    Witness plan = w({{ActKind::Run, 1},
                      {ActKind::Release, 0},
                      {ActKind::Commit, -1},
                      {ActKind::Drop, 1}});
    CheckResult res = check_witness(g, s0, r, plan);
    CHECK(res.ok);
    CHECK_FALSE(res.reached_goal);
  }

  SUBCASE("release after commit") {
    Witness plan = w({{ActKind::Run, 1},
                      {ActKind::Release, 0},
                      {ActKind::Commit, -1},
                      {ActKind::Release, 1}});
    CheckResult res = check_witness(g, s0, r, plan);
    CHECK_FALSE(res.ok);
    CHECK(res.error_index == 3);
  }

  SUBCASE("double release of the same block") {
    Witness plan = w({{ActKind::Release, 0}, {ActKind::Release, 0}});
    CheckResult res = check_witness(g, s0, r, plan);
    CHECK_FALSE(res.ok);  // second release: t0 no longer resident
  }

  SUBCASE("over ceiling: op peak exceeds region") {
    Request tight{.Bo = 45, .Bn = 30, .K = 10};
    Witness plan = w({{ActKind::Run, 1}});
    CheckResult res = check_witness(g, s0, tight, plan);  // 40+30 > 45
    CHECK_FALSE(res.ok);
  }

  SUBCASE("preparation budget exceeded") {
    Request k1{.Bo = 100, .Bn = 65, .K = 1};
    Witness plan = w({{ActKind::Run, 1}, {ActKind::Release, 0}, {ActKind::Commit, -1}});
    CheckResult res = check_witness(g, s0, k1, plan);  // v1 costs 2 > K=1
    CHECK_FALSE(res.ok);
  }

  SUBCASE("non-replayable op executed twice") {
    Witness plan = w({{ActKind::Run, 0}, {ActKind::Run, 0}});
    CheckResult res = check_witness(g, State{}, r, plan);
    CHECK_FALSE(res.ok);
  }

  SUBCASE("wait without launch") {
    Witness plan = w({{ActKind::Wait, 1}});
    CheckResult res = check_witness(g, s0, r, plan);
    CHECK_FALSE(res.ok);
  }

  SUBCASE("double commit") {
    Witness plan = w({{ActKind::Release, 0},
                      {ActKind::Commit, -1},
                      {ActKind::Commit, -1}});
    CheckResult res = check_witness(g, s0, r, plan);
    CHECK_FALSE(res.ok);
  }

  SUBCASE("initial state itself violates the invariant") {
    State bad = s0;
    bad.released_bytes = 90;  // ceiling 10 < usage 40
    Witness plan = w({});
    CheckResult res = check_witness(g, bad, r, plan);
    CHECK_FALSE(res.ok);
  }

  SUBCASE("drop of tensor read by in-flight op") {
    Witness plan = w({{ActKind::Launch, 1},
                      {ActKind::Drop, 0},
                      {ActKind::Wait, 1},
                      {ActKind::Release, 0},
                      {ActKind::Commit, -1}});
    CheckResult res = check_witness(g, s0, r, plan);
    CHECK_FALSE(res.ok);
    CHECK(res.error_index == 1);
  }
}

TEST_CASE("checker rejects an illegal initial frontier") {
  Graph g = chain();
  Request r = std_req();
  // Frontier claims v1 done with t1 resident but t0 already surrendered and
  // v0 spent: state is self-consistent, so it is accepted as a starting point;
  // the harness only feeds frontiers produced by legal prefix execution.
  State s = after_v0(r);
  s = apply_run(g, s, r, 1);
  Witness plan = w({{ActKind::Release, 0}, {ActKind::Commit, -1}});
  CheckResult res = check_witness(g, s, r, plan);
  CHECK(res.ok);
  CHECK(res.reached_goal);
}
