// Unit tests for the IR / state / action model (docs/SPEC.md §2–§6).
// Each test cites the SPEC rule it pins down. Mutation-style checks for the
// checker live in test_checker.cpp; this file pins the shared primitives.

#include <doctest/doctest.h>

#include <keel/checker.hpp>
#include <keel/model.hpp>

using namespace keel;

namespace {

// v0 (source, non-replayable) -> t0(40);  v1: t0 -> t1(30). Q = {t1}.
Graph make_chain() {
  GraphBuilder b;
  b.add_op({}, {b.add_tensor(40, -1)}, 0, 1, /*replayable=*/false);
  TensorId t0 = 0;
  TensorId t1 = b.add_tensor(30, -1, /*required=*/true);
  b.add_op({t0}, {t1}, 0, 2, /*replayable=*/true);
  return b.finish();
}

State frontier_after(const Graph& g, const Request& r,
                     const std::vector<Act>& acts) {
  State s;
  for (const Act& a : acts) {
    switch (a.kind) {
      case ActKind::Run: {
        Legality l = can_run(g, s, r, a.target);
        REQUIRE(l.ok);
        s = apply_run(g, s, r, a.target);
        break;
      }
      case ActKind::Launch: {
        Legality l = can_launch(g, s, r, a.target);
        REQUIRE(l.ok);
        s = apply_launch(g, s, r, a.target);
        break;
      }
      case ActKind::Wait: {
        Legality l = can_wait(g, s, a.target);
        REQUIRE(l.ok);
        s = apply_wait(g, s, a.target);
        break;
      }
      case ActKind::Drop: {
        Legality l = can_drop(g, s, a.target);
        REQUIRE(l.ok);
        s = apply_drop(g, s, a.target);
        break;
      }
      case ActKind::Release: {
        Legality l = can_release(g, s, a.target);
        REQUIRE(l.ok);
        s = apply_release(g, s, a.target);
        break;
      }
      case ActKind::Commit: break;
    }
  }
  return s;
}

}  // namespace

TEST_CASE("builder rejects malformed graphs") {
  SUBCASE("duplicate producer") {
    GraphBuilder b;
    TensorId t = b.add_tensor(10, -1);
    b.add_op({}, {t}, 0, 1);
    CHECK_THROWS(b.add_op({}, {t}, 0, 1));
  }
  SUBCASE("in-place op") {
    GraphBuilder b;
    TensorId t = b.add_tensor(10, -1);
    CHECK_THROWS(b.add_op({t}, {t}, 0, 1));
  }
  SUBCASE("zero size / zero cost") {
    GraphBuilder b;
    CHECK_THROWS(b.add_tensor(0, -1));
    CHECK_THROWS(b.add_op({}, {b.add_tensor(4, -1)}, 0, 0));
  }
  SUBCASE("orphan tensor") {
    GraphBuilder b;
    b.add_tensor(4, -1);
    CHECK_THROWS(b.finish());
  }
  SUBCASE("cycle") {
    // Direct cycles are unrepresentable (single producer assignment + no
    // in-place overlap), so the closest structural check is the self-input
    // rejection above; Kahn's cycle detection guards anything that remains.
    GraphBuilder b;
    TensorId t = b.add_tensor(4, -1);
    CHECK_THROWS(b.add_op({t}, {t}, 0, 1));
  }
}

TEST_CASE("accounting: region shrinks only via RELEASE (SPEC §4)") {
  Graph g = make_chain();
  Request r{.Bo = 100, .Bn = 65, .K = 5};
  State s = frontier_after(g, r, {{ActKind::Run, 0}});

  Memory m0 = memory_of(g, s, r);
  CHECK(m0.usage == 40);
  CHECK(m0.ceiling == 100);

  s = apply_run(g, s, r, 1);  // v1 consumes t0, produces t1
  CHECK(memory_of(g, s, r).usage == 70);
  CHECK(memory_of(g, s, r).ceiling == 100);

  s = apply_drop(g, s, 0);  // DROP returns bytes to the task's OWN pool only
  CHECK(memory_of(g, s, r).ceiling == 100);
  CHECK(memory_of(g, s, r).usage == 30);

  State s2 = frontier_after(g, r, {{ActKind::Run, 0}, {ActKind::Run, 1}});
  s2 = apply_release(g, s2, 0);  // surrender t0's live block to the co-tenant
  CHECK(memory_of(g, s2, r).ceiling == 60);  // region is Bo - released
  CHECK(memory_of(g, s2, r).usage == 30);

  State c = apply_commit(g, s2, r);
  CHECK(c.committed);
  // Ceiling stays Bo - released = 60 <= Bn (65) after COMMIT.
  CHECK(memory_of(g, c, r).ceiling == 60);
  CHECK(memory_of(g, c, r).usage == 30);
}

TEST_CASE("RUN legality (SPEC §5)") {
  Graph g = make_chain();
  Request r{.Bo = 100, .Bn = 65, .K = 3};
  State s0;

  SUBCASE("missing input") {
    Legality l = can_run(g, s0, r, 1);
    CHECK_FALSE(l.ok);
    CHECK(l.err == Err::INPUTS_NOT_RESIDENT);
  }
  SUBCASE("non-replayable op runs once") {
    State s = apply_run(g, s0, r, 0);
    s = apply_drop(g, s, 0);
    Legality l = can_run(g, s, r, 0);
    CHECK_FALSE(l.ok);
    CHECK(l.err == Err::NOT_REPLAYABLE);
  }
  SUBCASE("no re-run while output resident") {
    State s = apply_run(g, s0, r, 0);
    Legality l = can_run(g, s, r, 0);
    CHECK_FALSE(l.ok);
    CHECK(l.err == Err::OUTPUT_ALREADY_RESIDENT);
  }
  SUBCASE("prep budget K") {
    Request k2{.Bo = 100, .Bn = 65, .K = 3};
    State s = apply_run(g, apply_run(g, s0, k2, 0), k2, 1);  // prep_used = 3
    s = apply_drop(g, s, 1);
    Legality l2 = can_run(g, s, k2, 1);  // cost 2 would make prep_used 5 > 3
    CHECK_FALSE(l2.ok);
    CHECK(l2.err == Err::PREP_BUDGET_EXCEEDED);
  }
  SUBCASE("over ceiling") {
    Request tight{.Bo = 45, .Bn = 30, .K = 10};
    State s = apply_run(g, s0, tight, 0);
    Legality l = can_run(g, s, tight, 1);  // 40 + 30 > 45
    CHECK_FALSE(l.ok);
    CHECK(l.err == Err::OVER_CEILING);
  }
}

TEST_CASE("in-flight protection (SPEC §5, async F)") {
  Graph g = make_chain();
  Request r{.Bo = 100, .Bn = 65, .K = 10};
  State s = frontier_after(g, r, {{ActKind::Run, 0}});  // t0 resident

  Legality l = can_launch(g, s, r, 1);
  REQUIRE(l.ok);
  State inf = apply_launch(g, s, r, 1);

  // t0 is read by in-flight v1: DROP and RELEASE are illegal.
  CHECK_FALSE(can_drop(g, inf, 0).ok);
  CHECK_FALSE(can_release(g, inf, 0).ok);
  CHECK(can_drop(g, inf, 1).ok == false);  // t1 not resident anyway
  CHECK(can_wait(g, inf, 1).ok);
  CHECK_FALSE(can_wait(g, inf, 0).ok);

  State done = apply_wait(g, inf, 1);
  CHECK(can_drop(g, done, 0).ok);  // protection ends at completion
  CHECK(done.inflight == 0);
  CHECK((done.done & 2) != 0);
}

TEST_CASE("pinned tensors are never droppable (SPEC §2 P)") {
  GraphBuilder b;
  b.add_op({}, {b.add_tensor(10, -1, /*required=*/false, /*pinned=*/true)},
           0, 1, false);
  Graph g = b.finish();
  Request r{.Bo = 100, .Bn = 50, .K = 10};
  State s = apply_run(g, State{}, r, 0);
  CHECK_FALSE(can_drop(g, s, 0).ok);
  CHECK(can_drop(g, s, 0).err == Err::PINNED);
  CHECK_FALSE(can_release(g, s, 0).ok);
}

TEST_CASE("COMMIT contract (SPEC §5)") {
  Graph g = make_chain();
  Request r{.Bo = 100, .Bn = 65, .K = 10};
  State s = frontier_after(g, r, {{ActKind::Run, 0}});

  SUBCASE("shortfall rejected") {
    Legality l = can_commit(g, s, r);  // released 0 < 35
    CHECK_FALSE(l.ok);
    CHECK(l.err == Err::COMMIT_SHORTFALL);
  }
  SUBCASE("after sufficient release accepted, then floor applies") {
    s = apply_run(g, s, r, 1);   // v1 consumes t0, produces t1
    s = apply_release(g, s, 0);  // surrender t0 (40 bytes) >= Bo - Bn = 35
    CHECK(can_commit(g, s, r).ok);
    State c = apply_commit(g, s, r);
    CHECK(c.committed);
    CHECK_FALSE(can_release(g, c, 1).ok);  // RELEASE outside PREPARING
    CHECK(can_release(g, c, 1).err == Err::RELEASE_OUTSIDE_PREPARING);
    // DROP is still legal in the continuation.
    CHECK(can_drop(g, c, 1).ok);
  }
  SUBCASE("double commit rejected") {
    s = apply_release(g, s, 0);
    State c = apply_commit(g, s, r);
    CHECK_FALSE(can_commit(g, c, r).ok);
    CHECK(can_commit(g, c, r).err == Err::ALREADY_COMMITTED);
  }
}

TEST_CASE("goal condition (SPEC §5)") {
  Graph g = make_chain();
  Request r{.Bo = 100, .Bn = 65, .K = 10};
  State s = frontier_after(g, r, {{ActKind::Run, 0}});
  CHECK_FALSE(is_goal(g, s));
  s = apply_run(g, s, r, 1);
  CHECK_FALSE(is_goal(g, s));  // not committed
  s = apply_release(g, s, 0);  // surrender t0; t1 stays (it is Q)
  s = apply_commit(g, s, r);
  CHECK(is_goal(g, s));  // Q = {t1} resident, committed, nothing in flight
}
