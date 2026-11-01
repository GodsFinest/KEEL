// AUDIT REPRODUCTION (docs/PHASE1_AUDIT_REMEDIATION.md, finding 1).
// Minimal async-overcommit counterexample from the independent review:
//   Bo=10, Bn=9, initial releasable tensor 1 unit, two independent ops each
//   producing 6 units (one required, one disposable).
//   Trace: RELEASE(init) COMMIT LAUNCH(A) LAUNCH(B) WAIT(A) WAIT(B) DROP(A)
// The two in-flight output reservations (6 + 6 = 12) collectively exceed the
// 9-unit region. The trace must be REJECTED regardless of the final footprint.
//
// Expected status after remediation: every variant below REJECTS.
// Before remediation: the checker/oracle accept it (reproduction of the finding).

#include <doctest/doctest.h>

#include <keel/checker.hpp>
#include <keel/model.hpp>
#include <keel/oracle.hpp>

#include <string>

using namespace keel;

namespace {

struct OvercommitGraph {
  Graph g;
  TensorId init_t;
  TensorId a_t;  // disposable output
  TensorId b_t;  // required output
  OpId a, b;

  static OvercommitGraph make() {
    OvercommitGraph og;
    GraphBuilder bd;
    og.init_t = bd.add_tensor(1, -1);
    bd.add_op({}, {og.init_t}, 0, 1, /*replayable=*/false);  // v0: source
    og.a_t = bd.add_tensor(6, -1);
    og.a = bd.add_op({}, {og.a_t}, 0, 1, true);              // A
    og.b_t = bd.add_tensor(6, -1, /*required=*/true);
    og.b = bd.add_op({}, {og.b_t}, 0, 1, true);              // B
    og.g = bd.finish();
    return og;
  }

  // Frontier: source executed, init tensor resident.
  State frontier(const Request& r) const {
    return capture_frontier(apply_run(g, State{}, Request{r.Bo, r.Bn, 1 << 20}, 0));
  }
};

Witness audit_trace(const OvercommitGraph& og) {
  return Witness{{
      {ActKind::Release, og.init_t},
      {ActKind::Commit, -1},
      {ActKind::Launch, og.a},
      {ActKind::Launch, og.b},
      {ActKind::Wait, og.a},
      {ActKind::Wait, og.b},
      {ActKind::Drop, og.a_t},  // legal-looking cleanup; final usage = 6 <= 9
  }};
}

const Request& audit_request() {
  static const Request r{.Bo = 10, .Bn = 9, .K = 100};
  return r;
}

}  // namespace

TEST_CASE("audit f1: checker must reject the async-overcommit trace") {
  auto og = OvercommitGraph::make();
  const Request& r = audit_request();
  State s0 = og.frontier(r);
  CheckResult res = check_witness(og.g, s0, r, audit_trace(og));
  // REMEDIATION TARGET: ok must be false (transient overcommit at WAIT(B)).
  CHECK_MESSAGE(!res.ok,
                "FINDING 1 REPRODUCED: checker accepted a trace whose "
                "in-flight output reservations (6+6) exceed the 9-unit grant");
  if (!res.ok) {
    MESSAGE("checker rejected at index ", res.error_index, ": ", res.error);
  }
}

TEST_CASE("audit f1: model primitives block concurrent overcommit launches") {
  auto og = OvercommitGraph::make();
  const Request& r = audit_request();
  State s0 = og.frontier(r);

  // RELEASE(init) -> COMMIT -> LAUNCH(A): legal. LAUNCH(B) while A's 6-unit
  // output reservation is live must now be ILLEGAL (12 > 9).
  State s1 = s0;
  REQUIRE(can_release(og.g, s1, og.init_t).ok);
  s1 = apply_release(og.g, s1, og.init_t);
  REQUIRE(can_commit(og.g, s1, r).ok);
  s1 = apply_commit(og.g, s1, r);
  REQUIRE(can_launch(og.g, s1, r, og.a).ok);
  s1 = apply_launch(og.g, s1, r, og.a);
  Legality l = can_launch(og.g, s1, r, og.b);
  CHECK_MESSAGE(!l.ok,
                "FINDING 1 FIX: second concurrent launch must be illegal");
  CHECK(l.err == Err::OVER_CEILING);
  // Every completion ordering stays within the region: WAIT only converts the
  // reservation into residency (usage never grows at WAIT).
  State s2 = apply_wait(og.g, s1, og.a);
  CHECK(memory_of(og.g, s2, r).usage == 6);
}

TEST_CASE("audit f1: oracle under v1.1 emits only valid witnesses") {
  auto og = OvercommitGraph::make();
  const Request& r = audit_request();
  State s0 = og.frontier(r);
  OracleConfig cfg;
  cfg.allow_async = true;
  cfg.max_expansions = 100000;
  OracleResult res = solve_transition(og.g, s0, r, cfg);
  if (res.status == OracleStatus::Feasible) {
    // A legal serialized plan exists (WAIT A, DROP its disposable output,
    // then LAUNCH B); whatever the oracle finds must be checker-valid.
    CheckResult c = check_witness(og.g, s0, r, res.witness);
    const std::string msg =
        std::string("oracle witness must be valid under v1.1: ") + c.error;
    CHECK_MESSAGE(c.ok, msg);
    CHECK(c.reached_goal);
  }
}

// ---------------------------------------------------------------------------
// Finding 2 mutation tests: temporary overcommit followed by legal-looking
// cleanup must be rejected in BOTH phases, even though the final state fits
// the grant.

TEST_CASE("audit f2: preparation-phase overcommit with cleanup is rejected") {
  auto og = OvercommitGraph::make();
  const Request& r = audit_request();
  State s0 = og.frontier(r);
  // No RELEASE/COMMIT here: ceiling is Bo = 10. Two concurrent 6-unit
  // reservations = 12 > 10; dropping afterwards "cleans up" to 6 <= 10.
  Witness prep_overcommit = Witness{{
      {ActKind::Launch, og.a},
      {ActKind::Launch, og.b},
      {ActKind::Wait, og.a},
      {ActKind::Wait, og.b},
      {ActKind::Drop, og.a_t},
      {ActKind::Release, og.init_t},
      {ActKind::Commit, -1},
  }};
  CheckResult res = check_witness(og.g, s0, r, prep_overcommit);
  CHECK_MESSAGE(!res.ok,
                "F2: prep-phase overcommit with cleanup must be rejected");
}

TEST_CASE("audit f2: waiting first does not hide the overcommit") {
  auto og = OvercommitGraph::make();
  const Request& r = audit_request();
  State s0 = og.frontier(r);
  // Serialize the waits but never drop before both outputs exist: peak
  // resident 12 > 9 after the second WAIT. Final DROP "cleans up".
  Witness wait_first = Witness{{
      {ActKind::Release, og.init_t},
      {ActKind::Commit, -1},
      {ActKind::Launch, og.a},
      {ActKind::Wait, og.a},
      {ActKind::Launch, og.b},
      {ActKind::Wait, og.b},
      {ActKind::Drop, og.a_t},
  }};
  CheckResult res = check_witness(og.g, s0, r, wait_first);
  CHECK_MESSAGE(!res.ok,
                "F2: post-WAIT residency overcommit (12 > 9) must be rejected");
  // The legal serialization drops the disposable output BEFORE launching B.
  Witness legal_serial = Witness{{
      {ActKind::Release, og.init_t},
      {ActKind::Commit, -1},
      {ActKind::Launch, og.a},
      {ActKind::Wait, og.a},
      {ActKind::Drop, og.a_t},
      {ActKind::Launch, og.b},
      {ActKind::Wait, og.b},
  }};
  CheckResult ok = check_witness(og.g, s0, r, legal_serial);
  const std::string okmsg =
      std::string("legal serialization must be accepted: ") + ok.error;
  CHECK_MESSAGE(ok.ok, okmsg);
  CHECK(ok.reached_goal);
}

// ---------------------------------------------------------------------------
// Finding 3: input and arithmetic validation regressions.

TEST_CASE("audit f3: request validation") {
  CHECK(!validate_request({.Bo = 0, .Bn = 0, .K = 0}).empty());
  CHECK(!validate_request({.Bo = 10, .Bn = -1, .K = 0}).empty());
  CHECK(!validate_request({.Bo = 10, .Bn = 10, .K = 0}).empty());
  CHECK(!validate_request({.Bo = 10, .Bn = 11, .K = 0}).empty());
  CHECK(!validate_request({.Bo = 10, .Bn = 5, .K = -1}).empty());
  CHECK(validate_request({.Bo = 10, .Bn = 9, .K = 0}).empty());
}

TEST_CASE("audit f3: builder rejects negative scratch and preserves sanity") {
  GraphBuilder b;
  TensorId t = b.add_tensor(4, -1);
  CHECK_THROWS(b.add_op({}, {t}, /*scratch=*/-1, /*cost=*/1));
  CHECK_THROWS(b.add_op({}, {t}, 0, /*cost=*/-3));
  CHECK_THROWS(b.add_tensor(-4, -1));
}

TEST_CASE("audit f3: out-of-range ids are refused, not UB") {
  auto og = OvercommitGraph::make();
  const Request& r = audit_request();
  State s;
  CHECK(!can_run(og.g, s, r, -1).ok);
  CHECK(!can_run(og.g, s, r, 99).ok);
  CHECK(!can_drop(og.g, s, -1).ok);
  CHECK(!can_drop(og.g, s, 42).ok);
  CHECK(!can_release(og.g, s, -3).ok);
  CHECK(!can_wait(og.g, s, -1).ok);
  CHECK(!can_wait(og.g, s, 77).ok);
}

TEST_CASE("audit f3: malformed frontier states are detected") {
  auto og = OvercommitGraph::make();
  const Request& r = audit_request();

  SUBCASE("resident tensor without completed producer") {
    State s;
    s.resident = TensorMask(1) << og.a_t;
    CHECK(!validate_state(og.g, s, r).empty());
  }
  SUBCASE("op both done and in flight") {
    State s;
    s.done = OpMask(1) << og.a;
    s.inflight = OpMask(1) << og.a;
    CHECK(!validate_state(og.g, s, r).empty());
  }
  SUBCASE("in-flight op output already resident") {
    State s;
    s.inflight = OpMask(1) << og.a;
    s.resident = TensorMask(1) << og.a_t;
    CHECK(!validate_state(og.g, s, r).empty());
  }
  SUBCASE("released bytes above the grant") {
    State s;
    s.released_bytes = r.Bo + 1;
    CHECK(!validate_state(og.g, s, r).empty());
  }
  SUBCASE("prep budget already exceeded pre-commit") {
    State s;
    s.prep_used = r.K + 1;
    CHECK(!validate_state(og.g, s, r).empty());
  }
  SUBCASE("negative preparation usage") {
    State s;
    s.prep_used = -1;
    CHECK(!validate_state(og.g, s, r).empty());
  }
  SUBCASE("out-of-range state masks") {
    State s;
    s.done = OpMask(1) << og.g.ops.size();
    CHECK(!validate_state(og.g, s, r).empty());
    s = State{};
    s.resident = TensorMask(1) << og.g.tensors.size();
    CHECK(!validate_state(og.g, s, r).empty());
  }
  SUBCASE("committed state must already meet the release target") {
    State s;
    s.committed = true;
    CHECK(!validate_state(og.g, s, r).empty());
  }
  SUBCASE("invariant-violating state") {
    State s;
    s.resident = (TensorMask(1) << og.a_t) | (TensorMask(1) << og.b_t) |
                 (TensorMask(1) << og.init_t);
    s.done = OpMask(1) << 0;  // source produced init_t
    CHECK(!validate_state(og.g, s, r).empty());  // 13 > 10
  }
  SUBCASE("legal frontier passes") {
    State s = capture_frontier(apply_run(og.g, State{}, r, 0));
    CHECK(validate_state(og.g, s, r).empty());
  }
}

TEST_CASE("audit f3: checker rejects malformed requests outright") {
  auto og = OvercommitGraph::make();
  State s0 = og.frontier(audit_request());
  Request bad{.Bo = 5, .Bn = 9, .K = 1};  // Bn >= Bo
  CheckResult res = check_witness(og.g, s0, bad, Witness{});
  CHECK_FALSE(res.ok);
}

TEST_CASE("audit f3: checker rejects unknown action kinds") {
  auto og = OvercommitGraph::make();
  State s0 = og.frontier(audit_request());
  const ActKind unknown = static_cast<ActKind>(99);
  CheckResult res = check_witness(og.g, s0, audit_request(),
                                  Witness{{Act{unknown, -1}}});
  CHECK_FALSE(res.ok);
  CHECK(res.error_index == 0);
}
