// Policy tests: the flagship separations (docs/EXPERIMENTS.md hypotheses),
// each verified against the independent checker. These pin the BEHAVIORAL
// claims: which policies find which handovers on the declared traps.

#include <doctest/doctest.h>

#include <keel/checker.hpp>
#include <keel/gen.hpp>
#include <keel/policy.hpp>

using namespace keel;

TEST_CASE("oracle policy mapping preserves UNKNOWN") {
  CHECK(policy_status_from_oracle(OracleStatus::Feasible) ==
        PolicyStatus::Accept);
  CHECK(policy_status_from_oracle(OracleStatus::Infeasible) ==
        PolicyStatus::Reject);
  CHECK(policy_status_from_oracle(OracleStatus::Unknown) ==
        PolicyStatus::Unknown);
}

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

void expect_valid_accept(const Graph& g, const State& s0, const Request& r,
                         const PolicyOutcome& o) {
  REQUIRE(o.status == PolicyStatus::Accept);
  CheckResult c = check_witness(g, s0, r, o.witness);
  CHECK(c.ok);
  CHECK(c.reached_goal);
  CHECK(o.released_bytes >= r.Bo - r.Bn);
}

}  // namespace

TEST_CASE("prep_unlock separates drop-only policies from planned prep") {
  Generated gen = generate(Family::PrepUnlock, CostDist::Uniform, 1);
  const Graph& g = gen.graph;
  State s0 = run_prefix(g, gen.bo, {0});
  Request r{200, 120, 3};  // target 80; K = cost(v1) exactly
  PolicyConfig cfg;

  PolicyOutcome cons = conservative(g, s0, r, cfg);
  CHECK(cons.status == PolicyStatus::Reject);  // drop-only can never work

  PolicyOutcome grd = guarded_greedy(g, s0, r, cfg);
  CHECK(grd.status == PolicyStatus::Reject);  // validated drop-only

  PolicyOutcome keel = keel_plan(g, s0, r, cfg);
  expect_valid_accept(g, s0, r, keel);  // planned prep compute unlocks it
  CHECK(keel.prep_compute >= 3);
}

TEST_CASE("unguarded greedy commits into a dead continuation (H3)") {
  Generated gen = generate(Family::TrapNonreplayable, CostDist::Uniform, 1);
  const Graph& g = gen.graph;  // Bo = 240
  State s0 = run_prefix(g, gen.bo, {0, 1, 2});  // s, a, b resident
  Request r{240, 170, 10};  // target 70
  PolicyConfig cfg;

  PolicyOutcome grd = greedy(g, s0, r, cfg);
  // Largest-first surrenders s (80 >= 70) -> v3 can never run -> FAILED.
  CHECK(grd.status == PolicyStatus::Failed);
  CheckResult c = check_witness(g, s0, r, grd.witness);
  CHECK(c.ok);                    // every action was legal...
  CHECK_FALSE(c.reached_goal);    // ...but the goal is unreachable

  PolicyOutcome guarded = guarded_greedy(g, s0, r, cfg);
  expect_valid_accept(g, s0, r, guarded);  // smallest-first order survives

  PolicyOutcome keel = keel_plan(g, s0, r, cfg);
  expect_valid_accept(g, s0, r, keel);
}

TEST_CASE("trap_recompute at target 55: guarded orders all fail, keel finds it") {
  Generated gen = generate(Family::TrapRecompute, CostDist::Uniform, 1);
  const Graph& g = gen.graph;
  State s0 = run_prefix(g, gen.bo, {0, 1, 2});
  Request r{160, 105, 10};  // target 55
  PolicyConfig cfg;

  PolicyOutcome grd = greedy(g, s0, r, cfg);
  CHECK(grd.status == PolicyStatus::Failed);  // {x,s} overshoot is fatal

  PolicyOutcome guarded = guarded_greedy(g, s0, r, cfg);
  // All three deterministic orders overshoot or starve; the surviving plan
  // (release {x,y}, re-derive from s) is outside the drop-only ladder space.
  CHECK(guarded.status == PolicyStatus::Reject);

  PolicyOutcome keel = keel_plan(g, s0, r, cfg);
  expect_valid_accept(g, s0, r, keel);
}

TEST_CASE("honest rejections on genuinely infeasible requests") {
  Generated gen = generate(Family::Chain, CostDist::Uniform, 3);
  const Graph& g = gen.graph;
  State s0 = run_prefix(g, gen.bo, {0});
  // Target larger than anything the task can ever surrender (see SPEC):
  // Bn = 1 with Bo = gen.bo asks for an absurd release.
  Request r{gen.bo, 1, 1000};
  PolicyConfig cfg;
  CHECK(conservative(g, s0, r, cfg).status == PolicyStatus::Reject);
  CHECK(greedy(g, s0, r, cfg).status != PolicyStatus::Accept);
  CHECK(guarded_greedy(g, s0, r, cfg).status == PolicyStatus::Reject);
  PolicyOutcome keel = keel_plan(g, s0, r, cfg);
  CHECK(keel.status != PolicyStatus::Accept);  // never a false accept
  OracleResult o = solve_transition(g, s0, r, OracleConfig{});
  CHECK(o.status != OracleStatus::Feasible);
}

TEST_CASE("all policies accept an easy handover") {
  Generated gen = generate(Family::Chain, CostDist::Uniform, 3);
  const Graph& g = gen.graph;
  State s0 = run_prefix(g, gen.bo, {0, 1});  // everything produced
  // Target: smallest single tensor in the frontier — always achievable.
  int64_t smallest = INT64_MAX;
  for (auto& t : g.tensors) smallest = std::min(smallest, t.size);
  Request r{gen.bo, gen.bo - smallest, 10};
  PolicyConfig cfg;
  expect_valid_accept(g, s0, r, conservative(g, s0, r, cfg));
  expect_valid_accept(g, s0, r, greedy(g, s0, r, cfg));
  expect_valid_accept(g, s0, r, guarded_greedy(g, s0, r, cfg));
  expect_valid_accept(g, s0, r, keel_plan(g, s0, r, cfg));
}

TEST_CASE("keel regression: WAIT during preparation (async frontier)") {
  // Defect (fixed): keel's prep search generated WAIT only after a can_run
  // guard that in-flight ops always fail, so plans that must complete
  // in-flight work BEFORE releasing (the oracle's plan on this instance
  // starts with WAIT v3) were outside keel's candidate space -> honest but
  // unnecessary REJECT. The fix generates WAIT independently of RUN legality.
  Generated gen = generate(Family::Chain, CostDist::Uniform, 131);
  const Graph& g = gen.graph;
  Rng rng(1030);
  State s0 = sample_frontier(g, gen.bo, rng, 1, (int)g.ops.size() - 1, true, 2);
  REQUIRE(s0.inflight != 0);
  int64_t rem = remaining_work(g, s0);
  Request r{gen.bo, (int64_t)(gen.bo * 0.5), (int64_t)(rem * 0.10)};
  OracleConfig oc;
  oc.allow_async = true;
  OracleResult o = solve_transition(g, s0, r, oc);
  REQUIRE(o.status == OracleStatus::Feasible);
  PolicyOutcome k = keel_plan(g, s0, r, PolicyConfig{});
  expect_valid_accept(g, s0, r, k);
}
