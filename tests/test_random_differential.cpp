// Randomized differential testing (audit finding 7):
//  1. Model <-> checker agreement on random legal traces: any action sequence
//     built from model-legal steps must be accepted by the independent
//     checker, and the checker's reached_goal must match the model's goal.
//  2. Mutation equivalence: for randomly mutated traces, the checker's
//     verdict (ok / reached_goal) must EXACTLY match an independent
//     step-by-step model replay. A divergence in either direction is a
//     soundness bug in one of the two implementations.

#include <doctest/doctest.h>

#include <keel/checker.hpp>
#include <keel/gen.hpp>
#include <keel/oracle.hpp>

using namespace keel;

namespace {

struct Replay {
  bool legal_prefix;   // every act in `used` was model-legal
  bool goal;           // the legal prefix reached the goal
  size_t used;         // number of legal prefix acts
};

// Replay a witness through the MODEL primitives (never the checker) and
// report the legal prefix and whether it ends in the goal.
Replay model_replay(const Graph& g, State s, const Request& r,
                    const Witness& w) {
  Replay rep{true, is_goal(g, s), 0};
  for (const Act& a : w.acts) {
    Legality L{false, Err::OK};
    switch (a.kind) {
      case ActKind::Run: L = can_run(g, s, r, a.target); break;
      case ActKind::Launch: L = can_launch(g, s, r, a.target); break;
      case ActKind::Wait: L = can_wait(g, s, a.target); break;
      case ActKind::Drop: L = can_drop(g, s, a.target); break;
      case ActKind::Release: L = can_release(g, s, a.target); break;
      case ActKind::Commit: L = can_commit(g, s, r); break;
    }
    if (!L.ok) {
      rep.legal_prefix = false;
      break;
    }
    switch (a.kind) {
      case ActKind::Run: s = apply_run(g, s, r, a.target); break;
      case ActKind::Launch: s = apply_launch(g, s, r, a.target); break;
      case ActKind::Wait: s = apply_wait(g, s, a.target); break;
      case ActKind::Drop: s = apply_drop(g, s, a.target); break;
      case ActKind::Release: s = apply_release(g, s, a.target); break;
      case ActKind::Commit: s = apply_commit(g, s, r); break;
    }
    ++rep.used;
    rep.goal = is_goal(g, s);
  }
  return rep;
}

Act random_mutation(Rng& rng, const Graph& g, size_t nops, size_t nts,
                    size_t pos) {
  Act a{ActKind::Run, 0};
  switch (rng.uniform(0, 5)) {
    case 0: a.kind = ActKind::Run; a.target = (int)rng.uniform(0, (int64_t)nops - 1); break;
    case 1: a.kind = ActKind::Launch; a.target = (int)rng.uniform(0, (int64_t)nops - 1); break;
    case 2: a.kind = ActKind::Wait; a.target = (int)rng.uniform(0, (int64_t)nops - 1); break;
    case 3: a.kind = ActKind::Drop; a.target = (int)rng.uniform(0, (int64_t)nts - 1); break;
    case 4: a.kind = ActKind::Release; a.target = (int)rng.uniform(0, (int64_t)nts - 1); break;
    default: a.kind = ActKind::Commit; break;
  }
  (void)g;
  (void)pos;
  return a;
}

}  // namespace

TEST_CASE("randomized differential: model and checker agree, mutations too") {
  const int kSeeds = 25;
  int traces_checked = 0, mutations_checked = 0, goals_seen = 0;
  for (int fi = 0; fi < kFamilyCount; ++fi) {
    Family f = (Family)fi;
    for (int di = 0; di < 3; ++di) {
      CostDist d = (CostDist)di;
      if (family_is_adversarial(f) && d != CostDist::Uniform) continue;
      int reps = family_is_adversarial(f) ? 1 : 2;
      for (int rep = 0; rep < reps; ++rep) {
        for (int seed = 1; seed <= kSeeds; ++seed) {
          Generated gen = generate(f, d, (uint64_t)seed * 131u + rep);
          const Graph& g = gen.graph;
          const size_t nops = g.ops.size(), nts = g.tensors.size();
          for (int fseed = 0; fseed < 2; ++fseed) {
            Rng frng((uint64_t)seed * 977u + fseed * 31u + rep * 7u);
            State s0 = sample_frontier(g, gen.bo, frng, 1,
                                       (int)nops - 1, fseed == 1, 2);
            if (s0.resident == 0 && s0.inflight == 0) continue;
            int64_t rem = remaining_work(g, s0);
            Request r{gen.bo, (int64_t)(gen.bo * 0.65),
                      (int64_t)(rem * 0.25)};

            // Build a random legal trace via model primitives.
            Rng wrng((uint64_t)seed * 31u + fseed + 1);
            Witness w;
            State s = s0;
            for (int step = 0; step < 24; ++step) {
              std::vector<Act> legal;
              for (OpId v = 0; v < (OpId)nops; ++v) {
                if (can_run(g, s, r, v).ok)
                  legal.push_back({ActKind::Run, v});
                if (can_launch(g, s, r, v).ok)
                  legal.push_back({ActKind::Launch, v});
                if (can_wait(g, s, v).ok)
                  legal.push_back({ActKind::Wait, v});
              }
              for (TensorId t = 0; t < (OpId)nts; ++t) {
                if (can_drop(g, s, t).ok)
                  legal.push_back({ActKind::Drop, t});
                if (can_release(g, s, t).ok)
                  legal.push_back({ActKind::Release, t});
              }
              if (can_commit(g, s, r).ok)
                legal.push_back({ActKind::Commit, -1});
              if (legal.empty()) break;
              Act a = legal[(size_t)wrng.uniform(
                  0, (int64_t)legal.size() - 1)];
              w.acts.push_back(a);
              switch (a.kind) {
                case ActKind::Run: s = apply_run(g, s, r, a.target); break;
                case ActKind::Launch: s = apply_launch(g, s, r, a.target); break;
                case ActKind::Wait: s = apply_wait(g, s, a.target); break;
                case ActKind::Drop: s = apply_drop(g, s, a.target); break;
                case ActKind::Release: s = apply_release(g, s, a.target); break;
                case ActKind::Commit: s = apply_commit(g, s, r); break;
              }
              if (is_goal(g, s)) break;
            }

            // Differential 1: model-legal trace must be checker-accepted with
            // matching goal verdict.
            CheckResult c = check_witness(g, s0, r, w);
            Replay rep1 = model_replay(g, s0, r, w);
            CHECK(c.ok);
            CHECK(c.reached_goal == rep1.goal);
            ++traces_checked;
            if (rep1.goal) ++goals_seen;

            // Differential 2: mutated traces — checker verdict must equal
            // independent model replay verdict.
            for (int m = 0; m < 4; ++m) {
              Witness mw = w;
              size_t pos = (size_t)wrng.uniform(0, (int64_t)mw.acts.size());
              mw.acts.insert(mw.acts.begin() + (long)pos,
                             random_mutation(wrng, g, nops, nts, pos));
              CheckResult mc = check_witness(g, s0, r, mw);
              Replay mr = model_replay(g, s0, r, mw);
              CHECK(mc.ok == mr.legal_prefix);
              if (mc.ok) CHECK(mc.reached_goal == mr.goal);
              ++mutations_checked;
            }
          }
        }
      }
    }
  }
  MESSAGE("randomized differential: traces=", traces_checked,
          " mutations=", mutations_checked, " goals=", goals_seen);
  CHECK(traces_checked > 300);
  CHECK(mutations_checked > 1200);
}
