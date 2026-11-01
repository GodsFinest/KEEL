// Dominance-pruning differential validation (docs/SPEC.md §9, mandate §7).
// Re-solves every instance with the pruned and unpruned oracle and requires
// exact agreement on (status, optimal cost). Any mismatch is a soundness bug
// in the dominance rule and fails the run.

#include <keel/checker.hpp>
#include <keel/gen.hpp>
#include <keel/oracle.hpp>

#include <cstdio>
#include <cstdlib>

using namespace keel;

int main(int argc, char** argv) {
  int seed_hi = 20;
  if (argc > 1) seed_hi = atoi(argv[1]);
  const Family families[] = {
      Family::Chain,        Family::Diamond,         Family::SharedAncestor,
      Family::ForkJoin,     Family::Residual,        Family::BigWorkspace,
      Family::Heterogeneous, Family::TrapRecompute,
      Family::TrapNonreplayable, Family::TrapMemory, Family::PrepUnlock,
      Family::TrapConcurrentLaunch};
  const CostDist dists[] = {CostDist::Uniform, CostDist::HeavyTail,
                            CostDist::Decorrelated};
  const double bn_fracs[] = {0.50, 0.65, 0.80};
  const double k_fracs[] = {0.0, 0.10, 0.25};

  size_t checked = 0, feasible = 0, mismatches = 0, skipped_unknown = 0;
  for (Family f : families) {
    int reps = family_is_adversarial(f) ? 1 : 3;
    for (CostDist d : dists) {
      if (family_is_adversarial(f) && d != CostDist::Uniform) continue;
      for (int rep = 0; rep < reps; ++rep) {
        for (int seed = 1; seed <= seed_hi; ++seed) {
          Generated gen = generate(f, d, (uint64_t)seed * 131u + rep);
          for (int fseed = 0; fseed < 2; ++fseed) {
            Rng rng((uint64_t)seed * 977u + fseed * 31u + rep * 7u);
            State s0 = sample_frontier(gen.graph, gen.bo, rng, 1,
                                       (int)gen.graph.ops.size() - 1, false, 0);
            if (s0.resident == 0) continue;
            int64_t remaining = remaining_work(gen.graph, s0);
            for (double bf : bn_fracs) {
              int64_t Bn = (int64_t)(gen.bo * bf);
              if (Bn <= 0) continue;
              for (double kf : k_fracs) {
                Request r{gen.bo, Bn, (int64_t)(remaining * kf)};
                OracleConfig plain{}, dom{};
                dom.use_dominance = true;
                OracleResult a = solve_transition(gen.graph, s0, r, plain);
                OracleResult b = solve_transition(gen.graph, s0, r, dom);
                ++checked;
                if (a.status == OracleStatus::Unknown ||
                    b.status == OracleStatus::Unknown) {
                  ++skipped_unknown;
                  continue;
                }
                if (a.status != b.status ||
                    (a.status == OracleStatus::Feasible &&
                     a.total_compute != b.total_compute)) {
                  ++mismatches;
                  printf("MISMATCH family=%s seed=%d rep=%d fseed=%d bn=%.2f "
                         "K=%ld: plain=%s(%ld) dom=%s(%ld)\n",
                         family_name(f), seed, rep, fseed, bf, (long)r.K,
                         oracle_status_name(a.status), (long)a.total_compute,
                         oracle_status_name(b.status), (long)b.total_compute);
                } else if (a.status == OracleStatus::Feasible) {
                  // Both witnesses must independently validate.
                  CheckResult ca = check_witness(gen.graph, s0, r, a.witness);
                  CheckResult cb = check_witness(gen.graph, s0, r, b.witness);
                  if (!ca.ok || !ca.reached_goal || !cb.ok || !cb.reached_goal) {
                    ++mismatches;
                    printf("WITNESS-INVALID family=%s seed=%d\n",
                           family_name(f), seed);
                  }
                  ++feasible;
                }
              }
            }
          }
        }
      }
    }
  }
  printf("dominance differential: checked=%zu feasible=%zu unknown=%zu "
         "MISMATCHES=%zu\n",
         checked, feasible, skipped_unknown, mismatches);
  return mismatches == 0 ? 0 : 1;
}
