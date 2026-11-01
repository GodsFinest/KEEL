// KEEL Phase 1 — Seeded graph generators and live-frontier sampler.
//
// Families (docs/EXPERIMENTS.md grid): chain, diamond, shared_ancestor,
// fork_join, residual, big_workspace, heterogeneous — organic; plus the four
// DECLARED adversarial traps (trap_recompute, trap_nonreplayable, trap_memory,
// prep_unlock) which are hand-tuned structural instances and are reported
// separately from headline rates.
//
// Bo semantics (organic families): the task's original grant equals the
// sequential never-drop peak of the whole graph — the memory a naive runtime
// needs to run the task without any reclamation. Adversarial traps hand-tune
// Bo with explicit headroom where their design requires it (documented
// per-family in the report).
//
// Cost distributions: Uniform, HeavyTail (geometric sizes/costs),
// Decorrelated (large tensors on cheap ops). Traps use fixed sizes.

#pragma once

#include <keel/model.hpp>

#include <cstdint>

namespace keel {

enum class Family {
  Chain,
  Diamond,
  SharedAncestor,
  ForkJoin,
  Residual,
  BigWorkspace,
  Heterogeneous,
  TrapRecompute,
  TrapNonreplayable,
  TrapMemory,
  PrepUnlock,
  TrapConcurrentLaunch,  // audit finding 6: concurrent-launch oversubscription
};

const char* family_name(Family f);
const int kFamilyCount = 12;
bool family_is_adversarial(Family f);

enum class CostDist { Uniform, HeavyTail, Decorrelated };
const char* dist_name(CostDist d);

// Deterministic RNG (splitmix64). Same seed -> same sequence everywhere.
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed) {}
  uint64_t next_u64() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  // uniform integer in [lo, hi] inclusive
  int64_t uniform(int64_t lo, int64_t hi) {
    return lo + (int64_t)(next_u64() % (uint64_t)(hi - lo + 1));
  }
  bool chance(int percent) { return (int)(next_u64() % 100) < percent; }
};

struct Generated {
  Graph graph;
  int64_t bo = 0;  // never-drop sequential peak (organic families)
};

Generated generate(Family f, CostDist d, uint64_t seed);

// Sequential never-drop peak: run ops in topo order keeping every tensor;
// the peak includes each op's scratch + outputs before they materialize.
int64_t never_drop_peak(const Graph& g);

// A live frontier: legal prefix execution under the pre-request budget.
// Sequential mode runs `steps` legal RUNs (uniformly random among ready ops).
// Async mode additionally launches up to `max_inflight` ready ops and leaves
// them in flight. The returned state is capture_frontier-ed (prep_used = 0).
State sample_frontier(const Graph& g, int64_t bo, Rng& rng, int min_steps,
                      int max_steps, bool async, int max_inflight);

// Σ cost of ops not yet done and not in flight (the K denominator).
int64_t remaining_work(const Graph& g, const State& s);

}  // namespace keel
