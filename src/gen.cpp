#include <keel/gen.hpp>

#include <algorithm>
#include <stdexcept>

namespace keel {

const char* family_name(Family f) {
  switch (f) {
    case Family::Chain: return "chain";
    case Family::Diamond: return "diamond";
    case Family::SharedAncestor: return "shared_ancestor";
    case Family::ForkJoin: return "fork_join";
    case Family::Residual: return "residual";
    case Family::BigWorkspace: return "big_workspace";
    case Family::Heterogeneous: return "heterogeneous";
    case Family::TrapRecompute: return "trap_recompute";
    case Family::TrapNonreplayable: return "trap_nonreplayable";
    case Family::TrapMemory: return "trap_memory";
    case Family::PrepUnlock: return "prep_unlock";
    case Family::TrapConcurrentLaunch: return "trap_concurrent_launch";
  }
  return "?";
}

bool family_is_adversarial(Family f) {
  switch (f) {
    case Family::TrapRecompute:
    case Family::TrapNonreplayable:
    case Family::TrapMemory:
    case Family::PrepUnlock:
    case Family::TrapConcurrentLaunch:
      return true;
    default:
      return false;
  }
}

const char* dist_name(CostDist d) {
  switch (d) {
    case CostDist::Uniform: return "uniform";
    case CostDist::HeavyTail: return "heavytail";
    case CostDist::Decorrelated: return "decorrelated";
  }
  return "?";
}

namespace {

struct Draw {  // per-tensor size / per-op cost draw under a distribution
  CostDist d;
  Rng& rng;
  int64_t size() {
    switch (d) {
      case CostDist::Uniform: return rng.uniform(20, 100);
      case CostDist::HeavyTail: return (int64_t)1 << rng.uniform(4, 8);  // 16..256
      case CostDist::Decorrelated: return rng.uniform(20, 100);
    }
    return 0;
  }
  int64_t cost(int64_t size_hint) {
    switch (d) {
      case CostDist::Uniform: return rng.uniform(1, 6);
      case CostDist::HeavyTail: return (int64_t)1 << rng.uniform(0, 3);  // 1..8
      case CostDist::Decorrelated:
        // Anti-correlated: big tensors come from cheap ops.
        return std::max<int64_t>(1, 8 - size_hint / 20);
    }
    return 1;
  }
};

// Organic family builders. `n` scales structure mildly with the seed so the
// corpus is not degenerate.

Generated chain_family(Draw& draw) {
  GraphBuilder b;
  const int n = 6 + (int)(draw.rng.next_u64() % 4);  // 6..9 ops
  TensorId prev = b.add_tensor(draw.size(), -1);
  b.add_op({}, {prev}, 0, draw.cost(40), /*replayable=*/false);
  for (int i = 1; i < n; ++i) {
    int64_t s = draw.size();
    TensorId t = b.add_tensor(s, -1, /*required=*/i == n - 1);
    b.add_op({prev}, {t}, 0, draw.cost(s), true);
    prev = t;
  }
  Generated out;
  out.graph = b.finish();
  out.bo = never_drop_peak(out.graph);
  return out;
}

Generated diamond_family(Draw& draw) {
  GraphBuilder b;
  const int k = 2 + (int)(draw.rng.next_u64() % 2);  // 2..3 diamonds
  TensorId prev = b.add_tensor(draw.size(), -1);
  b.add_op({}, {prev}, 0, draw.cost(40), false);
  for (int i = 0; i < k; ++i) {
    int64_t s1 = draw.size(), s2 = draw.size();
    TensorId a = b.add_tensor(s1, -1);
    TensorId c = b.add_tensor(s2, -1, /*required=*/i == k - 1);
    b.add_op({prev}, {a}, 0, draw.cost(s1), true);
    b.add_op({a}, {c}, 0, draw.cost(s2), true);
    prev = c;
  }
  Generated out;
  out.graph = b.finish();
  out.bo = never_drop_peak(out.graph);
  return out;
}

Generated shared_ancestor_family(Draw& draw) {
  // Two ancestors fan out to consumers; the shared tensors are long-lived and
  // recomputing them amortizes across consumers.
  GraphBuilder b;
  TensorId a1 = b.add_tensor(draw.size(), -1);
  TensorId a2 = b.add_tensor(draw.size(), -1);
  b.add_op({}, {a1}, 0, draw.cost(40), false);
  b.add_op({}, {a2}, 0, draw.cost(40), false);
  const int k = 3 + (int)(draw.rng.next_u64() % 3);  // 3..5 consumers
  for (int i = 0; i < k; ++i) {
    int64_t s = draw.size();
    TensorId c = b.add_tensor(s, -1, /*required=*/i == k - 1);
    b.add_op({a1, a2}, {c}, 0, draw.cost(s), true);
  }
  Generated out;
  out.graph = b.finish();
  out.bo = never_drop_peak(out.graph);
  return out;
}

Generated fork_join_family(Draw& draw) {
  GraphBuilder b;
  TensorId root = b.add_tensor(draw.size(), -1);
  b.add_op({}, {root}, 0, draw.cost(40), false);
  const int branches = 2 + (int)(draw.rng.next_u64() % 3);  // 2..4
  std::vector<TensorId> tips;
  for (int i = 0; i < branches; ++i) {
    int64_t s1 = draw.size(), s2 = draw.size();
    TensorId m = b.add_tensor(s1, -1);
    TensorId t = b.add_tensor(s2, -1);
    b.add_op({root}, {m}, 0, draw.cost(s1), true);
    b.add_op({m}, {t}, 0, draw.cost(s2), true);
    tips.push_back(t);
  }
  int64_t sj = draw.size();
  TensorId join = b.add_tensor(sj, -1, /*required=*/true);
  b.add_op(tips, {join}, 0, draw.cost(sj), true);
  Generated out;
  out.graph = b.finish();
  out.bo = never_drop_peak(out.graph);
  return out;
}

Generated residual_family(Draw& draw) {
  // Residual blocks: block input is carried (skipped) to the block merge,
  // creating long-lived tensors with late last uses.
  GraphBuilder b;
  const int blocks = 2 + (int)(draw.rng.next_u64() % 2);  // 2..3
  TensorId carry = b.add_tensor(draw.size(), -1);
  b.add_op({}, {carry}, 0, draw.cost(40), false);
  for (int i = 0; i < blocks; ++i) {
    int64_t s1 = draw.size(), sm = draw.size();
    TensorId main_t = b.add_tensor(s1, -1);
    b.add_op({carry}, {main_t}, 0, draw.cost(s1), true);
    TensorId merged = b.add_tensor(sm, -1, /*required=*/i == blocks - 1);
    b.add_op({carry, main_t}, {merged}, 0, draw.cost(sm), true);
    carry = merged;
  }
  Generated out;
  out.graph = b.finish();
  out.bo = never_drop_peak(out.graph);
  return out;
}

Generated big_workspace_family(Draw& draw) {
  // A few ops carry huge scratch reservations; frontier release decisions must
  // reason about scratch peaks, not just tensor bytes.
  GraphBuilder b;
  TensorId prev = b.add_tensor(draw.size(), -1);
  b.add_op({}, {prev}, 0, draw.cost(40), false);
  const int n = 4 + (int)(draw.rng.next_u64() % 3);
  for (int i = 0; i < n; ++i) {
    int64_t s = draw.size();
    int64_t scratch = (i % 2 == 0) ? 3 * s : 0;  // every other op: 3x scratch
    TensorId t = b.add_tensor(s, -1, /*required=*/i == n - 1);
    b.add_op({prev}, {t}, scratch, draw.cost(s), true);
    prev = t;
  }
  Generated out;
  out.graph = b.finish();
  out.bo = never_drop_peak(out.graph);
  return out;
}

Generated heterogeneous_family(Draw& draw) {
  // Random DAG: op v_i reads 1..2 earlier tensors (chosen among the last few
  // produced), keeps widths small so the oracle remains exhaustive.
  GraphBuilder b;
  const int n = 6 + (int)(draw.rng.next_u64() % 4);  // 6..9 ops
  std::vector<TensorId> pool;
  TensorId t0 = b.add_tensor(draw.size(), -1);
  b.add_op({}, {t0}, 0, draw.cost(40), false);
  pool.push_back(t0);
  for (int i = 1; i < n; ++i) {
    std::vector<TensorId> ins;
    int reads = 1 + (int)(draw.rng.next_u64() % 2);
    for (int j = 0; j < reads; ++j) {
      size_t idx = (size_t)(draw.rng.next_u64() % pool.size());
      ins.push_back(pool[pool.size() - 1 - (idx % std::min<size_t>(pool.size(), 4))]);
    }
    // dedupe
    std::sort(ins.begin(), ins.end());
    ins.erase(std::unique(ins.begin(), ins.end()), ins.end());
    int64_t s = draw.size();
    TensorId t = b.add_tensor(s, -1, /*required=*/i == n - 1);
    b.add_op(ins, {t}, draw.rng.chance(30) ? s / 2 : 0, draw.cost(s), true);
    pool.push_back(t);
  }
  Generated out;
  out.graph = b.finish();
  out.bo = never_drop_peak(out.graph);
  return out;
}

// --- Declared adversarial traps (fixed structure + sizes, Bo with headroom).

// Greedy largest-first overshoots from x into s (target 55 exceeds x alone);
// with s surrendered nothing re-derives. The oracle instead releases {x,y} and
// re-derives the whole chain from s through replayable v1/v2.
Generated trap_recompute() {
  GraphBuilder b;
  TensorId s = b.add_tensor(25, -1);
  b.add_op({}, {s}, 0, 1, false);                    // v0 source -> s(25)
  TensorId x = b.add_tensor(50, -1);
  b.add_op({s}, {x}, 0, 2, true);                    // v1: s -> x(50)
  TensorId y = b.add_tensor(20, -1);
  b.add_op({x}, {y}, 0, 2, true);                    // v2: x -> y(20)
  TensorId z = b.add_tensor(20, -1, /*required=*/true);
  b.add_op({x, y}, {z}, 0, 2, true);                 // v3: (x,y) -> z(20)
  Generated out;
  out.graph = b.finish();
  out.bo = 160;  // never-drop peak 115; headroom so the re-derivation chain
                 // fits the post-release region (ceiling 90 at target 55)
  return out;
}

// The source output s(80) is the only way to re-produce a and b; releasing it
// (greedy largest-first reaches the target with s alone) kills v3, which also
// reads s directly. Keeping s and releasing a+b (70) reaches the target with
// recompute under the post-release region.
Generated trap_nonreplayable() {
  GraphBuilder b;
  TensorId s = b.add_tensor(80, -1);
  b.add_op({}, {s}, 0, 1, false);                    // v0 source -> s(80)
  TensorId a = b.add_tensor(30, -1);
  b.add_op({s}, {a}, 0, 2, true);                    // v1: s -> a(30)
  TensorId bb = b.add_tensor(40, -1);
  b.add_op({s}, {bb}, 0, 2, true);                   // v2: s -> b(40)
  TensorId z = b.add_tensor(10, -1, /*required=*/true);
  b.add_op({s, a, bb}, {z}, 0, 2, true);             // v3: (s,a,b) -> z(10)
  Generated out;
  out.graph = b.finish();
  out.bo = 240;  // peak 160; headroom so v1/v2 re-runs and the v3 run (160)
                 // fit the post-release region (170)
  return out;
}

// Greedy largest-first pairs big with c1 (target 150 exceeds big alone); with
// c1 surrendered, nothing re-derives it (big is gone too). The viable plans
// release the dead side branch (+c1) and re-run v1, or keep c1 entirely.
Generated trap_memory() {
  GraphBuilder b;
  TensorId big = b.add_tensor(100, -1);
  b.add_op({}, {big}, 0, 1, false);                  // v0 source -> big(100)
  TensorId c1 = b.add_tensor(60, -1);
  b.add_op({big}, {c1}, 0, 2, true);                 // v1: big -> c1(60)
  TensorId c2 = b.add_tensor(70, -1, /*required=*/true);
  b.add_op({c1}, {c2}, 0, 2, true);                  // v2: c1 -> c2(70)
  TensorId d1 = b.add_tensor(40, -1);
  b.add_op({}, {d1}, 0, 1, false);                   // v3 source -> d1(40)
  TensorId d2 = b.add_tensor(50, -1);
  b.add_op({d1}, {d2}, 0, 1, true);                  // v4: d1 -> d2(50)
  Generated out;
  out.graph = b.finish();
  out.bo = 320;  // never-drop peak; v1 re-run (160) fits ceiling 320-150=170
  return out;
}

// Drop-only policies can never satisfy the handover (the only releasable
// block, x, is unreproducible and y is required). One prep RUN of v1 converts
// x -> y, after which releasing x is safe. K >= cost(v1) unlocks the request.
Generated prep_unlock() {
  GraphBuilder b;
  TensorId x = b.add_tensor(80, -1);
  b.add_op({}, {x}, 0, 2, false);                    // v0 source -> x(80)
  TensorId y = b.add_tensor(80, -1, /*required=*/true);
  b.add_op({x}, {y}, 0, 3, true);                    // v1: x -> y(80)
  TensorId z = b.add_tensor(20, -1, /*required=*/true);
  b.add_op({y}, {z}, 0, 2, true);                    // v2: y -> z(20)
  Generated out;
  out.graph = b.finish();
  out.bo = 200;  // peak 180; headroom for the post-release v2 run
  return out;
}

// Audit finding 6: concurrent-launch oversubscription, sized for the frozen
// grid. At the 0.8 bucket (Bo=5, Bn=4) the release target is exactly the
// 1-unit init block; the post-release region is 4 units while each output is
// 3, so the concurrent trace (LAUNCH A, LAUNCH B, ...) is illegal at the
// second LAUNCH (3 + 3 > 4) — precisely the v1.0 defect — and the only
// v1.1-legal plan serializes: LAUNCH A, WAIT A, DROP the disposable output,
// LAUNCH B, WAIT B. Under the v1.0 model this instance accepted the
// concurrent trace; it is a permanent semantic regression pin.
Generated trap_concurrent_launch() {
  GraphBuilder b;
  TensorId init = b.add_tensor(1, -1);
  b.add_op({}, {init}, 0, 1, false);                 // v0 source -> init(1)
  TensorId x = b.add_tensor(3, -1);
  b.add_op({}, {x}, 0, 1, true);                     // A -> x(3), disposable
  TensorId y = b.add_tensor(3, -1, /*required=*/true);
  b.add_op({}, {y}, 0, 1, true);                     // B -> y(3), required
  Generated out;
  out.graph = b.finish();
  out.bo = 5;    // 0.8 bucket: Bn=4, target 1; post-release region 4 < 3+3
  return out;
}

}  // namespace

Generated generate(Family f, CostDist d, uint64_t seed) {
  Rng rng(seed * 0x2545F4914F6CDD1Dull + 0x9E3779B9ull);
  Draw draw{d, rng};
  switch (f) {
    case Family::Chain: return chain_family(draw);
    case Family::Diamond: return diamond_family(draw);
    case Family::SharedAncestor: return shared_ancestor_family(draw);
    case Family::ForkJoin: return fork_join_family(draw);
    case Family::Residual: return residual_family(draw);
    case Family::BigWorkspace: return big_workspace_family(draw);
    case Family::Heterogeneous: return heterogeneous_family(draw);
    case Family::TrapRecompute: return trap_recompute();
    case Family::TrapNonreplayable: return trap_nonreplayable();
    case Family::TrapMemory: return trap_memory();
    case Family::PrepUnlock: return prep_unlock();
    case Family::TrapConcurrentLaunch: return trap_concurrent_launch();
  }
  throw std::invalid_argument("unknown family");
}

int64_t never_drop_peak(const Graph& g) {
  // Checked arithmetic (audit finding 3): generator-side accounting must not
  // overflow silently either.
  int64_t usage = 0, peak = 0;
  for (OpId v : g.topo) {
    int64_t during = usage;
    for (TensorId t : g.ops[v].outputs)
      if (__builtin_add_overflow(during, g.tensors[t].size, &during))
        throw std::overflow_error("never-drop peak overflows");
    if (__builtin_add_overflow(during, g.ops[v].scratch, &during))
      throw std::overflow_error("never-drop peak overflows");
    peak = std::max(peak, during);
    for (TensorId t : g.ops[v].outputs)
      if (__builtin_add_overflow(usage, g.tensors[t].size, &usage))
        throw std::overflow_error("never-drop peak overflows");
  }
  return peak;
}

State sample_frontier(const Graph& g, int64_t bo, Rng& rng, int min_steps,
                      int max_steps, bool async, int max_inflight) {
  Request exec{bo, bo, (int64_t)1 << 40};  // pre-request budget: K unlimited
  State s;
  int steps = (int)rng.uniform(min_steps, max_steps);
  for (int i = 0; i < steps; ++i) {
    std::vector<OpId> ready;
    for (OpId v = 0; v < (OpId)g.ops.size(); ++v)
      if (can_run(g, s, exec, v).ok) ready.push_back(v);
    if (ready.empty()) break;
    OpId v = ready[(size_t)rng.uniform(0, (int64_t)ready.size() - 1)];
    s = apply_run(g, s, exec, v);
  }
  if (async) {
    int launches = (int)rng.uniform(1, max_inflight);
    for (int i = 0; i < launches; ++i) {
      std::vector<OpId> ready;
      for (OpId v = 0; v < (OpId)g.ops.size(); ++v)
        if (can_launch(g, s, exec, v).ok) ready.push_back(v);
      if (ready.empty()) break;
      OpId v = ready[(size_t)rng.uniform(0, (int64_t)ready.size() - 1)];
      s = apply_launch(g, s, exec, v);
    }
  }
  return capture_frontier(s);
}

int64_t remaining_work(const Graph& g, const State& s) {
  int64_t sum = 0;
  for (OpId v = 0; v < (OpId)g.ops.size(); ++v)
    if (!(s.done & (OpMask(1) << v)) && !(s.inflight & (OpMask(1) << v)))
      if (__builtin_add_overflow(sum, g.ops[v].cost, &sum))
        throw std::overflow_error("remaining work overflows");
  return sum;
}

}  // namespace keel
