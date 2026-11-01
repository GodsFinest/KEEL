#include <keel/rt/bench_gen.hpp>

#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace keel::rt {

const char* bench_family_name(BenchFamily f) {
  switch (f) {
    case BenchFamily::Chain: return "chain";
    case BenchFamily::Diamond: return "diamond";
    case BenchFamily::FanoutFanin: return "fanout_fanin";
    case BenchFamily::SharedAncestor: return "shared_ancestor";
    case BenchFamily::Residual: return "residual";
    case BenchFamily::WideIndependent: return "wide_independent";
    case BenchFamily::MatmulHeavy: return "matmul_heavy";
    case BenchFamily::ScratchHeavy: return "scratch_heavy";
    case BenchFamily::TransformerLike: return "transformer_like";
  }
  return "?";
}

int bench_family_count() { return 9; }

namespace {

// Deterministic per-call RNG for shape/cost variation.
struct R {
  uint64_t s;
  explicit R(uint64_t seed) : s(seed * 6364136223846793005ull + 1) {}
  int64_t next(int64_t lo, int64_t hi) {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    return lo + (int64_t)((s >> 33) % (uint64_t)(hi - lo + 1));
  }
};

int64_t varied_dim(R& rng) { return 8 + 4 * rng.next(0, 3); }

TensorId add_source(RuntimeGraph& g, int64_t rows, int64_t cols) {
  const TensorId t = g.add_tensor({DType::F64, {rows, cols}});
  g.add_op({OpKind::Source, {}, {t}, 0, 1, false});
  return t;
}

TensorId add_unary(RuntimeGraph& g, OpKind kind, TensorId input,
                   bool required, int64_t cost) {
  const auto shape = g.tensors()[input].shape;
  const TensorId t = g.add_tensor({DType::F64, shape, required});
  g.add_op({kind, {input}, {t}, 0, cost, true});
  return t;
}

TensorId add_binary(RuntimeGraph& g, OpKind kind, TensorId a, TensorId b,
                    bool required, int64_t cost) {
  const auto shape = g.tensors()[a].shape;
  const TensorId t = g.add_tensor({DType::F64, shape, required});
  g.add_op({kind, {a, b}, {t}, 0, cost, true});
  return t;
}

TensorId add_matmul(RuntimeGraph& g, OpKind kind, TensorId a, TensorId b,
                    int64_t scratch, bool required, int64_t cost) {
  const auto& as = g.tensors()[a].shape;
  const auto& bs = g.tensors()[b].shape;
  const TensorId t = g.add_tensor({DType::F64, {as[0], bs[1]}, required});
  g.add_op({kind, {a, b}, {t}, scratch, cost, true});
  return t;
}

TensorId add_reduce(RuntimeGraph& g, TensorId input, bool required,
                    int64_t cost) {
  const auto& shape = g.tensors()[input].shape;
  const TensorId t = g.add_tensor({DType::F64, {shape[0], 1}, required});
  g.add_op({OpKind::ReduceSum, {input}, {t}, 0, cost, true});
  return t;
}

RuntimeGraph varied_chain(uint64_t seed) {
  RuntimeGraph g;
  R rng(seed);
  // Keep the generated research graphs inside the exact-planner envelope.
  // Variation is still present in depth, shape, operator kind, and cost, but
  // the final evaluator must finish rather than silently turning a long exact
  // search into an unrecorded timeout.
  const int depth = 4 + static_cast<int>(rng.next(0, 4));
  const int64_t d = varied_dim(rng);
  TensorId cur = add_source(g, d, d);
  for (int i = 1; i < depth; ++i)
    cur = add_unary(g, (rng.next(0, 1) == 0) ? OpKind::Add : OpKind::Mul,
                    cur, i + 1 == depth, rng.next(1, 6));
  return g;
}

RuntimeGraph varied_diamond(uint64_t seed) {
  RuntimeGraph g;
  R rng(seed);
  const int64_t d = varied_dim(rng);
  const TensorId s0 = add_source(g, d, d);
  const TensorId s1 = add_source(g, d, d);
  const int branches = 2 + static_cast<int>(rng.next(0, 1));
  std::vector<TensorId> leaves;
  leaves.reserve(static_cast<size_t>(branches));
  for (int i = 0; i < branches; ++i) {
    TensorId cur = (i & 1) ? s1 : s0;
    const int length = 1 + static_cast<int>(rng.next(0, 1));
    for (int j = 0; j < length; ++j)
      cur = add_unary(g, (j & 1) ? OpKind::Mul : OpKind::Add, cur, false,
                      rng.next(1, 5));
    leaves.push_back(cur);
  }
  while (leaves.size() > 1) {
    std::vector<TensorId> next;
    for (size_t i = 0; i < leaves.size(); i += 2) {
      if (i + 1 == leaves.size()) next.push_back(leaves[i]);
      else next.push_back(add_binary(g, OpKind::Add, leaves[i], leaves[i + 1],
                                     false, rng.next(1, 5)));
    }
    leaves = std::move(next);
  }
  add_unary(g, OpKind::Add, leaves.front(), true, rng.next(1, 5));
  return g;
}

RuntimeGraph varied_fanout_fanin(uint64_t seed) {
  RuntimeGraph g;
  R rng(seed);
  const int64_t d = varied_dim(rng);
  const TensorId base = add_source(g, d, d);
  const TensorId side = add_source(g, d, d);
  const int branches = 2 + static_cast<int>(rng.next(0, 1));
  std::vector<TensorId> leaves;
  for (int i = 0; i < branches; ++i) {
    TensorId cur = add_unary(g, (i & 1) ? OpKind::Mul : OpKind::Add, base,
                             false, rng.next(1, 5));
    if (i == 0 && rng.next(0, 1))
      cur = add_unary(g, OpKind::Add, cur, false, rng.next(1, 4));
    leaves.push_back(cur);
  }
  TensorId joined = add_binary(g, OpKind::Add, leaves[0], side, false, 2);
  for (size_t i = 1; i < leaves.size(); ++i)
    joined = add_binary(g, (i & 1) ? OpKind::Add : OpKind::Mul, joined,
                        leaves[i], i + 1 == leaves.size(), rng.next(1, 5));
  (void)joined;
  return g;
}

RuntimeGraph varied_shared_ancestor(uint64_t seed) {
  RuntimeGraph g;
  R rng(seed);
  const int64_t d = varied_dim(rng);
  const TensorId s0 = add_source(g, d, d);
  const TensorId s1 = add_source(g, d, d);
  const TensorId ancestor = add_binary(g, OpKind::Mul, s0, s1, false, 2);
  const int consumers = 2 + static_cast<int>(rng.next(0, 3));
  TensorId prev = s1;
  for (int i = 0; i < consumers; ++i)
    prev = add_binary(g, OpKind::Add, ancestor, prev, i + 1 == consumers,
                      rng.next(1, 5));
  return g;
}

RuntimeGraph varied_residual(uint64_t seed) {
  RuntimeGraph g;
  R rng(seed);
  const int64_t d = varied_dim(rng);
  TensorId x = add_source(g, d, d);
  const TensorId w = add_source(g, d, d);
  const int depth = 2 + static_cast<int>(rng.next(0, 2));
  for (int i = 0; i < depth; ++i) {
    const int64_t min_scratch = d * d * 8;
    const TensorId branch = add_matmul(
        g, OpKind::ScratchGemm, x, w,
        min_scratch * (1 + rng.next(0, 2)), false, rng.next(3, 7));
    x = add_binary(g, OpKind::ResidualAdd, x, branch, i + 1 == depth,
                   rng.next(1, 4));
  }
  return g;
}

RuntimeGraph varied_wide(uint64_t seed) {
  RuntimeGraph g;
  R rng(seed);
  const int64_t d = varied_dim(rng);
  const TensorId base = add_source(g, d, d);
  const int branches = 2 + static_cast<int>(rng.next(0, 2));
  for (int i = 0; i < branches; ++i) {
    TensorId cur = base;
    const int depth = 1 + static_cast<int>(rng.next(0, 1));
    for (int j = 0; j < depth; ++j)
      cur = add_unary(g, (j & 1) ? OpKind::Mul : OpKind::Add, cur,
                      j + 1 == depth, rng.next(1, 5));
  }
  return g;
}

RuntimeGraph varied_matmul(uint64_t seed) {
  RuntimeGraph g;
  R rng(seed);
  const int64_t d = varied_dim(rng);
  TensorId cur = add_source(g, d, d);
  const int depth = 2 + static_cast<int>(rng.next(0, 1));
  for (int i = 0; i < depth; ++i) {
    const TensorId weight = add_source(g, d, d);
    cur = add_matmul(g, (i & 1) ? OpKind::MatMul : OpKind::ScratchGemm,
                     cur, weight, d * d * 8, i + 1 == depth, rng.next(4, 8));
  }
  return g;
}

RuntimeGraph varied_scratch(uint64_t seed) {
  RuntimeGraph g;
  R rng(seed);
  const int64_t d = varied_dim(rng);
  const TensorId b = add_source(g, d, d);
  TensorId cur = add_source(g, d, d);
  const int depth = 2 + static_cast<int>(rng.next(0, 1));
  for (int i = 0; i < depth; ++i) {
    const int64_t min_scratch = d * d * 8;
    cur = add_matmul(g, OpKind::ScratchGemm, cur, b,
                     min_scratch * (2 + rng.next(0, 3)), i + 1 == depth,
                     rng.next(4, 8));
    if (i + 1 != depth)
      cur = add_unary(g, OpKind::Add, cur, false, rng.next(1, 3));
  }
  return g;
}

RuntimeGraph varied_transformer(uint64_t seed) {
  RuntimeGraph g;
  R rng(seed);
  const int64_t d = varied_dim(rng);
  TensorId src = add_source(g, d, d);
  TensorId value = add_source(g, d, d);
  // A single varied layer retains the transformer-like dependency pattern
  // without making exact continuation search exponential in layer count.
  const int layers = 1;
  for (int layer = 0; layer < layers; ++layer) {
    const TensorId wq = add_source(g, d, d);
    const TensorId wk = add_source(g, d, d);
    const TensorId q = add_matmul(g, OpKind::MatMul, src, wq, 0, false, 4);
    const TensorId k = add_matmul(g, OpKind::MatMul, src, wk, 0, false, 4);
    const TensorId score = add_binary(g, OpKind::Mul, q, k, false, 2);
    const TensorId scale = add_reduce(g, score, false, 3);
    const TensorId ctx = add_binary(g, OpKind::Add, value, scale, false, 2);
    const TensorId mix = add_binary(g, OpKind::Add, ctx, scale, false, 2);
    const TensorId out = add_matmul(g, OpKind::ScratchGemm, ctx, mix,
                                    d * d * 8, layer + 1 == layers, 5);
    src = out;
    value = out;
  }
  return g;
}

RuntimeGraph chain(uint64_t seed) {
  RuntimeGraph g;
  R rng(seed);
  const int n = 8;
  TensorId cur = g.add_tensor({DType::F64, {64, 48}});
  g.add_op({OpKind::Source, {}, {cur}, 0, 1, false});
  for (int i = 1; i < n; ++i) {
    TensorId t =
        g.add_tensor({DType::F64, {64, 48}, /*required=*/i == n - 1});
    g.add_op({i % 2 ? OpKind::Mul : OpKind::Add, {cur}, {t}, 0, rng.next(1, 4),
              true});
    cur = t;
  }
  return g;
}

RuntimeGraph diamond(uint64_t seed) {
  (void)seed;
  RuntimeGraph g;
  TensorId s0 = g.add_tensor({DType::F64, {48, 32}});
  TensorId s1 = g.add_tensor({DType::F64, {48, 32}});
  g.add_op({OpKind::Source, {}, {s0}, 0, 1, false});
  g.add_op({OpKind::Source, {}, {s1}, 0, 1, false});
  TensorId m = g.add_tensor({DType::F64, {48, 32}});
  g.add_op({OpKind::Mul, {s0, s1}, {m}, 0, 2, true});
  TensorId a = g.add_tensor({DType::F64, {48, 32}});
  g.add_op({OpKind::Add, {s0, s1}, {a}, 0, 2, true});
  TensorId j = g.add_tensor({DType::F64, {48, 32}});
  g.add_op({OpKind::Add, {m, a}, {j}, 0, 2, true});
  TensorId r = g.add_tensor({DType::F64, {48, 1}, /*required=*/true});
  g.add_op({OpKind::ReduceSum, {j}, {r}, 0, 3, true});
  return g;
}

RuntimeGraph fanout_fanin(uint64_t seed) {
  (void)seed;
  RuntimeGraph g;
  TensorId s0 = g.add_tensor({DType::F64, {24, 40}});
  TensorId s1 = g.add_tensor({DType::F64, {40, 24}});
  g.add_op({OpKind::Source, {}, {s0}, 0, 1, false});
  g.add_op({OpKind::Source, {}, {s1}, 0, 1, false});
  TensorId m1 = g.add_tensor({DType::F64, {24, 24}});
  g.add_op({OpKind::MatMul, {s0, s1}, {m1}, 0, 4, true});
  TensorId m2 = g.add_tensor({DType::F64, {24, 24}});
  g.add_op({OpKind::ScratchGemm, {s0, s1}, {m2}, 40 * 24 * 8, 4, true});
  TensorId j = g.add_tensor({DType::F64, {24, 24}, /*required=*/true});
  g.add_op({OpKind::Add, {m1, m2}, {j}, 0, 2, true});
  return g;
}

RuntimeGraph shared_ancestor(uint64_t seed) {
  (void)seed;
  RuntimeGraph g;
  TensorId s0 = g.add_tensor({DType::F64, {40, 40}});
  TensorId s1 = g.add_tensor({DType::F64, {40, 40}});
  g.add_op({OpKind::Source, {}, {s0}, 0, 1, false});
  g.add_op({OpKind::Source, {}, {s1}, 0, 1, false});
  TensorId anc = g.add_tensor({DType::F64, {40, 40}});
  g.add_op({OpKind::Mul, {s0, s1}, {anc}, 0, 2, true});
  TensorId prev = s1;  // first consumer pairs the ancestor with s1
  for (int i = 0; i < 3; ++i) {
    TensorId c = g.add_tensor({DType::F64, {40, 40}, /*required=*/i == 2});
    g.add_op({OpKind::Add, {anc, prev}, {c}, 0, i + 1, true});
    prev = c;
  }
  return g;
}

RuntimeGraph residual(uint64_t seed) {
  (void)seed;
  RuntimeGraph g;
  TensorId x = g.add_tensor({DType::F64, {32, 32}});
  g.add_op({OpKind::Source, {}, {x}, 0, 1, false});
  TensorId w = g.add_tensor({DType::F64, {32, 32}});
  g.add_op({OpKind::Add, {x}, {w}, 0, 1, true});  // w = 2x (unary add)
  const int blocks = 3;
  for (int i = 0; i < blocks; ++i) {
    TensorId l = g.add_tensor({DType::F64, {32, 32}});
    g.add_op({OpKind::ScratchGemm, {x, w}, {l}, 32 * 32 * 8, 5, true});
    TensorId y =
        g.add_tensor({DType::F64, {32, 32}, /*required=*/i == blocks - 1});
    g.add_op({OpKind::ResidualAdd, {x, l}, {y}, 0, 2, true});
    x = y;
  }
  return g;
}

RuntimeGraph matmul_heavy(uint64_t seed) {
  (void)seed;
  RuntimeGraph g;
  TensorId s0 = g.add_tensor({DType::F64, {24, 24}});
  TensorId s1 = g.add_tensor({DType::F64, {24, 24}});
  g.add_op({OpKind::Source, {}, {s0}, 0, 1, false});
  g.add_op({OpKind::Source, {}, {s1}, 0, 1, false});
  TensorId cur = s0;
  for (int i = 0; i < 3; ++i) {
    TensorId t = g.add_tensor({DType::F64, {24, 24}, /*required=*/i == 2});
    g.add_op({i % 2 ? OpKind::MatMul : OpKind::ScratchGemm, {cur, s1}, {t},
              24 * 24 * 8, 6, true});
    cur = t;
  }
  return g;
}

RuntimeGraph scratch_heavy(uint64_t seed) {
  (void)seed;
  RuntimeGraph g;
  TensorId s0 = g.add_tensor({DType::F64, {16, 48}});
  TensorId s1 = g.add_tensor({DType::F64, {48, 16}});
  g.add_op({OpKind::Source, {}, {s0}, 0, 1, false});
  g.add_op({OpKind::Source, {}, {s1}, 0, 1, false});
  TensorId cur = g.add_tensor({DType::F64, {16, 16}});
  g.add_op({OpKind::ScratchGemm, {s0, s1}, {cur}, 64 * 1024, 4, true});
  // Derived second operand so every op has distinct inputs; refresh it after
  // each step (w tracks a linear function of cur).
  TensorId w = g.add_tensor({DType::F64, {16, 16}});
  g.add_op({OpKind::Add, {cur}, {w}, 0, 1, true});  // w = 2*cur (unary)
  for (int i = 0; i < 2; ++i) {
    TensorId t = g.add_tensor({DType::F64, {16, 16}, /*required=*/i == 1});
    g.add_op({OpKind::ScratchGemm, {cur, w}, {t}, 96 * 1024, 5, true});
    TensorId w2 = g.add_tensor({DType::F64, {16, 16}});
    g.add_op({OpKind::Add, {t, w}, {w2}, 0, 1, true});
    cur = t;
    w = w2;
  }
  return g;
}

RuntimeGraph transformer_like(uint64_t seed) {
  (void)seed;
  RuntimeGraph g;
  // Attention-like block from supported primitives:
  //   q = src x wq, k = src x wk; sk = q*k; sc = rowsum(sk);
  //   ctx = v + sc (broadcast); w2 = ctx + sc; out = ctx x w2 (required).
  TensorId src = g.add_tensor({DType::F64, {16, 24}});
  g.add_op({OpKind::Source, {}, {src}, 0, 1, false});
  TensorId wq = g.add_tensor({DType::F64, {24, 16}});
  TensorId wk = g.add_tensor({DType::F64, {24, 16}});
  g.add_op({OpKind::Source, {}, {wq}, 0, 1, false});
  g.add_op({OpKind::Source, {}, {wk}, 0, 1, false});
  TensorId q = g.add_tensor({DType::F64, {16, 16}});
  TensorId k = g.add_tensor({DType::F64, {16, 16}});
  g.add_op({OpKind::MatMul, {src, wq}, {q}, 0, 4, true});
  g.add_op({OpKind::MatMul, {src, wk}, {k}, 0, 4, true});
  TensorId sk = g.add_tensor({DType::F64, {16, 16}});
  g.add_op({OpKind::Mul, {q, k}, {sk}, 0, 2, true});
  TensorId sc = g.add_tensor({DType::F64, {16, 1}});
  g.add_op({OpKind::ReduceSum, {sk}, {sc}, 0, 3, true});
  TensorId v = g.add_tensor({DType::F64, {16, 16}});
  g.add_op({OpKind::Source, {}, {v}, 0, 1, false});
  TensorId ctx = g.add_tensor({DType::F64, {16, 16}});
  g.add_op({OpKind::Add, {sc, v}, {ctx}, 0, 2, true});
  TensorId w2 = g.add_tensor({DType::F64, {16, 16}});
  g.add_op({OpKind::Add, {ctx, sc}, {w2}, 0, 1, true});
  TensorId out = g.add_tensor({DType::F64, {16, 16}, /*required=*/true});
  g.add_op({OpKind::ScratchGemm, {ctx, w2}, {out}, 16 * 16 * 8, 5, true});
  return g;
}

}  // namespace

RuntimeGraph make_bench_graph(BenchFamily f, uint64_t seed) {
  RuntimeGraph g;
  switch (f) {
    case BenchFamily::Chain: g = varied_chain(seed); break;
    case BenchFamily::Diamond: g = varied_diamond(seed); break;
    case BenchFamily::FanoutFanin: g = varied_fanout_fanin(seed); break;
    case BenchFamily::SharedAncestor: g = varied_shared_ancestor(seed); break;
    case BenchFamily::Residual: g = varied_residual(seed); break;
    case BenchFamily::WideIndependent: g = varied_wide(seed); break;
    case BenchFamily::MatmulHeavy: g = varied_matmul(seed); break;
    case BenchFamily::ScratchHeavy: g = varied_scratch(seed); break;
    case BenchFamily::TransformerLike: g = varied_transformer(seed); break;
  }
  g.finish();  // validates: distinct inputs, single producers, shapes, DAG
  return g;
}

RuntimeGraph make_fixed_bench_graph(BenchFamily f, uint64_t seed) {
  RuntimeGraph g;
  switch (f) {
    case BenchFamily::Chain: g = chain(seed); break;
    case BenchFamily::Diamond: g = diamond(seed); break;
    case BenchFamily::FanoutFanin: g = fanout_fanin(seed); break;
    case BenchFamily::SharedAncestor: g = shared_ancestor(seed); break;
    case BenchFamily::Residual: g = residual(seed); break;
    case BenchFamily::WideIndependent: g = varied_wide(seed); break;
    case BenchFamily::MatmulHeavy: g = matmul_heavy(seed); break;
    case BenchFamily::ScratchHeavy: g = scratch_heavy(seed); break;
    case BenchFamily::TransformerLike: g = transformer_like(seed); break;
  }
  g.finish();
  return g;
}

std::string bench_graph_fingerprint(const RuntimeGraph& g) {
  uint64_t hash = 1469598103934665603ull;
  const auto mix = [&hash](uint64_t value) {
    for (int i = 0; i < 8; ++i) {
      hash ^= (value >> (i * 8)) & 0xffu;
      hash *= 1099511628211ull;
    }
  };
  mix(g.tensors().size());
  for (const TensorSpec& t : g.tensors()) {
    mix(static_cast<uint64_t>(t.dtype));
    mix(t.required ? 1 : 0);
    mix(t.pinned ? 1 : 0);
    mix(t.shape.size());
    for (int64_t d : t.shape) mix(static_cast<uint64_t>(d));
  }
  mix(g.ops().size());
  for (const OpSpec& op : g.ops()) {
    mix(static_cast<uint64_t>(op.kind));
    mix(op.scratch);
    mix(static_cast<uint64_t>(op.cost));
    mix(op.replayable ? 1 : 0);
    mix(op.inputs.size());
    for (TensorId t : op.inputs) mix(static_cast<uint64_t>(t));
    mix(op.outputs.size());
    for (TensorId t : op.outputs) mix(static_cast<uint64_t>(t));
  }
  std::ostringstream out;
  out << "fnv1a64-" << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}

std::vector<uint64_t> bench_source_seeds(const RuntimeGraph& g, uint64_t seed) {
  std::vector<uint64_t> seeds(g.ops().size(), 42);
  uint64_t s = seed;
  for (size_t v = 0; v < g.ops().size(); ++v) {
    if (g.ops()[v].kind == OpKind::Source) {
      s = s * 6364136223846793005ull + 1442695040888963407ull;
      seeds[v] = s;
    }
  }
  return seeds;
}

}  // namespace keel::rt
