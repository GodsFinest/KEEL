// Phase 7 — runtime workload generators for benchmarking (mandate Part VII
// shapes: chains, diamonds, fan-out/fan-in, shared-ancestor, residual,
// matrix-heavy, scratch-heavy, transformer-like). All graphs stay within the
// abstract-model caps (<= 64 ops/tensors) so the exact oracle remains usable
// as a reference instrument on every workload.

#pragma once

#include <keel/rt/ir.hpp>

#include <cstdint>
#include <string>

namespace keel::rt {

enum class BenchFamily {
  Chain,
  Diamond,
  FanoutFanin,
  SharedAncestor,
  Residual,
  WideIndependent,
  MatmulHeavy,
  ScratchHeavy,
  TransformerLike,
};

const char* bench_family_name(BenchFamily f);
int bench_family_count();

// Deterministic runtime graph for (family, seed). Shapes scale with depth so
// memory pressure is meaningful; all ops are supported primitives.
RuntimeGraph make_bench_graph(BenchFamily f, uint64_t seed);

// Original fixed-shape workloads retained for historical benchmark
// reproducibility. New research evaluation uses make_bench_graph().
RuntimeGraph make_fixed_bench_graph(BenchFamily f, uint64_t seed);

// Stable hash of topology and resource properties (not source values).
std::string bench_graph_fingerprint(const RuntimeGraph& g);

// Source seeds for a workload instance (one per source op, deterministic).
std::vector<uint64_t> bench_source_seeds(const RuntimeGraph& g, uint64_t seed);

}  // namespace keel::rt
