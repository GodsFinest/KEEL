#include <doctest/doctest.h>

#include <keel/rt/bench_gen.hpp>
#include <keel/rt/executor.hpp>

#include <set>

using namespace keel::rt;
using keel::TensorId;

TEST_CASE("bench generator: seed reproducibility and structural diversity") {
  for (int fi = 0; fi < bench_family_count(); ++fi) {
    const BenchFamily family = static_cast<BenchFamily>(fi);
    std::set<std::string> fingerprints;
    for (uint64_t seed = 1; seed <= 8; ++seed) {
      RuntimeGraph first = make_bench_graph(family, seed);
      RuntimeGraph second = make_bench_graph(family, seed);
      const std::string first_id = bench_graph_fingerprint(first);
      CHECK(first_id == bench_graph_fingerprint(second));
      fingerprints.insert(first_id);

      const auto source_seeds = bench_source_seeds(first, seed * 1009 + 7);
      const auto values = reference_execute_f64(first, source_seeds);
      CHECK(values.size() == first.tensors().size());
      for (TensorId t = 0; t < static_cast<TensorId>(first.tensors().size());
           ++t)
        if (first.tensors()[t].required) CHECK_FALSE(values[t].empty());
    }
    CHECK(fingerprints.size() >= 2);
  }
}

TEST_CASE("bench generator: fixed workloads remain available as fixtures") {
  for (int fi = 0; fi < bench_family_count(); ++fi) {
    const BenchFamily family = static_cast<BenchFamily>(fi);
    const RuntimeGraph first = make_fixed_bench_graph(family, 1);
    const RuntimeGraph second = make_fixed_bench_graph(family, 1);
    CHECK(bench_graph_fingerprint(first) == bench_graph_fingerprint(second));
    CHECK(first.ops().size() <= 64);
    CHECK(first.tensors().size() <= 64);
  }
}
