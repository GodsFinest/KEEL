// Phase 2: production IR tests (typed graphs, checked sizes, guarded
// conversion to the abstract model, op-shape consistency).

#include <doctest/doctest.h>

#include <keel/rt/ir.hpp>

#include <keel/oracle.hpp>
#include <keel/checker.hpp>

using namespace keel;
using namespace keel::rt;

namespace {

// A small but realistic mixed graph:
//   src1 (source) -> a [8x16 f64]
//   src2 (source) -> b [16x32 f64]
//   mm  = a x b   -> c [8x32]         (matmul, scratch 1024)
//   red = rowsum c -> d [8x1]
//   res = residual_add(d, a_broadcast?) -- keep shapes valid: add(d,c) -> e
RuntimeGraph make_mixed_graph() {
  RuntimeGraph g;
  TensorSpec a{DType::F64, {8, 16}};
  TensorSpec b{DType::F64, {16, 32}};
  TensorSpec c{DType::F64, {8, 32}};
  TensorSpec d{DType::F64, {8, 1}};
  TensorSpec e{DType::F64, {8, 32}, /*required=*/true};
  g.add_tensor(a);
  g.add_tensor(b);
  g.add_tensor(c);
  g.add_tensor(d);
  g.add_tensor(e);
  g.add_op({OpKind::Source, {}, {0}, 0, 1, false});
  g.add_op({OpKind::Source, {}, {1}, 0, 1, false});
  g.add_op({OpKind::MatMul, {0, 1}, {2}, 1024, 4, true});
  g.add_op({OpKind::ReduceSum, {2}, {3}, 0, 2, true});
  g.add_op({OpKind::Add, {3, 2}, {4}, 0, 1, true});  // broadcast-add row over c
  g.finish();
  return g;
}

}  // namespace

TEST_CASE("runtime IR: sizes, shapes, dtype checks") {
  RuntimeGraph g = make_mixed_graph();
  CHECK(g.tensor_bytes(0) == 8 * 16 * 8);
  CHECK(g.tensor_bytes(1) == 16 * 32 * 8);
  CHECK(g.tensor_bytes(2) == 8 * 32 * 8);
  CHECK(g.tensor_bytes(3) == 8 * 8);
  CHECK(g.topo().size() == 5);
  CHECK(g.producer_of(2) == 2);
  SUBCASE("dtype mismatch rejected") {
    RuntimeGraph bad;
    bad.add_tensor({DType::F64, {4, 4}});
    bad.add_tensor({DType::I64, {4, 4}});
    bad.add_tensor({DType::F64, {4, 4}});
    bad.add_op({OpKind::Source, {}, {0}, 0, 1, false});
    bad.add_op({OpKind::Source, {}, {1}, 0, 1, false});
    bad.add_op({OpKind::Add, {0, 1}, {2}, 0, 1, true});
    CHECK_THROWS(bad.finish());
  }
  SUBCASE("matmul inner-dim mismatch rejected") {
    RuntimeGraph bad;
    bad.add_tensor({DType::F64, {4, 5}});
    bad.add_tensor({DType::F64, {6, 7}});
    bad.add_tensor({DType::F64, {4, 7}});
    bad.add_op({OpKind::Source, {}, {0}, 0, 1, false});
    bad.add_op({OpKind::Source, {}, {1}, 0, 1, false});
    bad.add_op({OpKind::MatMul, {0, 1}, {2}, 0, 1, true});
    CHECK_THROWS(bad.finish());
  }
  SUBCASE("declared output size inconsistent with op rejected") {
    RuntimeGraph bad;
    bad.add_tensor({DType::F64, {4, 4}});
    bad.add_tensor({DType::F64, {4, 4}});
    bad.add_tensor({DType::F64, {5, 5}});  // wrong output size
    bad.add_op({OpKind::Source, {}, {0}, 0, 1, false});
    bad.add_op({OpKind::Source, {}, {1}, 0, 1, false});
    bad.add_op({OpKind::Add, {0, 1}, {2}, 0, 1, true});
    CHECK_THROWS(bad.finish());
  }
  SUBCASE("source ops must be inputless and non-replayable") {
    RuntimeGraph bad;
    bad.add_tensor({DType::F64, {4}});
    CHECK_THROWS(bad.add_op({OpKind::Source, {}, {0}, 0, 1, true}));
  }
  SUBCASE("zero/negative dims rejected") {
    RuntimeGraph bad;
    CHECK_THROWS(bad.add_tensor({DType::F64, {0, 4}}));
    CHECK_THROWS(bad.add_tensor({DType::F64, {-1}}));
  }
}

TEST_CASE("runtime IR -> abstract model conversion is exact") {
  RuntimeGraph g = make_mixed_graph();
  Graph abs = g.to_abstract();
  REQUIRE(abs.ops.size() == g.ops().size());
  REQUIRE(abs.tensors.size() == g.tensors().size());
  for (size_t t = 0; t < abs.tensors.size(); ++t) {
    CHECK(abs.tensors[t].size == g.tensor_bytes((TensorId)t));
    CHECK(abs.tensors[t].required == g.tensors()[t].required);
    CHECK(abs.tensors[t].pinned == g.tensors()[t].pinned);
    // Producer identity survives: abstract producer of abstract tensor t must
    // be the abstract op corresponding to the runtime producer.
    if (g.producer_of((TensorId)t) >= 0) {
      OpId abs_producer = abs.tensors[t].producer;
      CHECK(abs_producer == g.producer_of((TensorId)t));
    }
  }
  for (size_t v = 0; v < abs.ops.size(); ++v) {
    CHECK(abs.ops[v].scratch == g.ops()[v].scratch);
    CHECK(abs.ops[v].cost == g.ops()[v].cost);
    CHECK(abs.ops[v].replayable == g.ops()[v].replayable);
    CHECK(abs.ops[v].inputs == g.ops()[v].inputs);
    CHECK(abs.ops[v].outputs == g.ops()[v].outputs);
  }
  // The converted graph is usable by the exact oracle end-to-end. Budgets
  // are deliberately instance-sized: the exact oracle is a small-instance
  // instrument, and an unconstrained request (huge Bo/K) legitimately
  // explodes its state space to UNKNOWN.
  Request r{.Bo = 10240, .Bn = 8192, .K = 8};
  State s0 = capture_frontier(apply_run(abs, State{}, r, 0));
  OracleResult res = solve_transition(abs, s0, r, OracleConfig{});
  CHECK(res.status == OracleStatus::Feasible);
  REQUIRE(res.status == OracleStatus::Feasible);
  keel::CheckResult c = keel::check_witness(abs, s0, r, res.witness);
  CHECK(c.ok);
  CHECK(c.reached_goal);
}

TEST_CASE("runtime IR projection preserves non-identity tensor ids") {
  RuntimeGraph g;
  g.add_tensor({DType::F64, {4}});             // runtime t0, op 1
  g.add_tensor({DType::F64, {4}, true});      // runtime t1, op 0
  g.add_tensor({DType::F64, {4}});             // runtime t2, op 2
  g.add_op({OpKind::Source, {}, {1}, 0, 1, false});
  g.add_op({OpKind::Source, {}, {0}, 0, 1, false});
  g.add_op({OpKind::Source, {}, {2}, 0, 1, false});
  g.finish();

  RuntimeGraph::AbstractProjection projection = g.to_abstract_projection();
  REQUIRE(projection.runtime_to_abstract_tensor.size() == 3);
  CHECK(projection.runtime_to_abstract_tensor == std::vector<TensorId>{1, 0, 2});
  CHECK(projection.abstract_to_runtime_tensor == std::vector<TensorId>{1, 0, 2});
  CHECK(projection.runtime_to_abstract_op == std::vector<OpId>{0, 1, 2});
  CHECK(projection.abstract_to_runtime_op == std::vector<OpId>{0, 1, 2});
  CHECK(projection.graph.tensors[0].required);
  CHECK(projection.graph.tensors[0].producer == 0);
  CHECK(projection.graph.tensors[1].producer == 1);
  CHECK(projection.graph.tensors[2].producer == 2);
}

TEST_CASE("runtime IR: cap enforcement keeps abstract conversion exact") {
  // >64 tensors must fail loudly (abstract cap), not truncate.
  RuntimeGraph big;
  for (int i = 0; i < 64; ++i) big.add_tensor({DType::F64, {1}});
  CHECK_THROWS(big.add_tensor({DType::F64, {1}}));
  // A maximal 64-op / 64-tensor graph is representable and converts.
  RuntimeGraph full;
  full.add_tensor({DType::F64, {1}, /*required=*/true});
  full.add_op({OpKind::Source, {}, {0}, 0, 1, false});
  for (int i = 1; i < 64; ++i) {
    full.add_tensor({DType::F64, {1}});
    full.add_op({OpKind::Add, {i - 1}, {i}, 0, 1, true});
  }
  full.finish();
  Graph abs = full.to_abstract();
  CHECK(abs.ops.size() == 64);
  CHECK(abs.tensors.size() == 64);
  // 65 ops are structurally unreachable (each op needs a distinct output
  // tensor, so 65 ops would need 66 tensors > the tensor cap): the caps are
  // mutually consistent, and the tensor cap fires first.
}

TEST_CASE("runtime IR: dtype, arity, broadcast, scratch, and size checks are exact") {
  SUBCASE("non-F64 storage is rejected before execution") {
    for (DType dtype : {DType::F32, DType::I64}) {
      RuntimeGraph g;
      g.add_tensor({dtype, {4}});
      g.add_op({OpKind::Source, {}, {0}, 0, 1, false});
      CHECK_THROWS(g.finish());
    }
  }
  SUBCASE("incompatible broadcast and wrong declared shape are rejected") {
    RuntimeGraph bad;
    bad.add_tensor({DType::F64, {2, 2}});
    bad.add_tensor({DType::F64, {2, 3}});
    bad.add_tensor({DType::F64, {2, 3}});
    bad.add_op({OpKind::Source, {}, {0}, 0, 1, false});
    bad.add_op({OpKind::Source, {}, {1}, 0, 1, false});
    bad.add_op({OpKind::Add, {0, 1}, {2}, 0, 1, true});
    CHECK_THROWS(bad.finish());

    RuntimeGraph wrong;
    wrong.add_tensor({DType::F64, {2, 1}});
    wrong.add_tensor({DType::F64, {2, 3}});
    wrong.add_tensor({DType::F64, {2, 1}});
    wrong.add_op({OpKind::Source, {}, {0}, 0, 1, false});
    wrong.add_op({OpKind::Source, {}, {1}, 0, 1, false});
    wrong.add_op({OpKind::Add, {0, 1}, {2}, 0, 1, true});
    CHECK_THROWS(wrong.finish());
  }
  SUBCASE("operator arity and scratch minimum are enforced") {
    RuntimeGraph arity;
    arity.add_tensor({DType::F64, {2}});
    arity.add_tensor({DType::F64, {2}});
    arity.add_op({OpKind::Source, {}, {0}, 0, 1, false});
    CHECK_THROWS(arity.add_op({OpKind::Add, {}, {1}, 0, 1, true}));

    RuntimeGraph scratch;
    scratch.add_tensor({DType::F64, {2, 3}});
    scratch.add_tensor({DType::F64, {3, 4}});
    scratch.add_tensor({DType::F64, {2, 4}});
    scratch.add_op({OpKind::Source, {}, {0}, 0, 1, false});
    scratch.add_op({OpKind::Source, {}, {1}, 0, 1, false});
    scratch.add_op({OpKind::ScratchGemm, {0, 1}, {2}, 8, 1, true});
    CHECK_THROWS(scratch.finish());
  }
  SUBCASE("tensor byte multiplication is checked") {
    RuntimeGraph huge;
    huge.add_tensor({DType::F64, {INT64_MAX, 2}});
    huge.add_op({OpKind::Source, {}, {0}, 0, 1, false});
    CHECK_THROWS(huge.finish());
  }
}
