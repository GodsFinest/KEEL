// KEEL Phase 2 — Production computation IR (typed runtime graphs).
//
// Extends the Phase 1 abstract model with what a real executor needs: dtypes,
// shapes, byte sizes (checked arithmetic), and stable op/tensor identities.
// `to_abstract()` is the guarded conversion into the verified Phase 1 model
// used by the planner/checker/oracle; it fails loudly when a runtime graph
// cannot be represented exactly (op/tensor caps), so no silent semantic
// mismatch can arise (docs/ROADMAP.md Phase 2 acceptance).
//
// Operation contract (SPEC v1.1 §1, unchanged): immutable intermediate values,
// deterministic ops, declared scratch, no mutation/aliases/control flow.
// `Source` ops model captured external inputs and execute at most once.

#pragma once

#include <keel/model.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace keel::rt {

enum class DType { F64, F32, I64 };

size_t dtype_size(DType t);

const char* dtype_name(DType t);

enum class OpKind { Source, Add, Mul, MatMul, ReduceSum, ResidualAdd, ScratchGemm };
const char* op_kind_name(OpKind k);

// One tensor of a runtime graph.
struct TensorSpec {
  DType dtype = DType::F64;
  std::vector<int64_t> shape;  // all dims > 0
  bool required = false;       // persistent output obligation
  bool pinned = false;
};

// One operation of a runtime graph.
struct OpSpec {
  OpKind kind = OpKind::Add;
  std::vector<TensorId> inputs;
  std::vector<TensorId> outputs;
  int64_t scratch = 0;    // bytes of workspace reserved while executing
  int64_t cost = 1;       // work units (planning currency, as in the model)
  bool replayable = true; // false only for Source
};

// Output ELEMENT COUNT of an op, computed from its input shapes (checked).
// Elementwise ops broadcast (output = the larger operand's element count).
// Source ops declare sizes via their output TensorSpecs instead.
int64_t infer_output_bytes(OpKind kind, const std::vector<int64_t>& in0,
                           const std::vector<int64_t>& in1);

class RuntimeGraph {
 public:
  TensorId add_tensor(TensorSpec spec);
  OpId add_op(OpSpec spec);

  // Validates everything (ids, single producer, no in-place, duplicate free,
  // positive shapes, non-negative scratch/cost, replay rules, DAG, declared
  // output sizes consistent with op kind) and computes derived tables.
  void finish();

  const std::vector<TensorSpec>& tensors() const { return tensors_; }
  const std::vector<OpSpec>& ops() const { return ops_; }
  const std::vector<OpId>& topo() const { return topo_; }
  OpId producer_of(TensorId t) const { return producer_[t]; }
  int64_t tensor_bytes(TensorId t) const { return tensor_bytes_[t]; }
  const std::vector<int64_t>& tensor_bytes_all() const { return tensor_bytes_; }
  OpMask readers_of(TensorId t) const { return readers_[t]; }
  TensorMask outputs_mask(OpId v) const {
    TensorMask m = 0;
    for (TensorId t : ops_[v].outputs) m |= (TensorMask(1) << t);
    return m;
  }
  TensorMask inputs_mask(OpId v) const {
    TensorMask m = 0;
    for (TensorId t : ops_[v].inputs) m |= (TensorMask(1) << t);
    return m;
  }

  // Guarded conversion into the Phase 1 abstract model. Throws when the
  // runtime graph cannot be represented exactly (caps, sizes).
  Graph to_abstract() const;

  // The planner/checker use abstract tensor identities, which are not
  // required to equal runtime tensor identities.  This projection is the
  // explicit translation boundary used by handover capture and witness
  // application.  Operation ids are currently identity-mapped, but are kept
  // explicit so that a future abstract rebuild cannot silently change the
  // contract.
  struct AbstractProjection {
    Graph graph;
    std::vector<TensorId> runtime_to_abstract_tensor;
    std::vector<TensorId> abstract_to_runtime_tensor;
    std::vector<OpId> runtime_to_abstract_op;
    std::vector<OpId> abstract_to_runtime_op;
  };

  AbstractProjection to_abstract_projection() const;

 private:
  std::vector<TensorSpec> tensors_;
  std::vector<OpSpec> ops_;
  std::vector<OpId> producer_;  // per tensor: -1 until produced
  std::vector<int64_t> tensor_bytes_;
  std::vector<OpMask> readers_;
  std::vector<OpId> topo_;
  bool finished_ = false;
};

}  // namespace keel::rt
