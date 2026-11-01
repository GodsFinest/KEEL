#include <keel/rt/ir.hpp>

#include <algorithm>
#include <stdexcept>

namespace keel::rt {

size_t dtype_size(DType t) {
  switch (t) {
    case DType::F64: return 8;
    case DType::F32: return 4;
    case DType::I64: return 8;
  }
  return 0;
}

const char* dtype_name(DType t) {
  switch (t) {
    case DType::F64: return "f64";
    case DType::F32: return "f32";
    case DType::I64: return "i64";
  }
  return "?";
}

const char* op_kind_name(OpKind k) {
  switch (k) {
    case OpKind::Source: return "source";
    case OpKind::Add: return "add";
    case OpKind::Mul: return "mul";
    case OpKind::MatMul: return "matmul";
    case OpKind::ReduceSum: return "reduce_sum";
    case OpKind::ResidualAdd: return "residual_add";
    case OpKind::ScratchGemm: return "scratch_gemm";
  }
  return "?";
}

namespace {

int64_t checked_mul(int64_t a, int64_t b, const char* what) {
  int64_t r;
  if (__builtin_mul_overflow(a, b, &r))
    throw std::overflow_error(std::string(what) + ": multiplication overflows");
  return r;
}

int64_t shape_numel(const std::vector<int64_t>& shape, const char* what) {
  if (shape.empty()) throw std::invalid_argument("tensor rank must be positive");
  int64_t n = 1;
  for (int64_t d : shape) {
    if (d <= 0) throw std::invalid_argument("shape dims must be positive");
    n = checked_mul(n, d, what);
  }
  return n;
}

std::vector<int64_t> broadcast_shape(const std::vector<int64_t>& a,
                                     const std::vector<int64_t>& b) {
  const size_t rank = std::max(a.size(), b.size());
  std::vector<int64_t> out(rank, 1);
  const auto aligned_dim = [rank](const std::vector<int64_t>& shape,
                                  size_t axis) {
    const size_t offset = rank - shape.size();
    return axis < offset ? int64_t{1} : shape[axis - offset];
  };
  for (size_t i = 0; i < rank; ++i) {
    const int64_t ad = aligned_dim(a, i);
    const int64_t bd = aligned_dim(b, i);
    if (ad != bd && ad != 1 && bd != 1)
      throw std::invalid_argument("incompatible broadcast shapes");
    out[i] = std::max(ad, bd);
  }
  return out;
}

std::vector<int64_t> expected_shape(OpKind kind,
                                    const std::vector<int64_t>& in0,
                                    const std::vector<int64_t>& in1) {
  switch (kind) {
    case OpKind::Add:
    case OpKind::Mul:
    case OpKind::ResidualAdd:
      return broadcast_shape(in0, in1);
    case OpKind::ReduceSum:
      if (in0.size() != 2)
        throw std::invalid_argument("reduce_sum requires a 2-D input");
      return {in0[0], 1};
    case OpKind::MatMul:
    case OpKind::ScratchGemm:
      if (in0.size() != 2 || in1.size() != 2)
        throw std::invalid_argument("matmul requires 2-D operands");
      if (in0[1] != in1[0])
        throw std::invalid_argument("matmul inner dims mismatch");
      return {in0[0], in1[1]};
    case OpKind::Source:
      throw std::invalid_argument("source ops declare their own sizes");
  }
  throw std::invalid_argument("unknown op kind");
}

}  // namespace

int64_t infer_output_bytes(OpKind kind, const std::vector<int64_t>& in0,
                           const std::vector<int64_t>& in1) {
  const std::vector<int64_t> shape = expected_shape(kind, in0, in1);
  return shape_numel(shape, "infer_output_bytes");
}

TensorId RuntimeGraph::add_tensor(TensorSpec spec) {
  if (finished_)
    throw std::invalid_argument("graph already finished");
  if ((int)tensors_.size() >= kMaxTensors)
    throw std::invalid_argument("too many tensors (abstract-model cap is 64)");
  if (spec.shape.empty())
    throw std::invalid_argument("tensor shape must be non-empty");
  for (int64_t d : spec.shape)
    if (d <= 0) throw std::invalid_argument("shape dims must be positive");
  tensors_.push_back(std::move(spec));
  producer_.push_back(-1);
  return (TensorId)tensors_.size() - 1;
}

OpId RuntimeGraph::add_op(OpSpec spec) {
  if (finished_)
    throw std::invalid_argument("graph already finished");
  if ((int)ops_.size() >= kMaxOps)
    throw std::invalid_argument("too many ops (abstract-model cap is 64)");
  if (spec.cost <= 0) throw std::invalid_argument("op cost must be > 0");
  if (spec.scratch < 0) throw std::invalid_argument("op scratch must be >= 0");
  for (TensorId t : spec.inputs)
    if (t < 0 || t >= (TensorId)tensors_.size())
      throw std::invalid_argument("input tensor out of range");
  for (TensorId t : spec.outputs) {
    if (t < 0 || t >= (TensorId)tensors_.size())
      throw std::invalid_argument("output tensor out of range");
    if (producer_[t] != -1)
      throw std::invalid_argument("tensor already has a producer");
  }
  for (TensorId t : spec.inputs)
    for (TensorId u : spec.outputs)
      if (t == u) throw std::invalid_argument("in-place op (input == output)");
  for (size_t i = 0; i < spec.inputs.size(); ++i)
    for (size_t j = i + 1; j < spec.inputs.size(); ++j)
      if (spec.inputs[i] == spec.inputs[j])
        throw std::invalid_argument("duplicate input tensor");
  for (size_t i = 0; i < spec.outputs.size(); ++i)
    for (size_t j = i + 1; j < spec.outputs.size(); ++j)
      if (spec.outputs[i] == spec.outputs[j])
        throw std::invalid_argument("duplicate output tensor");
  if (spec.kind == OpKind::Source && (!spec.inputs.empty() || spec.replayable))
    throw std::invalid_argument("source ops take no inputs and are non-replayable");
  if (spec.kind != OpKind::Source && spec.inputs.empty() && !spec.outputs.empty()) {
    // inputless compute ops are allowed only for Source; everything else must
    // consume something (keeps the model's "inputs resident" rule meaningful).
    throw std::invalid_argument("non-source op requires inputs");
  }
  OpId id = (OpId)ops_.size();
  for (TensorId t : spec.outputs) producer_[t] = id;
  ops_.push_back(std::move(spec));
  return id;
}

void RuntimeGraph::finish() {
  if (finished_) return;
  const int n = (int)ops_.size();
  // Byte sizes (checked).
  tensor_bytes_.assign(tensors_.size(), 0);
  for (int t = 0; t < (int)tensors_.size(); ++t) {
    const TensorSpec& ts = tensors_[t];
    if (ts.dtype != DType::F64)
      throw std::invalid_argument("runtime currently supports F64 tensors only");
    int64_t numel = shape_numel(ts.shape, "tensor size");
    tensor_bytes_[t] = checked_mul(numel, (int64_t)dtype_size(ts.dtype),
                                   "tensor size");
  }
  // Every tensor must have exactly one producer.
  for (int t = 0; t < (int)tensors_.size(); ++t)
    if (producer_[t] == -1)
      throw std::invalid_argument("tensor with no producer");
  for (int v = 0; v < n; ++v)
    for (TensorId t : ops_[v].inputs)
      if (producer_[t] >= v)
        throw std::invalid_argument(
            "operations must be added in dependency order");
  // Op-level shape/type consistency.
  for (int v = 0; v < n; ++v) {
    const OpSpec& op = ops_[v];
    if (op.outputs.size() != 1)
      throw std::invalid_argument("runtime ops must produce exactly one output");
    if (op.kind == OpKind::Source) {
      if (!op.inputs.empty() || op.replayable)
        throw std::invalid_argument("source ops take no inputs and are non-replayable");
      continue;
    }
    const bool elementwise = op.kind == OpKind::Add || op.kind == OpKind::Mul ||
                             op.kind == OpKind::ResidualAdd;
    const size_t expected_inputs = elementwise ? 1 :
                                   (op.kind == OpKind::ReduceSum ? 1 : 2);
    if (op.inputs.size() != expected_inputs &&
        !(elementwise && op.inputs.size() == 2))
      throw std::invalid_argument("unsupported operator arity");
    for (TensorId t : op.inputs) {
      if (tensors_[t].dtype != tensors_[op.outputs[0]].dtype)
        throw std::invalid_argument("dtype mismatch between inputs and output");
    }
    const std::vector<int64_t>& in0 = tensors_[op.inputs[0]].shape;
    const std::vector<int64_t>& in1 = op.inputs.size() > 1
                                          ? tensors_[op.inputs[1]].shape
                                          : in0;
    const std::vector<int64_t> want_shape = expected_shape(op.kind, in0, in1);
    if (tensors_[op.outputs[0]].shape != want_shape)
      throw std::invalid_argument("declared output shape does not match op semantics");
    int64_t want_numel = infer_output_bytes(
        op.kind, tensors_[op.inputs[0]].shape,
        op.inputs.size() > 1 ? tensors_[op.inputs[1]].shape
                             : tensors_[op.inputs[0]].shape);
    int64_t declared_bytes = tensor_bytes_[op.outputs[0]];
    int64_t declared_numel = declared_bytes / (int64_t)dtype_size(tensors_[op.outputs[0]].dtype);
    if (want_numel != declared_numel ||
        declared_bytes % (int64_t)dtype_size(tensors_[op.outputs[0]].dtype) != 0)
      throw std::invalid_argument(
          "declared output size does not match op semantics");
    if (op.kind == OpKind::ScratchGemm) {
      const int64_t scratch_elements = checked_mul(in1[0], in1[1],
                                                   "scratch size");
      const int64_t min_scratch = checked_mul(
          scratch_elements, static_cast<int64_t>(dtype_size(DType::F64)),
          "scratch size");
      if (op.scratch < min_scratch)
        throw std::invalid_argument("scratch_gemm scratch is too small");
    }
  }
  // Reader masks + Kahn topo over producer->consumer edges.
  readers_.assign(tensors_.size(), 0);
  std::vector<int> indeg(n, 0);
  std::vector<std::vector<OpId>> succ(n);
  for (int v = 0; v < n; ++v) {
    for (TensorId t : ops_[v].inputs) {
      readers_[t] |= (OpMask(1) << v);
      succ[producer_[t]].push_back(v);
      ++indeg[v];
    }
  }
  std::vector<OpId> ready;
  for (int v = 0; v < n; ++v)
    if (indeg[v] == 0) ready.push_back(v);
  topo_.clear();
  while (!ready.empty()) {
    OpId v = ready.back();
    ready.pop_back();
    topo_.push_back(v);
    for (OpId w : succ[v])
      if (--indeg[w] == 0) ready.push_back(w);
  }
  if ((int)topo_.size() != n) throw std::invalid_argument("graph has a cycle");
  finished_ = true;
}

RuntimeGraph::AbstractProjection RuntimeGraph::to_abstract_projection() const {
  if (!finished_) throw std::invalid_argument("call finish() before to_abstract()");
  GraphBuilder b;
  // Ops and tensors must be interleaved exactly as the builder requires
  // (a tensor's producer must exist when a consumer references it). Runtime
  // graphs record producers, so rebuild in the same creation order using the
  // recorded producer ids: create tensors lazily with the right producer.
  // Strategy: iterate ops in creation order; before adding op v, add any
  // tensors whose producer is v or that are not yet created and produced by
  // an earlier op.
  const int n = (int)ops_.size();
  std::vector<TensorId> map(tensors_.size(), -1);
  for (int v = 0; v < n; ++v) {
    // Inputs must already exist (creation order guarantees it: a tensor is
    // created before its consumer op because consumers come later).
    for (TensorId t : ops_[v].inputs)
      if (map[t] == -1)
        throw std::invalid_argument("input tensor created after its consumer");
    // Create this op's outputs now.
    std::vector<TensorId> abs_outs;
    for (TensorId t : ops_[v].outputs) {
      const TensorSpec& ts = tensors_[t];
      // Producer -1: the abstract builder assigns the real producer when the
      // consuming add_op runs below (same pattern as the Phase 1 tests).
      map[t] = b.add_tensor(tensor_bytes_[t], /*producer=*/-1,
                            ts.required, ts.pinned);
      abs_outs.push_back(map[t]);
    }
    std::vector<TensorId> abs_ins;
    for (TensorId t : ops_[v].inputs) abs_ins.push_back(map[t]);
    b.add_op(abs_ins, abs_outs, ops_[v].scratch, ops_[v].cost,
             ops_[v].replayable);
  }
  AbstractProjection projection;
  projection.graph = b.finish();
  projection.runtime_to_abstract_tensor = map;
  projection.abstract_to_runtime_tensor.assign(map.size(), -1);
  for (TensorId runtime_id = 0; runtime_id < (TensorId)map.size();
       ++runtime_id) {
    const TensorId abstract_id = map[runtime_id];
    if (abstract_id < 0 || abstract_id >= (TensorId)map.size() ||
        projection.abstract_to_runtime_tensor[abstract_id] != -1)
      throw std::invalid_argument("internal: runtime/abstract tensor map is not bijective");
    projection.abstract_to_runtime_tensor[abstract_id] = runtime_id;
  }
  projection.runtime_to_abstract_op.resize(ops_.size());
  projection.abstract_to_runtime_op.resize(ops_.size());
  for (OpId v = 0; v < (OpId)ops_.size(); ++v) {
    // GraphBuilder receives operations in runtime creation order.  Keep the
    // mapping explicit and validate that assumption instead of relying on it
    // at each handover call site.
    projection.runtime_to_abstract_op[v] = v;
    projection.abstract_to_runtime_op[v] = v;
  }
  if (projection.graph.tensors.size() != tensors_.size() ||
      projection.graph.ops.size() != ops_.size())
    throw std::invalid_argument("internal: abstract projection size mismatch");
  for (TensorId abstract_id = 0;
       abstract_id < (TensorId)projection.abstract_to_runtime_tensor.size();
       ++abstract_id) {
    if (projection.abstract_to_runtime_tensor[abstract_id] < 0)
      throw std::invalid_argument("internal: abstract tensor map is incomplete");
  }
  // Re-check every semantic edge across the boundary.  A bijection alone is
  // insufficient: a future builder change must not silently permute required
  // or pinned obligations, producers, inputs, outputs, or op accounting.
  for (TensorId runtime_t = 0; runtime_t < (TensorId)tensors_.size();
       ++runtime_t) {
    const TensorId abstract_t = projection.runtime_to_abstract_tensor[runtime_t];
    const TensorDef& at = projection.graph.tensors[abstract_t];
    const TensorSpec& rt = tensors_[runtime_t];
    if (at.size != tensor_bytes_[runtime_t] || at.required != rt.required ||
        at.pinned != rt.pinned ||
        at.producer != projection.runtime_to_abstract_op[producer_[runtime_t]])
      throw std::invalid_argument("internal: tensor projection changed semantics");
  }
  for (OpId runtime_v = 0; runtime_v < (OpId)ops_.size(); ++runtime_v) {
    const OpId abstract_v = projection.runtime_to_abstract_op[runtime_v];
    const OpDef& ao = projection.graph.ops[abstract_v];
    const OpSpec& ro = ops_[runtime_v];
    std::vector<TensorId> abstract_inputs;
    std::vector<TensorId> abstract_outputs;
    for (TensorId t : ro.inputs)
      abstract_inputs.push_back(projection.runtime_to_abstract_tensor[t]);
    for (TensorId t : ro.outputs)
      abstract_outputs.push_back(projection.runtime_to_abstract_tensor[t]);
    if (ao.inputs != abstract_inputs || ao.outputs != abstract_outputs ||
        ao.scratch != ro.scratch || ao.cost != ro.cost ||
        ao.replayable != ro.replayable)
      throw std::invalid_argument("internal: operation projection changed semantics");
  }
  return projection;
}

Graph RuntimeGraph::to_abstract() const {
  return to_abstract_projection().graph;
}

}  // namespace keel::rt
