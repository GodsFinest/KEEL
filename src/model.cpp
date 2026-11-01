#include <keel/model.hpp>

#include <algorithm>
#include <stdexcept>

namespace keel {

// ---------------------------------------------------------------------------
// Graph builder

TensorId GraphBuilder::add_tensor(int64_t size, OpId producer, bool required,
                                  bool pinned) {
  if (g_.tensors.size() >= (size_t)kMaxTensors)
    throw std::invalid_argument("too many tensors (Phase 1 cap is 64)");
  if (size <= 0) throw std::invalid_argument("tensor size must be > 0");
  if (producer < -1 || producer >= (OpId)g_.ops.size())
    throw std::invalid_argument("producer op out of range");
  TensorDef t;
  t.size = size;
  t.producer = producer;
  t.required = required;
  t.pinned = pinned;
  g_.tensors.push_back(t);
  return (TensorId)g_.tensors.size() - 1;
}

OpId GraphBuilder::add_op(std::vector<TensorId> inputs,
                          std::vector<TensorId> outputs, int64_t scratch,
                          int64_t cost, bool replayable) {
  if (g_.ops.size() >= (size_t)kMaxOps)
    throw std::invalid_argument("too many ops (Phase 1 cap is 64)");
  if (cost <= 0) throw std::invalid_argument("op cost must be > 0");
  if (scratch < 0) throw std::invalid_argument("op scratch must be >= 0");
  for (TensorId t : inputs)
    if (t < 0 || t >= (TensorId)g_.tensors.size())
      throw std::invalid_argument("input tensor out of range");
  for (TensorId t : outputs) {
    if (t < 0 || t >= (TensorId)g_.tensors.size())
      throw std::invalid_argument("output tensor out of range");
    if (g_.tensors[t].producer != -1)
      throw std::invalid_argument("tensor has a producer already");
    // The op being added is its producer; producer index will be ops.size().
  }
  // No aliases/mutation (SPEC §1): an op may not read and write the same
  // tensor, and input/output lists must not repeat entries.
  for (TensorId t : inputs)
    for (TensorId u : outputs)
      if (t == u) throw std::invalid_argument("in-place op (input == output)");
  for (size_t i = 0; i < inputs.size(); ++i)
    for (size_t j = i + 1; j < inputs.size(); ++j)
      if (inputs[i] == inputs[j])
        throw std::invalid_argument("duplicate input tensor");
  for (size_t i = 0; i < outputs.size(); ++i)
    for (size_t j = i + 1; j < outputs.size(); ++j)
      if (outputs[i] == outputs[j])
        throw std::invalid_argument("duplicate output tensor");
  OpDef op;
  op.inputs = std::move(inputs);
  op.outputs = std::move(outputs);
  op.scratch = scratch;
  op.cost = cost;
  op.replayable = replayable;
  OpId id = (OpId)g_.ops.size();
  for (TensorId t : op.outputs) g_.tensors[t].producer = id;
  g_.ops.push_back(std::move(op));
  return id;
}

Graph GraphBuilder::finish() {
  // Every tensor must have exactly one producer (source ops produce captured
  // external inputs; a producer-less tensor could never become resident).
  for (int t = 0; t < (int)g_.tensors.size(); ++t)
    if (g_.tensors[t].producer == -1)
      throw std::invalid_argument("tensor with no producer");
  // Kahn topological sort over producer->consumer edges. An op depends on the
  // producers of its inputs. Cycles are rejected here.
  const int n = (int)g_.ops.size();
  std::vector<int> indeg(n, 0);
  std::vector<std::vector<OpId>> succ(n);
  for (int v = 0; v < n; ++v) {
    for (TensorId t : g_.ops[v].inputs) {
      succ[g_.tensors[t].producer].push_back(v);
      ++indeg[v];
    }
  }
  std::vector<OpId> ready;
  for (int v = 0; v < n; ++v)
    if (indeg[v] == 0) ready.push_back(v);
  g_.topo.clear();
  while (!ready.empty()) {
    OpId v = ready.back();
    ready.pop_back();
    g_.topo.push_back(v);
    for (OpId w : succ[v])
      if (--indeg[w] == 0) ready.push_back(w);
  }
  if ((int)g_.topo.size() != n) throw std::invalid_argument("graph has a cycle");
  // Precompute per-tensor reader masks (hot path for DROP/RELEASE legality).
  g_.readers.assign(g_.tensors.size(), 0);
  for (int v = 0; v < n; ++v)
    for (TensorId t : g_.ops[v].inputs)
      g_.readers[t] |= (OpMask(1) << v);
  return std::move(g_);
}

bool Graph::is_dag() const { return topo.size() == ops.size(); }

// ---------------------------------------------------------------------------
// Accounting

int64_t tensor_bytes(const Graph& g, TensorMask m) {
  int64_t sum = 0;
  for (int i = 0; i < (int)g.tensors.size(); ++i)
    if (m & (TensorMask(1) << i)) {
      if (__builtin_add_overflow(sum, g.tensors[i].size, &sum))
        throw std::overflow_error("tensor byte sum overflow");
    }
  return sum;
}

int64_t inflight_reserved(const Graph& g, OpMask m) {
  // SPEC v1.1 §4: each in-flight op holds a live reservation of its scratch
  // AND its output bytes until its completion is observed. This prevents
  // concurrent launches from collectively oversubscribing the region
  // (audit finding 1): two 6-unit launches against a 9-unit region cannot
  // both be in flight, regardless of when their outputs materialize.
  int64_t sum = 0;
  for (int i = 0; i < (int)g.ops.size(); ++i) {
    if (!(m & (OpMask(1) << i))) continue;
    int64_t step;
    if (__builtin_add_overflow(sum, g.ops[i].scratch, &step))
      throw std::overflow_error("in-flight reservation overflow");
    for (TensorId t : g.ops[i].outputs)
      if (__builtin_add_overflow(step, g.tensors[t].size, &step))
        throw std::overflow_error("in-flight reservation overflow");
    sum = step;
  }
  return sum;
}

Memory memory_of(const Graph& g, const State& s, const Request& r) {
  // SPEC v1.1 §4: the ceiling is the shrinking physical region in both
  // phases; COMMIT(Bn) guarantees it is <= Bn from acknowledgment onward.
  Memory mem;
  if (__builtin_add_overflow(tensor_bytes(g, s.resident),
                             inflight_reserved(g, s.inflight), &mem.usage))
    throw std::overflow_error("usage overflow");
  if (__builtin_sub_overflow(r.Bo, s.released_bytes, &mem.ceiling))
    throw std::overflow_error("ceiling overflow");
  mem.ok = mem.usage >= 0 && mem.ceiling >= 0;
  return mem;
}

std::string validate_state(const Graph& g, const State& s, const Request& r) {
  // SPEC v1.1 §3: structural sanity of a captured (or mid-search) state.
  if (std::string e = validate_request(r); !e.empty()) return e;
  const int nops = (int)g.ops.size();
  const int nts = (int)g.tensors.size();
  for (int t = 0; t < nts; ++t) {
    if (!(s.resident & (TensorMask(1) << t))) continue;
    OpId p = g.tensors[t].producer;
    if (p < 0 || p >= nops || !(s.done & (OpMask(1) << p)))
      return "resident tensor without a completed producer";
  }
  if (s.done & s.inflight) return "op both done and in flight";
  for (int v = 0; v < nops; ++v) {
    if (!(s.inflight & (OpMask(1) << v))) continue;
    if ((s.resident & g.outputs_mask(v)) != 0)
      return "in-flight op's output already resident";
    for (TensorId t : g.ops[v].inputs)
      if (!(s.resident & (TensorMask(1) << t)))
        return "in-flight op with a non-resident input";
  }
  if (!s.committed && s.prep_used > r.K) return "prep budget already exceeded";
  if (s.released_bytes < 0 || s.released_bytes > r.Bo)
    return "released bytes out of range";
  Memory m = memory_of(g, s, r);
  if (!m.ok) return "accounting overflow";
  if (m.usage > m.ceiling) return "state violates the memory invariant";
  return "";
}

std::string validate_request(const Request& r) {
  if (r.Bo <= 0) return "Bo must be positive";
  if (r.Bn < 0) return "Bn must be non-negative";
  if (r.Bn >= r.Bo) return "Bn must be strictly below Bo";
  if (r.K < 0) return "K must be non-negative";
  return "";
}

// ---------------------------------------------------------------------------
// Legality (SPEC §5). Each predicate is written from the SPEC text.

const char* err_name(Err e) {
  switch (e) {
    case Err::OK: return "OK";
    case Err::OP_INFLIGHT: return "OP_INFLIGHT";
    case Err::INPUTS_NOT_RESIDENT: return "INPUTS_NOT_RESIDENT";
    case Err::OUTPUT_ALREADY_RESIDENT: return "OUTPUT_ALREADY_RESIDENT";
    case Err::NOT_REPLAYABLE: return "NOT_REPLAYABLE";
    case Err::OVER_CEILING: return "OVER_CEILING";
    case Err::PREP_BUDGET_EXCEEDED: return "PREP_BUDGET_EXCEEDED";
    case Err::NOT_RESIDENT: return "NOT_RESIDENT";
    case Err::PROTECTED_BY_INFLIGHT: return "PROTECTED_BY_INFLIGHT";
    case Err::PINNED: return "PINNED";
    case Err::NOT_INFLIGHT: return "NOT_INFLIGHT";
    case Err::RELEASE_OUTSIDE_PREPARING: return "RELEASE_OUTSIDE_PREPARING";
    case Err::COMMIT_SHORTFALL: return "COMMIT_SHORTFALL";
    case Err::ALREADY_COMMITTED: return "ALREADY_COMMITTED";
  }
  return "?";
}

namespace {

// Shared core of RUN and LAUNCH (SPEC writes them as one check). This helper
// is model-internal; the checker re-derives it from the SPEC.
Legality run_core(const Graph& g, const State& s, const Request& r, OpId v) {
  if (v < 0 || v >= (OpId)g.ops.size()) return {false, Err::INPUTS_NOT_RESIDENT};
  const OpDef& op = g.ops[v];
  if (s.inflight & (OpMask(1) << v)) return {false, Err::OP_INFLIGHT};
  if ((s.resident & g.inputs_mask(v)) != g.inputs_mask(v))
    return {false, Err::INPUTS_NOT_RESIDENT};
  if (s.resident & g.outputs_mask(v))
    return {false, Err::OUTPUT_ALREADY_RESIDENT};
  if (s.done & (OpMask(1) << v)) {
    if (!op.replayable) return {false, Err::NOT_REPLAYABLE};
  }
  Memory mem = memory_of(g, s, r);
  if (!mem.ok) return {false, Err::OVER_CEILING};
  int64_t need;
  if (__builtin_add_overflow(mem.usage, tensor_bytes(g, g.outputs_mask(v)),
                             &need))
    return {false, Err::OVER_CEILING};
  if (__builtin_add_overflow(need, op.scratch, &need))
    return {false, Err::OVER_CEILING};
  if (need > mem.ceiling) return {false, Err::OVER_CEILING};
  if (!s.committed) {  // PREPARING: compute counts against K
    int64_t used;
    if (__builtin_add_overflow(s.prep_used, op.cost, &used))
      return {false, Err::PREP_BUDGET_EXCEEDED};
    if (used > r.K) return {false, Err::PREP_BUDGET_EXCEEDED};
  }
  return {true, Err::OK};
}

}  // namespace

Legality can_run(const Graph& g, const State& s, const Request& r, OpId v) {
  return run_core(g, s, r, v);
}

Legality can_launch(const Graph& g, const State& s, const Request& r, OpId v) {
  return run_core(g, s, r, v);
}

Legality can_wait([[maybe_unused]] const Graph& g, const State& s, OpId v) {
  if (v < 0 || v >= (OpId)g.ops.size()) return {false, Err::NOT_INFLIGHT};
  if (!(s.inflight & (OpMask(1) << v))) return {false, Err::NOT_INFLIGHT};
  return {true, Err::OK};
}

Legality can_drop(const Graph& g, const State& s, TensorId t) {
  if (t < 0 || t >= (TensorId)g.tensors.size()) return {false, Err::NOT_RESIDENT};
  if (!(s.resident & (TensorMask(1) << t))) return {false, Err::NOT_RESIDENT};
  if (g.tensors[t].pinned) return {false, Err::PINNED};
  if (s.inflight & g.readers_of(t)) return {false, Err::PROTECTED_BY_INFLIGHT};
  return {true, Err::OK};
}

Legality can_release(const Graph& g, const State& s, TensorId t) {
  if (s.committed) return {false, Err::RELEASE_OUTSIDE_PREPARING};
  if (t < 0 || t >= (TensorId)g.tensors.size()) return {false, Err::NOT_RESIDENT};
  if (!(s.resident & (TensorMask(1) << t))) return {false, Err::NOT_RESIDENT};
  if (g.tensors[t].pinned) return {false, Err::PINNED};
  if (s.inflight & g.readers_of(t)) return {false, Err::PROTECTED_BY_INFLIGHT};
  return {true, Err::OK};
}

Legality can_commit([[maybe_unused]] const Graph& g, const State& s, const Request& r) {
  if (s.committed) return {false, Err::ALREADY_COMMITTED};
  // SPEC §5: co-tenant must physically receive at least the promised bytes.
  if (s.released_bytes < r.Bo - r.Bn) return {false, Err::COMMIT_SHORTFALL};
  return {true, Err::OK};
}

// ---------------------------------------------------------------------------
// Transitions (precondition: legality checked by caller)

State apply_run(const Graph& g, const State& s, const Request& r, OpId v) {
  (void)r;
  State n = s;
  n.done |= (OpMask(1) << v);
  n.resident |= g.outputs_mask(v);
  if (!s.committed) n.prep_used += g.ops[v].cost;
  return n;
}

State apply_launch(const Graph& g, const State& s, const Request& r, OpId v) {
  State n = s;
  n.inflight |= (OpMask(1) << v);
  if (!s.committed) n.prep_used += g.ops[v].cost;
  (void)g;
  (void)r;
  return n;
}

State apply_wait(const Graph& g, const State& s, OpId v) {
  State n = s;
  n.inflight &= ~(OpMask(1) << v);
  n.done |= (OpMask(1) << v);
  n.resident |= g.outputs_mask(v);
  return n;
}

State apply_drop(const Graph& g, const State& s, TensorId t) {
  State n = s;
  n.resident &= ~(TensorMask(1) << t);
  (void)g;
  return n;
}

State apply_release(const Graph& g, const State& s, TensorId t) {
  State n = s;
  n.resident &= ~(TensorMask(1) << t);
  n.released_bytes += g.tensors[t].size;
  return n;
}

State apply_commit(const Graph& g, const State& s, const Request& r) {
  (void)g;
  (void)r;
  State n = s;
  n.committed = true;
  return n;
}

bool is_goal(const Graph& g, const State& s) {
  TensorMask q = 0;
  for (int i = 0; i < (int)g.tensors.size(); ++i)
    if (g.tensors[i].required) q |= (TensorMask(1) << i);
  return s.committed && s.inflight == 0 && (s.resident & q) == q;
}

State capture_frontier(const State& s) {
  State n = s;
  n.prep_used = 0;
  return n;
}

// ---------------------------------------------------------------------------
// Witness text form (debugging + logs)

const char* act_name(ActKind k) {
  switch (k) {
    case ActKind::Run: return "RUN";
    case ActKind::Launch: return "LAUNCH";
    case ActKind::Wait: return "WAIT";
    case ActKind::Drop: return "DROP";
    case ActKind::Release: return "RELEASE";
    case ActKind::Commit: return "COMMIT";
  }
  return "?";
}

std::string witness_to_text([[maybe_unused]] const Graph& g, const Witness& w) {
  std::string out;
  for (const Act& a : w.acts) {
    out += act_name(a.kind);
    if (a.kind == ActKind::Commit) {
      out += "\n";
      continue;
    }
    out += " ";
    if (a.kind == ActKind::Drop || a.kind == ActKind::Release)
      out += "t" + std::to_string(a.target);
    else
      out += "v" + std::to_string(a.target);
    out += "\n";
  }
  return out;
}

}  // namespace keel
