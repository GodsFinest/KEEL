#include <keel/rt/executor.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <utility>

namespace keel::rt {

namespace {

// Deterministic source pattern: seeded hash reinterpreted as f64 in [0, 1).
// Same seed -> same bits everywhere (reference included).
double source_value(uint64_t seed, uint64_t index) {
  uint64_t x = seed * 0x9E3779B97F4A7C15ull + index * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 30)) * 0x94D049BB133111EBull;
  x ^= x >> 31;
  return (double)(x % 1000000ull) / 1000000.0;
}

int64_t numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) {
    if (d <= 0 || __builtin_mul_overflow(n, d, &n))
      throw std::invalid_argument("invalid tensor shape");
  }
  return n;
}

int64_t broadcast_index(int64_t linear, const std::vector<int64_t>& out_shape,
                        const std::vector<int64_t>& in_shape) {
  int64_t input_linear = 0;
  int64_t stride = 1;
  for (size_t out_axis = out_shape.size(); out_axis-- > 0;) {
    const int64_t coord = linear % out_shape[out_axis];
    linear /= out_shape[out_axis];
    const size_t offset = out_shape.size() - in_shape.size();
    const bool has_input_axis = out_axis >= offset;
    const size_t input_axis = has_input_axis ? out_axis - offset : 0;
    const int64_t input_coord = has_input_axis && in_shape[input_axis] != 1
                                    ? coord
                                    : 0;
    if (has_input_axis) {
      input_linear += input_coord * stride;
      stride *= in_shape[input_axis];
    }
  }
  return input_linear;
}

// Production kernel. The plain-vector reference below deliberately does not
// call this function: numerical agreement is an independently computed
// property, not a consequence of sharing implementation.
void run_kernel(const RuntimeGraph& g, OpId v, const std::vector<const double*>& ins,
                double* out, double* scratch) {
  const OpSpec& op = g.ops()[v];
  const auto& shape_out = g.tensors()[op.outputs[0]].shape;
  const int64_t n_out = numel(shape_out);
  const double* a = ins.empty() ? nullptr : ins[0];
  // Unary elementwise (single input) uses the same buffer as both operands.
  const double* b = ins.size() > 1 ? ins[1] : a;
  switch (op.kind) {
    case OpKind::Source:
      for (int64_t i = 0; i < n_out; ++i)
        out[i] = source_value(42, (uint64_t)i);
      break;
    case OpKind::Add:
    case OpKind::Mul:
    case OpKind::ResidualAdd: {
      for (int64_t i = 0; i < n_out; ++i) {
        const int64_t ia = ins.size() > 1
                               ? broadcast_index(i, shape_out,
                                                 g.tensors()[op.inputs[0]].shape)
                               : i;
        const int64_t ib = ins.size() > 1
                               ? broadcast_index(i, shape_out,
                                                 g.tensors()[op.inputs[1]].shape)
                               : i;
        if (op.kind == OpKind::Mul)
          out[i] = a[ia] * b[ib];
        else
          out[i] = a[ia] + b[ib];
      }
      break;
    }
    case OpKind::MatMul:
    case OpKind::ScratchGemm: {
      const int64_t m = shape_out[0], n = shape_out[1];
      const int64_t k = g.tensors()[op.inputs[0]].shape[1];
      if (op.kind == OpKind::ScratchGemm) {
        // Genuine workspace use: materialize B^T into scratch, then multiply
        // over contiguous rows of B^T. Identical summation order to MatMul's
        // i-k-j? No: i-j-k over B^T elements — the SUMMATION ORDER over p is
        // the same (p ascending), so bits match the MatMul kernel.
        for (int64_t i = 0; i < k; ++i)
          for (int64_t j = 0; j < n; ++j) scratch[j * k + i] = b[i * n + j];
        for (int64_t i = 0; i < m; ++i)
          for (int64_t j = 0; j < n; ++j) {
            volatile double s = 0.0;
            for (int64_t p = 0; p < k; ++p) s += a[i * k + p] * scratch[j * k + p];
            out[i * n + j] = s;
          }
      } else {
        for (int64_t i = 0; i < m; ++i)
          for (int64_t j = 0; j < n; ++j) {
            volatile double s = 0.0;
            for (int64_t p = 0; p < k; ++p) s += a[i * k + p] * b[p * n + j];
            out[i * n + j] = s;
          }
      }
      break;
    }
    case OpKind::ReduceSum: {
      const int64_t m = shape_out[0];
      const int64_t kk = g.tensors()[op.inputs[0]].shape[1];
      for (int64_t i = 0; i < m; ++i) {
        volatile double s = 0.0;
        for (int64_t j = 0; j < kk; ++j) s += a[i * kk + j];
        out[i] = s;
      }
      break;
    }
  }
}

}  // namespace

TaskRuntime::TaskRuntime(RuntimeGraph g, BlockArena& arena, TaskId task,
                         int64_t grant_bytes, int workers)
    : g_(std::move(g)), arena_(arena), task_(task), grant_bytes_(grant_bytes),
      workers_(workers < 1 ? 1 : workers) {
  g_.finish();
  const int64_t bs = arena_.block_size();
  if (grant_bytes_ <= 0 || grant_bytes_ % bs != 0)
    throw std::invalid_argument("grant must be a positive block multiple");
  arena_.claim(task_, grant_bytes_ / bs);
  slots_.assign(g_.tensors().size(), TensorSlot{});
  op_state_.assign(g_.ops().size(), OpState::Pending);
  source_initialized_.assign(g_.ops().size(), 0);
  for (int w = 0; w < workers_; ++w)
    threads_.emplace_back([this, w] { worker_loop(w); });
}

TaskRuntime::~TaskRuntime() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stopping_ = true;
  }
  cv_.notify_all();
  for (auto& t : threads_) t.join();
}

void TaskRuntime::worker_loop(int) {
  for (;;) {
    std::pair<OpId, std::vector<double*>> job;
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [&] { return stopping_ || !work_.empty(); });
      if (stopping_ && work_.empty()) return;
      job = std::move(work_.front());
      work_.pop();
    }
    std::exception_ptr error;
    try {
      // Payload: [inputs..., out] (+ scratch).
      const OpSpec& op = g_.ops()[job.first];
      std::vector<const double*> ins;
      for (size_t i = 0; i < op.inputs.size(); ++i)
        ins.push_back(job.second[i]);
      double* out = job.second[op.inputs.size()];
      double* scratch = job.second.size() > op.inputs.size() + 1
                            ? job.second[op.inputs.size() + 1]
                            : nullptr;
      WorkerHook hook;
      {
        std::lock_guard<std::mutex> lk(mu_);
        hook = worker_hook_;
      }
      if (hook) hook(job.first, WorkerEvent::BeforeExecute);
      run_kernel(g_, job.first, ins, out, scratch);
      if (hook) hook(job.first, WorkerEvent::BeforeComplete);
    } catch (...) {
      error = std::current_exception();
    }
    {
      std::lock_guard<std::mutex> lk(mu_);
      done_.push(Completion{job.first, error});
    }
    cv_.notify_all();
  }
}

bool TaskRuntime::inputs_ready(OpId v) const {
  for (TensorId t : g_.ops()[v].inputs) {
    if (!slots_[t].valid) return false;
    if (!slots_[t].live) return false;
  }
  return true;
}

bool TaskRuntime::try_dispatch(OpId v, bool allow_uninitialized_source) {
  if (v < 0 || v >= (OpId)g_.ops().size()) return false;
  const OpSpec& op = g_.ops()[v];
  if (op.kind == OpKind::Source && !source_initialized_[v] &&
      !allow_uninitialized_source)
    return false;
  if (op_state_[v] != OpState::Pending) return false;
  if (!inputs_ready(v)) return false;
  // Reservation BEFORE dispatch — physical, from the task's owned blocks.
  InFlightRecord rec;
  rec.op = v;
  if (op.outputs.size() != 1)
    throw std::logic_error("runtime dispatch requires exactly one output");
  std::vector<double*> payload;
  bool enqueued = false;
  try {
    rec.outputs.reserve(op.outputs.size());
    inflight_records_.reserve(inflight_records_.size() + 1);
    inflight_list_.reserve(inflight_list_.size() + 1);
    payload.reserve(op.inputs.size() + 2);
    for (TensorId t : op.outputs) {
      AllocHandle h = arena_.alloc(task_, g_.tensor_bytes(t));
      try {
        rec.outputs.push_back(h);
      } catch (...) {
        arena_.free(h);
        throw;
      }
    }
    if (op.scratch > 0) rec.scratch = arena_.alloc(task_, op.scratch);
    // Payload: [inputs..., out] (+ scratch). Pointers into the arena.
    for (TensorId t : op.inputs)
      payload.push_back((double*)const_cast<void*>(tensor_ptr(t)));
    payload.push_back((double*)arena_.ptr(rec.outputs[0]));
    if (op.scratch > 0) payload.push_back((double*)arena_.ptr(rec.scratch));

    // In-flight marking happens BEFORE the job becomes visible to workers.
    // Vector capacities were reserved above. If the queue allocation itself
    // fails, this block rolls all metadata and physical reservations back.
    std::lock_guard<std::mutex> lk(mu_);
    for (TensorId t : op.outputs) {
      slots_[t].alloc = rec.outputs[0];
      slots_[t].live = true;
      slots_[t].valid = false;
    }
    op_state_[v] = OpState::InFlight;
    inflight_records_.push_back(std::move(rec));
    inflight_list_.push_back(v);
    try {
      work_.push({v, std::move(payload)});
    } catch (...) {
      for (TensorId t : op.outputs) {
        slots_[t].alloc = AllocHandle{};
        slots_[t].live = false;
        slots_[t].valid = false;
      }
      op_state_[v] = OpState::Pending;
      inflight_list_.pop_back();
      InFlightRecord rollback = std::move(inflight_records_.back());
      inflight_records_.pop_back();
      for (AllocHandle h : rollback.outputs) arena_.free(h);
      if (rollback.scratch.valid()) arena_.free(rollback.scratch);
      throw;
    }
    ++stats_.dispatches;
    enqueued = true;
    cv_.notify_one();
  } catch (const std::exception&) {
    if (!enqueued) {
      for (AllocHandle h : rec.outputs) {
        try {
          arena_.free(h);
        } catch (...) {
        }
      }
      if (rec.scratch.valid()) {
        try {
          arena_.free(rec.scratch);
        } catch (...) {
        }
      }
    }
    ++stats_.alloc_failures;
    return false;  // memory/metadata pressure: remain ready for a later pass
  }
  return true;
}

void TaskRuntime::publish(OpId v) {
  const OpSpec& op = g_.ops()[v];
  // Completion: outputs keep their allocations (they become the tensors'
  // storage); the scratch reservation is released now that the op is done
  // (SPEC v1.1 §4: scratch is reserved only while the op executes).
  for (auto it = inflight_records_.begin(); it != inflight_records_.end();
       ++it) {
    if (it->op != v) continue;
    if (it->scratch.valid()) arena_.free(it->scratch);
    inflight_records_.erase(it);
    break;
  }
  inflight_list_.erase(
      std::remove(inflight_list_.begin(), inflight_list_.end(), v),
      inflight_list_.end());
  for (TensorId t : op.outputs) {
    // A Source operation only reserves storage.  The storage becomes valid
    // after load_source writes the caller-selected seed; exposing the worker's
    // placeholder pattern as a valid source would permit execution before
    // initialization.
    slots_[t].valid = op.kind != OpKind::Source;
    slots_[t].version += 1;
  }
  op_state_[v] = OpState::Done;
  ++stats_.completions;
  state_version_ += 1;
  const int64_t live = arena_.live_bytes(task_);
  if (live > stats_.peak_live_bytes) stats_.peak_live_bytes = live;
}

void TaskRuntime::drain_completions() {
  for (;;) {
    OpId v;
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (done_.empty()) return;
      Completion completion = done_.front();
      done_.pop();
      v = completion.op;
      if (completion.error) {
        worker_error_ = completion.error;
        continue;
      }
    }
    publish(v);
  }
}

void TaskRuntime::wait_op(OpId v) {
  if (v < 0 || v >= (OpId)op_state_.size())
    throw std::invalid_argument("wait_op: op id out of range");
  for (;;) {
    drain_completions();
    if (worker_error_) std::rethrow_exception(worker_error_);
    if (op_state_[v] == OpState::Done) return;
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait_for(lk, std::chrono::milliseconds(1));
  }
}

int TaskRuntime::dispatch_ready() {
  drain_completions();
  int n = 0;
  for (OpId v : g_.topo())
    if (try_dispatch(v)) ++n;
  return n;
}

bool TaskRuntime::run_to_completion() {
  for (;;) {
    drain_completions();
    if (worker_error_) return false;
    const int dispatched = dispatch_ready();
    bool all_done = true;
    for (OpState s : op_state_)
      if (s != OpState::Done) all_done = false;
    if (all_done) {
      std::string why;
      if (!required_outputs_ready(&why)) {
        std::vector<char> visiting(slots_.size(), 0);
        bool restored = true;
        for (TensorId t = 0; t < (TensorId)g_.tensors().size(); ++t) {
          if (!g_.tensors()[t].required || tensor_valid(t)) continue;
          if (!restore_tensor(t, visiting, &why)) {
            restored = false;
            break;
          }
        }
        if (!restored) return false;
        drain_completions();
      }
      return required_outputs_ready(&why);
    }
    if (dispatched == 0 && inflight_list_.empty()) return false;  // stall
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait_for(lk, std::chrono::milliseconds(1));
  }
}

void TaskRuntime::begin_handover(TaskId to_task) {
  if (staging_active_) throw std::logic_error("handover staging already active");
  if (to_task < 0 || to_task == task_)
    throw std::invalid_argument("invalid handover recipient");
  staging_active_ = true;
  staging_to_ = to_task;
  staged_.clear();
  staged_blocks_.clear();
}

void TaskRuntime::end_handover(bool commit) {
  if (!staging_active_) return;
  if (commit) {
    std::vector<std::pair<int32_t, int32_t>> extents;
    extents.reserve(staged_.size());
    for (const StagedExtent& e : staged_)
      extents.emplace_back(e.first, e.nblocks);
    staged_blocks_ = arena_.transfer_held_extents(task_, staging_to_, extents);
  } else {
    std::vector<std::pair<int32_t, int32_t>> extents;
    extents.reserve(staged_.size());
    for (const StagedExtent& e : staged_)
      extents.emplace_back(e.first, e.nblocks);
    arena_.unhold_extents(task_, extents);
  }
  staging_active_ = false;
  staging_to_ = -1;
  staged_.clear();
}

void TaskRuntime::set_grant(int64_t new_bytes) {
  const int64_t bs = arena_.block_size();
  if (new_bytes <= 0 || new_bytes % bs != 0)
    throw std::invalid_argument("grant must be a positive block multiple");
  if (!grant_accepts(new_bytes) || arena_.bytes_owned(task_) > new_bytes)
    throw std::invalid_argument("new grant below live footprint");
  grant_bytes_ = new_bytes;
}

bool TaskRuntime::grant_accepts(int64_t new_bytes) const {
  const int64_t bs = arena_.block_size();
  return new_bytes > 0 && new_bytes % bs == 0 &&
         new_bytes >= arena_.live_bytes(task_);
}

void TaskRuntime::set_worker_hook(WorkerHook hook) {
  std::lock_guard<std::mutex> lk(mu_);
  worker_hook_ = std::move(hook);
}

bool TaskRuntime::inflight_reads(TensorId t) const {
  OpMask readers = g_.readers_of(t);
  for (OpId v : inflight_list_)
    if (readers & (OpMask(1) << v)) return true;
  return false;
}

bool TaskRuntime::inflight_writes(TensorId t) const {
  for (const InFlightRecord& rec : inflight_records_)
    for (size_t i = 0; i < rec.outputs.size(); ++i)
      if (g_.ops()[rec.op].outputs[i] == t) return true;
  return false;
}

bool TaskRuntime::drop_tensor(TensorId t, std::string* why) {
  auto refuse = [&](const char* m) {
    if (why) *why = m;
    return false;
  };
  if (t < 0 || t >= (TensorId)slots_.size()) return refuse("tensor id out of range");
  TensorSlot& s = slots_[t];
  if (!s.live) return refuse("tensor not resident");
  if (g_.tensors()[t].pinned) return refuse("tensor pinned");
  if (inflight_reads(t)) return refuse("tensor read by an in-flight op");
  if (inflight_writes(t)) return refuse("tensor written by an in-flight op");
  arena_.free(s.alloc);
  s.alloc = AllocHandle{};
  s.live = false;
  s.valid = false;
  state_version_ += 1;
  return true;
}

std::vector<int32_t> TaskRuntime::release_tensor(TensorId t, TaskId to) {
  if (t < 0 || t >= (TensorId)slots_.size() || !tensor_resident(t)) return {};
  if (staging_active_ && to != staging_to_) return {};
  if (g_.tensors()[t].pinned || inflight_reads(t) || inflight_writes(t))
    return {};
  const AllocHandle handle = slots_[t].alloc;
  const AllocRecord rec = arena_.record(handle);
  const int32_t first = rec.first_block;
  const int32_t n = rec.nblocks;
  if (staging_active_) {
    staged_.reserve(staged_.size() + 1);
    staged_blocks_.reserve(staged_blocks_.size() + static_cast<size_t>(n));
  }
  std::vector<int32_t> designated;
  designated.reserve(static_cast<size_t>(n));
  for (int32_t b = first; b < first + n; ++b) designated.push_back(b);
  if (staging_active_) {
    arena_.hold_allocation(task_, handle);
    slots_[t].alloc = AllocHandle{};
    slots_[t].live = false;
    slots_[t].valid = false;
    state_version_ += 1;
    staged_.push_back({t, first, n});
    return designated;
  }
  std::vector<int32_t> moved = arena_.transfer_allocation(task_, to, handle);
  slots_[t].alloc = AllocHandle{};
  slots_[t].live = false;
  slots_[t].valid = false;
  state_version_ += 1;
  return moved;
}

bool TaskRuntime::recompute_tensor(TensorId t, std::string* why) {
  auto refuse = [&](const char* m) {
    if (why) *why = m;
    return false;
  };
  if (t < 0 || t >= (TensorId)slots_.size()) return refuse("tensor id out of range");
  if (tensor_valid(t)) return true;  // nothing to do
  OpId p = g_.producer_of(t);
  if (p < 0) return refuse("no producer");
  if (!g_.ops()[p].replayable) return refuse("producer not replayable");
  if (op_state_[p] != OpState::Done) return refuse("producer not done");
  // Inputs of p must be resident to re-run (model rule: RUN needs inputs).
  const OpSpec& op = g_.ops()[p];
  for (TensorId u : op.inputs)
    if (!slots_[u].valid || !slots_[u].live) return refuse("inputs not resident");
  op_state_[p] = OpState::Pending;  // allow re-dispatch
  ++stats_.recomputations;
  if (!try_dispatch(p)) {
    op_state_[p] = OpState::Done;
    return refuse("recompute dispatch failed (memory pressure)");
  }
  wait_op(p);
  const bool ok = tensor_resident(t) && tensor_valid(t);
  if (ok) state_version_ += 1;
  return ok;
}

bool TaskRuntime::required_outputs_ready(std::string* why) const {
  if (worker_error_) {
    if (why) *why = "worker execution failed";
    return false;
  }
  if (!inflight_list_.empty()) {
    if (why) *why = "required-output check observed in-flight work";
    return false;
  }
  for (OpState state : op_state_) {
    if (state != OpState::Done) {
      if (why) *why = "required-output check observed an unfinished operation";
      return false;
    }
  }
  for (TensorId t = 0; t < (TensorId)g_.tensors().size(); ++t) {
    if (g_.tensors()[t].required && !tensor_valid(t)) {
      if (why) *why = "required tensor t" + std::to_string(t) +
                      " is not valid and resident";
      return false;
    }
  }
  return true;
}

bool TaskRuntime::completion_ready(std::string* why) const {
  return required_outputs_ready(why);
}

bool TaskRuntime::restore_tensor(TensorId t, std::vector<char>& visiting,
                                 std::string* why) {
  if (t < 0 || t >= (TensorId)slots_.size()) {
    if (why) *why = "required tensor id is out of range";
    return false;
  }
  if (tensor_valid(t)) return true;
  if (visiting[t]) {
    if (why) *why = "cycle while restoring required output";
    return false;
  }
  visiting[t] = 1;
  const OpId producer = g_.producer_of(t);
  if (producer < 0 || producer >= (OpId)g_.ops().size()) {
    if (why) *why = "required tensor has no valid producer";
    visiting[t] = 0;
    return false;
  }
  const OpSpec& op = g_.ops()[producer];
  if (!op.replayable) {
    if (why) *why = "required tensor t" + std::to_string(t) +
                    " was produced by a non-replayable source";
    visiting[t] = 0;
    return false;
  }
  for (TensorId input : op.inputs) {
    if (!restore_tensor(input, visiting, why)) {
      visiting[t] = 0;
      return false;
    }
  }
  if (op_state_[producer] == OpState::InFlight) {
    try {
      wait_op(producer);
    } catch (...) {
      visiting[t] = 0;
      throw;
    }
  }
  bool ok = false;
  if (op_state_[producer] == OpState::Done) {
    ok = recompute_tensor(t, why);
  } else if (op_state_[producer] == OpState::Pending) {
    if (!try_dispatch(producer)) {
      if (why) *why = "required-output restoration dispatch failed";
      visiting[t] = 0;
      return false;
    }
    wait_op(producer);
    ok = tensor_valid(t);
  }
  visiting[t] = 0;
  return ok;
}

OpState TaskRuntime::op_state(OpId v) const {
  if (v < 0 || v >= (OpId)op_state_.size())
    throw std::invalid_argument("op_state: op id out of range");
  return op_state_[v];
}
bool TaskRuntime::tensor_resident(TensorId t) const {
  return t >= 0 && t < (TensorId)slots_.size() && slots_[t].live;
}
bool TaskRuntime::tensor_valid(TensorId t) const {
  return tensor_resident(t) && slots_[t].valid;
}
const void* TaskRuntime::tensor_ptr(TensorId t) const {
  if (!tensor_valid(t))
    throw std::invalid_argument("tensor_ptr: tensor is not valid");
  return arena_.ptr(slots_[t].alloc);
}
AllocHandle TaskRuntime::tensor_allocation(TensorId t) const {
  if (t < 0 || t >= (TensorId)slots_.size() || !slots_[t].live)
    throw std::invalid_argument("tensor allocation is not resident");
  return slots_[t].alloc;
}

bool TaskRuntime::preflight_continuation(const std::vector<Act>& acts,
                                         size_t start,
                                         int64_t new_grant_bytes,
                                         std::string* why) const {
  auto refuse = [&](const std::string& message) {
    if (why) *why = message;
    return false;
  };
  if (!staging_active_) return refuse("handover staging is not active");
  if (start > acts.size()) return refuse("continuation offset is out of range");
  if (!grant_accepts(new_grant_bytes))
    return refuse("new grant does not cover the live footprint");
  if (staged_.empty()) return refuse("no explicitly released physical extents");

  try {
    BlockArena shadow(arena_);
    std::vector<std::pair<int32_t, int32_t>> extents;
    extents.reserve(staged_.size());
    for (const StagedExtent& e : staged_)
      extents.emplace_back(e.first, e.nblocks);
    shadow.transfer_held_extents(task_, staging_to_, extents);
    if (shadow.bytes_owned(task_) > new_grant_bytes)
      return refuse("post-transfer ownership exceeds the new grant");

    std::vector<TensorSlot> slots = slots_;
    std::vector<OpState> states = op_state_;
    std::vector<InFlightRecord> inflight = inflight_records_;

    auto find_inflight = [&](OpId v) {
      return std::find_if(inflight.begin(), inflight.end(),
                          [v](const InFlightRecord& r) { return r.op == v; });
    };
    auto complete = [&](OpId v) -> bool {
      auto it = find_inflight(v);
      if (it == inflight.end()) return false;
      if (it->scratch.valid()) shadow.free(it->scratch);
      for (TensorId t : g_.ops()[v].outputs) {
        slots[t].valid = true;
        slots[t].live = true;
      }
      states[v] = OpState::Done;
      inflight.erase(it);
      return true;
    };
    auto dispatch = [&](OpId v, bool synchronous) -> bool {
      if (v < 0 || v >= (OpId)g_.ops().size()) return false;
      if (states[v] == OpState::InFlight) return complete(v);
      const OpSpec& op = g_.ops()[v];
      for (TensorId t : op.inputs)
        if (!slots[t].live || !slots[t].valid) return false;
      bool all_outputs_resident = true;
      for (TensorId t : op.outputs)
        if (!slots[t].live || !slots[t].valid) all_outputs_resident = false;
      if (states[v] == OpState::Done && all_outputs_resident) return true;
      if (states[v] == OpState::Done && !op.replayable) return false;
      std::vector<AllocHandle> outputs;
      try {
        for (TensorId t : op.outputs)
          outputs.push_back(shadow.alloc(task_, g_.tensor_bytes(t)));
        AllocHandle scratch;
        if (op.scratch > 0) scratch = shadow.alloc(task_, op.scratch);
        InFlightRecord rec{v, outputs, scratch};
        for (size_t i = 0; i < op.outputs.size(); ++i) {
          slots[op.outputs[i]].alloc = outputs[i];
          slots[op.outputs[i]].live = true;
          slots[op.outputs[i]].valid = false;
        }
        states[v] = OpState::InFlight;
        inflight.push_back(std::move(rec));
      } catch (...) {
        for (AllocHandle h : outputs) {
          try {
            shadow.free(h);
          } catch (...) {
          }
        }
        return false;
      }
      if (synchronous) return complete(v);
      return true;
    };
    auto drop = [&](TensorId t) -> bool {
      if (t < 0 || t >= (TensorId)slots.size() || !slots[t].live ||
          g_.tensors()[t].pinned)
        return false;
      for (const InFlightRecord& r : inflight) {
        for (TensorId input : g_.ops()[r.op].inputs)
          if (input == t) return false;
        for (TensorId output : g_.ops()[r.op].outputs)
          if (output == t) return false;
      }
      shadow.free(slots[t].alloc);
      slots[t].alloc = AllocHandle{};
      slots[t].live = false;
      slots[t].valid = false;
      return true;
    };

    for (size_t i = start; i < acts.size(); ++i) {
      const Act& a = acts[i];
      switch (a.kind) {
        case ActKind::Run:
          if (!dispatch(a.target, true))
            return refuse("continuation RUN is not physically allocatable");
          break;
        case ActKind::Launch:
          if (!dispatch(a.target, false))
            return refuse("continuation LAUNCH is not physically allocatable");
          break;
        case ActKind::Wait:
          if (!complete(a.target))
            return refuse("continuation WAIT has no physical in-flight record");
          break;
        case ActKind::Drop:
          if (!drop(a.target)) return refuse("continuation DROP is not legal");
          break;
        case ActKind::Release:
          return refuse("continuation contains RELEASE after COMMIT");
        case ActKind::Commit:
          return refuse("continuation contains a second COMMIT");
      }
      if (shadow.live_bytes(task_) > new_grant_bytes)
        return refuse("continuation live footprint exceeds the new grant");
      std::string shadow_error;
      if (!shadow.verify(&shadow_error))
        return refuse("shadow arena invariant failed: " + shadow_error);
    }
    if (shadow.bytes_owned(task_) > new_grant_bytes)
      return refuse("continuation ownership exceeds the new grant");
    return true;
  } catch (const std::exception& e) {
    return refuse(std::string("physical preflight failed: ") + e.what());
  }
}

std::string TaskRuntime::frontier_error() const {
  // Coherence of a captured frontier (SPEC v1.1 §3, runtime form).
  for (TensorId t = 0; t < (TensorId)g_.tensors().size(); ++t) {
    if (tensor_resident(t) && !tensor_valid(t)) {
      OpId p = g_.producer_of(t);
      bool in_flight = false;
      for (OpId v : inflight_list_)
        if (v == p) in_flight = true;
      if (!in_flight) return "resident tensor without completed producer";
    }
  }
  return "";
}

bool TaskRuntime::invariants_hold(std::string* err) const {
  if (controlled_bytes() > grant_bytes_) {
    if (err) *err = "controlled bytes exceed grant";
    return false;
  }
  try {
    if (arena_.bytes_owned(task_) > grant_bytes_) {
      if (err) *err = "owned blocks exceed grant";
      return false;
    }
  } catch (const std::exception& e) {
    if (err) *err = e.what();
    return false;
  }
  std::string aerr;
  if (!arena_.verify(&aerr)) {
    if (err) *err = "arena: " + aerr;
    return false;
  }
  return true;
}

void TaskRuntime::load_source(OpId source_op, uint64_t seed) {
  if (source_op < 0 || source_op >= (OpId)g_.ops().size())
    throw std::invalid_argument("load_source: op id out of range");
  const OpSpec& op = g_.ops()[source_op];
  if (op.kind != OpKind::Source || op.outputs.size() != 1)
    throw std::invalid_argument("load_source expects a source op");
  if (source_initialized_[source_op])
    throw std::logic_error("load_source: source has already been initialized");
  TensorId t = op.outputs[0];
  if (!tensor_resident(t)) {
    if (op_state_[source_op] == OpState::Pending) {
      if (!try_dispatch(source_op, true))
        throw std::runtime_error("cannot allocate source output");
      wait_op(source_op);
    } else if (op_state_[source_op] == OpState::InFlight) {
      wait_op(source_op);
    } else {
      // The source op may have been dispatched by a general scheduler before
      // its external bytes were supplied.  Its completion leaves an
      // allocation but no valid value, so reserve a fresh output without
      // executing the non-replayable source a second time.
      AllocHandle h = arena_.alloc(task_, g_.tensor_bytes(t));
      slots_[t].alloc = h;
      slots_[t].live = true;
      slots_[t].valid = false;
    }
  }
  if (inflight_reads(t) || inflight_writes(t))
    throw std::logic_error("load_source would race with an in-flight operation");
  // Overwrite with the deterministic pattern (bit-exact with the reference).
  auto* p = static_cast<double*>(arena_.ptr(slots_[t].alloc));
  const int64_t n = g_.tensor_bytes(t) / 8;
  for (int64_t i = 0; i < n; ++i) p[i] = source_value(seed, (uint64_t)i);
  slots_[t].valid = true;
  slots_[t].version += 1;
  source_initialized_[source_op] = 1;
  state_version_ += 1;
}

std::vector<std::vector<double>> reference_execute_f64(
    const RuntimeGraph& g, const std::vector<uint64_t>& source_seeds) {
  const_cast<RuntimeGraph&>(g).finish();  // graphs are finished once, idempotent
  std::vector<std::vector<double>> vals(g.tensors().size());
  auto ref_numel = [](const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) {
      if (d <= 0 || __builtin_mul_overflow(n, d, &n))
        throw std::invalid_argument("reference: invalid tensor shape");
    }
    return n;
  };
  auto ref_broadcast_index = [&](int64_t linear,
                                 const std::vector<int64_t>& out_shape,
                                 const std::vector<int64_t>& in_shape) {
    int64_t result = 0;
    int64_t stride = 1;
    const size_t offset = out_shape.size() - in_shape.size();
    for (size_t axis = out_shape.size(); axis-- > 0;) {
      const int64_t coordinate = linear % out_shape[axis];
      linear /= out_shape[axis];
      if (axis < offset) continue;
      const size_t input_axis = axis - offset;
      const int64_t input_coordinate = in_shape[input_axis] == 1
                                           ? 0
                                           : coordinate;
      result += input_coordinate * stride;
      stride *= in_shape[input_axis];
    }
    return result;
  };
  for (OpId v : g.topo()) {
    const OpSpec& op = g.ops()[v];
    for (TensorId t : op.outputs)
      vals[t].assign((size_t)ref_numel(g.tensors()[t].shape), 0.0);
    if (op.kind == OpKind::Source) {
      TensorId t = op.outputs[0];
      uint64_t seed = (v < (OpId)source_seeds.size()) ? source_seeds[v] : 42;
      for (size_t i = 0; i < vals[t].size(); ++i)
        vals[t][i] = source_value(seed, i);
      continue;
    }
    const TensorId out_id = op.outputs[0];
    const std::vector<int64_t>& out_shape = g.tensors()[out_id].shape;
    std::vector<double>& out = vals[out_id];
    const std::vector<double>& a = vals[op.inputs[0]];
    const std::vector<double>& b = op.inputs.size() > 1 ? vals[op.inputs[1]] : a;
    switch (op.kind) {
      case OpKind::Add:
      case OpKind::Mul:
      case OpKind::ResidualAdd: {
        const bool binary = op.inputs.size() == 2;
        const auto& ashape = g.tensors()[op.inputs[0]].shape;
        const auto& bshape = binary ? g.tensors()[op.inputs[1]].shape : ashape;
        for (int64_t i = 0; i < ref_numel(out_shape); ++i) {
          const int64_t ia = binary ? ref_broadcast_index(i, out_shape, ashape) : i;
          const int64_t ib = binary ? ref_broadcast_index(i, out_shape, bshape) : i;
          out[(size_t)i] = op.kind == OpKind::Mul ? a[(size_t)ia] * b[(size_t)ib]
                                                 : a[(size_t)ia] + b[(size_t)ib];
        }
        break;
      }
      case OpKind::MatMul:
      case OpKind::ScratchGemm: {
        const int64_t m = out_shape[0];
        const int64_t n = out_shape[1];
        const int64_t k = g.tensors()[op.inputs[0]].shape[1];
        // The reference uses the direct textbook loop. In particular it does
        // not use the production scratch packing path.
        for (int64_t i = 0; i < m; ++i)
          for (int64_t j = 0; j < n; ++j) {
            volatile double sum = 0.0;
            for (int64_t p = 0; p < k; ++p)
              sum += a[(size_t)(i * k + p)] * b[(size_t)(p * n + j)];
            out[(size_t)(i * n + j)] = sum;
          }
        break;
      }
      case OpKind::ReduceSum: {
        const int64_t m = out_shape[0];
        const int64_t k = g.tensors()[op.inputs[0]].shape[1];
        for (int64_t i = 0; i < m; ++i) {
          volatile double sum = 0.0;
          for (int64_t j = 0; j < k; ++j) sum += a[(size_t)(i * k + j)];
          out[(size_t)i] = sum;
        }
        break;
      }
      case OpKind::Source:
        break;
    }
  }
  return vals;
}

}  // namespace keel::rt
