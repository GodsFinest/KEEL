#include <keel/checker.hpp>

#include <stdexcept>

namespace keel {

namespace {

// Checker-local live state. Deliberately NOT keel::State: independent fields,
// independent arithmetic, so model.cpp bugs cannot correlate.
struct CState {
  std::vector<char> op_done;
  std::vector<char> op_inflight;
  std::vector<char> t_resident;
  int64_t released_bytes = 0;
  int64_t prep_used = 0;
  bool committed = false;
  bool committed_seen = false;  // COMMIT appeared in the trace
  OpMask done_at_commit = 0;    // for the recompute metric
};

[[noreturn]] void fail(CheckResult& res, int idx, const std::string& why) {
  res.ok = false;
  res.error_index = idx;
  res.error = why;
  throw res;
}

// Checker-local usage (SPEC v1.1 §4): resident bytes PLUS each in-flight
// op's live reservation (scratch + output bytes) until completion. This is
// re-derived here independently of keel/model.cpp (audit findings 1 & 2): the
// checker must not import the model's transition implementation.
// Throws on overflow: the checker never saturates silently (audit finding 3:
// no suppression, no undocumented saturation). Callers convert the throw into
// a rejected witness with an explicit message.
int64_t usage_of(const Graph& g, const CState& s) {
  int64_t u = 0;
  for (int t = 0; t < (int)g.tensors.size(); ++t)
    if (s.t_resident[t] &&
        __builtin_add_overflow(u, g.tensors[t].size, &u))
      throw std::overflow_error("checker: resident byte sum overflows");
  for (int v = 0; v < (int)g.ops.size(); ++v) {
    if (!s.op_inflight[v]) continue;
    if (__builtin_add_overflow(u, g.ops[v].scratch, &u))
      throw std::overflow_error("checker: scratch reservation overflows");
    for (TensorId t : g.ops[v].outputs)
      if (__builtin_add_overflow(u, g.tensors[t].size, &u))
        throw std::overflow_error("checker: output reservation overflows");
  }
  return u;
}

int64_t ceiling_of([[maybe_unused]] const Graph& g, const CState& s,
                   const Request& r) {
  // SPEC §4: the shrinking region is the ceiling in both phases.
  int64_t ceiling = 0;
  if (__builtin_sub_overflow(r.Bo, s.released_bytes, &ceiling))
    throw std::overflow_error("checker: ceiling arithmetic overflows");
  return ceiling;
}

bool resident_all(const CState& s, const std::vector<TensorId>& ts) {
  for (TensorId t : ts)
    if (!s.t_resident[t]) return false;
  return true;
}

bool any_output_resident(const Graph& g, const CState& s, OpId v) {
  for (TensorId t : g.ops[v].outputs)
    if (s.t_resident[t]) return true;
  return false;
}

bool reads_protected(const Graph& g, const CState& s, TensorId t) {
  for (int v = 0; v < (int)g.ops.size(); ++v)
    if (s.op_inflight[v])
      for (TensorId u : g.ops[v].inputs)
        if (u == t) return true;
  return false;
}

}  // namespace

CheckResult check_witness(const Graph& g, const State& initial,
                          const Request& r, const Witness& w) {
  CheckResult res;
  auto bad_req = [](const char* msg) {
    return CheckResult{.ok = false,
                       .reached_goal = false,
                       .error_index = -1,
                       .error = msg};
  };
  if (r.Bo <= 0) return bad_req("Bo must be positive");
  if (r.Bn < 0 || r.Bn >= r.Bo) return bad_req("Bn out of range vs Bo");
  if (r.K < 0) return bad_req("K must be non-negative");

  CState s;
  s.op_done.assign(g.ops.size(), 0);
  s.op_inflight.assign(g.ops.size(), 0);
  s.t_resident.assign(g.tensors.size(), 0);
  for (int v = 0; v < (int)g.ops.size(); ++v)
    s.op_done[v] = (initial.done >> v) & 1 ? 1 : 0;
  for (int v = 0; v < (int)g.ops.size(); ++v)
    s.op_inflight[v] = (initial.inflight >> v) & 1 ? 1 : 0;
  for (int t = 0; t < (int)g.tensors.size(); ++t)
    s.t_resident[t] = (initial.resident >> t) & 1 ? 1 : 0;
  s.released_bytes = initial.released_bytes;
  s.prep_used = initial.prep_used;
  s.committed = initial.committed;
  s.committed_seen = initial.committed;

  const int nops = static_cast<int>(g.ops.size());
  const int nts = static_cast<int>(g.tensors.size());
  const OpMask valid_ops =
      nops == 64 ? ~OpMask(0)
                 : (nops == 0 ? OpMask(0) : (OpMask(1) << nops) - 1);
  const TensorMask valid_tensors =
      nts == 64 ? ~TensorMask(0)
                : (nts == 0 ? TensorMask(0) : (TensorMask(1) << nts) - 1);
  if ((initial.done & ~valid_ops) != 0 ||
      (initial.inflight & ~valid_ops) != 0)
    return bad_req("initial state contains an out-of-range operation bit");
  if ((initial.resident & ~valid_tensors) != 0)
    return bad_req("initial state contains an out-of-range tensor bit");
  if (initial.released_bytes < 0 || initial.released_bytes > r.Bo)
    return bad_req("initial released bytes out of range");
  if (initial.prep_used < 0)
    return bad_req("initial preparation usage is negative");
  if (!initial.committed && initial.prep_used > r.K)
    return bad_req("initial preparation usage exceeds K");
  if (initial.committed && initial.released_bytes < r.Bo - r.Bn)
    return bad_req("initial committed state does not satisfy release target");

  // Validate the initial state itself (the checker is also the tool that
  // proves frontier states are legal): invariant + structural sanity.
  {
    int64_t u = 0, c = 0;
    try {
      u = usage_of(g, s);
      c = ceiling_of(g, s, r);
    } catch (const std::overflow_error& e) {
      res.error = e.what();
      return res;
    }
    if (u > c) {
      res.error = "initial state violates memory invariant";
      return res;
    }
    for (int t = 0; t < (int)g.tensors.size(); ++t)
      if (s.t_resident[t]) {
        OpId p = g.tensors[t].producer;
        if (p < 0 || p >= (int)g.ops.size() || !s.op_done[p]) {
          res.error = "initial state: resident tensor without completed producer";
          return res;
        }
      }
    for (int v = 0; v < (int)g.ops.size(); ++v)
      if (s.op_inflight[v]) {
        if (s.op_done[v]) {
          res.error = "initial state: op both done and in flight";
          return res;
        }
        for (TensorId t : g.ops[v].inputs)
          if (!s.t_resident[t]) {
            res.error = "initial state: in-flight op missing its input";
            return res;
          }
        for (TensorId t : g.ops[v].outputs)
          if (s.t_resident[t]) {
            res.error = "initial state: in-flight op output already resident";
            return res;
          }
      }
  }

  res.ok = true;  // provisionally; any failure sets it false

  // Full invariant after EVERY action (SPEC v1.1 §4; audit finding 2):
  // checked via the lambda below at the end of each case.
  auto check_invariant = [&](int idx) {
    int64_t u = usage_of(g, s);
    int64_t c = ceiling_of(g, s, r);
    if (u > c) fail(res, idx, "memory invariant violated after this action");
  };

  try {
    for (int i = 0; i < (int)w.acts.size(); ++i) {
      const Act& a = w.acts[i];
      switch (a.kind) {
        case ActKind::Run:
        case ActKind::Launch: {
          const bool async = a.kind == ActKind::Launch;
          OpId v = a.target;
          if (v < 0 || v >= (int)g.ops.size())
            fail(res, i, "op id out of range");
          if (s.op_inflight[v])
            fail(res, i, "RUN/LAUNCH of in-flight op");
          if (!resident_all(s, g.ops[v].inputs))
            fail(res, i, "inputs not resident");
          if (any_output_resident(g, s, v))
            fail(res, i, "output already resident (no partial re-materialization)");
          if (s.op_done[v] && !g.ops[v].replayable)
            fail(res, i, "non-replayable op executed twice");
          // Memory: usage + outputs + scratch <= ceiling (SPEC §5).
          int64_t u = usage_of(g, s);
          int64_t need = u;
          for (TensorId t : g.ops[v].outputs)
            if (__builtin_add_overflow(need, g.tensors[t].size, &need))
              fail(res, i, "op byte accounting overflows");
          if (__builtin_add_overflow(need, g.ops[v].scratch, &need))
            fail(res, i, "op scratch accounting overflows");
          if (need > ceiling_of(g, s, r))
            fail(res, i, "op exceeds memory ceiling");
          if (!s.committed) {
            int64_t used = 0;
            if (__builtin_add_overflow(s.prep_used, g.ops[v].cost, &used))
              fail(res, i, "preparation accounting overflows");
            if (used > r.K) fail(res, i, "preparation budget exceeded");
            s.prep_used = used;
            if (__builtin_add_overflow(res.prep_compute, g.ops[v].cost,
                                       &res.prep_compute))
              fail(res, i, "preparation metric overflows");
          } else {
            if (__builtin_add_overflow(res.cont_compute, g.ops[v].cost,
                                       &res.cont_compute))
              fail(res, i, "continuation metric overflows");
            if (s.done_at_commit & (OpMask(1) << v))
              if (__builtin_add_overflow(res.recompute_compute, g.ops[v].cost,
                                         &res.recompute_compute))
                fail(res, i, "recomputation metric overflows");
          }
          if (async) {
            s.op_inflight[v] = 1;
          } else {
            s.op_done[v] = 1;
            for (TensorId t : g.ops[v].outputs) s.t_resident[t] = 1;
          }
          check_invariant(i);
          break;
        }
        case ActKind::Wait: {
          OpId v = a.target;
          if (v < 0 || v >= (int)g.ops.size())
            fail(res, i, "op id out of range");
          if (!s.op_inflight[v]) fail(res, i, "WAIT of op not in flight");
          s.op_inflight[v] = 0;
          s.op_done[v] = 1;
          for (TensorId t : g.ops[v].outputs) s.t_resident[t] = 1;
          check_invariant(i);
          break;
        }
        case ActKind::Drop: {
          TensorId t = a.target;
          if (t < 0 || t >= (int)g.tensors.size())
            fail(res, i, "tensor id out of range");
          if (!s.t_resident[t]) fail(res, i, "DROP of non-resident tensor");
          if (g.tensors[t].pinned) fail(res, i, "DROP of pinned tensor");
          if (reads_protected(g, s, t))
            fail(res, i, "DROP of tensor read by an in-flight op");
          s.t_resident[t] = 0;
          check_invariant(i);
          break;
        }
        case ActKind::Release: {
          TensorId t = a.target;
          if (t < 0 || t >= (int)g.tensors.size())
            fail(res, i, "tensor id out of range");
          if (s.committed)
            fail(res, i, "RELEASE after COMMIT (Phase 1: one handover)");
          if (!s.t_resident[t])
            fail(res, i, "RELEASE of block not currently held");
          if (g.tensors[t].pinned) fail(res, i, "RELEASE of pinned tensor");
          if (reads_protected(g, s, t))
            fail(res, i, "RELEASE of tensor read by an in-flight op");
          s.t_resident[t] = 0;
          if (__builtin_add_overflow(s.released_bytes, g.tensors[t].size,
                                     &s.released_bytes))
            fail(res, i, "released-byte accounting overflows");
          res.released_bytes = s.released_bytes;
          check_invariant(i);
          break;
        }
        case ActKind::Commit: {
          if (s.committed_seen) fail(res, i, "more than one COMMIT");
          if (s.released_bytes < r.Bo - r.Bn)
            fail(res, i, "COMMIT shortfall: released < Bo - Bn");
          s.committed = true;
          s.committed_seen = true;
          s.done_at_commit = 0;
          for (int v = 0; v < (int)g.ops.size(); ++v)
            if (s.op_done[v]) s.done_at_commit |= (OpMask(1) << v);
          check_invariant(i);
          break;
        }
        default:
          fail(res, i, "unknown action kind");
      }
    }
  } catch (CheckResult& failure) {
    return failure;
  } catch (const std::overflow_error& e) {
    res.ok = false;
    res.error = e.what();
    return res;
  }

  // Goal condition (SPEC §5): committed, nothing in flight, all Q resident.
  bool goal = s.committed;
  for (int v = 0; v < (int)g.ops.size() && goal; ++v)
    if (s.op_inflight[v]) goal = false;
  for (int t = 0; t < (int)g.tensors.size() && goal; ++t)
    if (g.tensors[t].required && !s.t_resident[t]) goal = false;
  res.reached_goal = goal;

  // Final-state memory invariant must also hold (it held at every step).
  if (usage_of(g, s) > ceiling_of(g, s, r)) {
    res.ok = false;
    res.error = "final state violates memory invariant";
  }
  return res;
}

}  // namespace keel
