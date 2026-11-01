#include <keel/policy.hpp>

#include <keel/checker.hpp>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <queue>
#include <unordered_map>

namespace keel {

const char* policy_status_name(PolicyStatus s) {
  switch (s) {
    case PolicyStatus::Accept: return "ACCEPT";
    case PolicyStatus::Reject: return "REJECT";
    case PolicyStatus::Failed: return "FAILED";
    case PolicyStatus::Unknown: return "UNKNOWN";
  }
  return "?";
}

PolicyStatus policy_status_from_oracle(OracleStatus s) {
  switch (s) {
    case OracleStatus::Feasible: return PolicyStatus::Accept;
    case OracleStatus::Infeasible: return PolicyStatus::Reject;
    case OracleStatus::Unknown: return PolicyStatus::Unknown;
  }
  return PolicyStatus::Unknown;
}

LastUse compute_last_use(const Graph& g) {
  LastUse lu(g.tensors.size(), -1);
  for (size_t i = 0; i < g.topo.size(); ++i) {
    OpId v = g.topo[i];
    for (TensorId t : g.ops[v].inputs) lu[t] = (int)i;
  }
  for (int t = 0; t < (int)g.tensors.size(); ++t)
    if (g.tensors[t].required) lu[t] = (int)g.topo.size();  // needed at the end
  return lu;
}

namespace {

struct Sized { TensorId t; int64_t size; int last_use; };

std::vector<TensorId> drop_order(const Graph& g, const State& s, int order,
                                 const LastUse& lu) {
  std::vector<Sized> cands;
  for (int t = 0; t < (int)g.tensors.size(); ++t) {
    if (!(s.resident & (TensorMask(1) << t))) continue;
    if (g.tensors[t].required) continue;      // never surrender Q eagerly
    if (g.tensors[t].pinned) continue;
    if (can_drop(g, s, t).err == Err::PROTECTED_BY_INFLIGHT) continue;
    cands.push_back({t, g.tensors[t].size, lu[t]});
  }
  switch (order) {
    case 0:  // largest first
      std::sort(cands.begin(), cands.end(), [](auto& a, auto& b) {
        return a.size != b.size ? a.size > b.size : a.t < b.t;
      });
      break;
    case 1:  // last use farthest first (end-of-life first)
      std::sort(cands.begin(), cands.end(), [](auto& a, auto& b) {
        return a.last_use != b.last_use ? a.last_use < b.last_use : a.t < b.t;
      });
      break;
    default:  // smallest first
      std::sort(cands.begin(), cands.end(), [](auto& a, auto& b) {
        return a.size != b.size ? a.size < b.size : a.t < b.t;
      });
      break;
  }
  std::vector<TensorId> out;
  for (auto& c : cands) out.push_back(c.t);
  return out;
}

// Prep ladder: RELEASE droppable tensors in the given order until COMMIT is
// legal. Every action is individually legal, so an aborted ladder is safe.
struct Ladder {
  bool commitable;
  std::vector<Act> acts;
  State state;
};

Ladder build_ladder(const Graph& g, const State& s0, const Request& r,
                    int order, const LastUse& lu) {
  Ladder L;
  L.state = s0;
  const int64_t target = r.Bo - r.Bn;
  for (TensorId t : drop_order(g, L.state, order, lu)) {
    if (L.state.released_bytes >= target) break;
    if (!can_release(g, L.state, t).ok) continue;
    L.acts.push_back({ActKind::Release, t});
    L.state = apply_release(g, L.state, t);
  }
  L.commitable = can_commit(g, L.state, r).ok;
  return L;
}

// Greedy continuation executor from a COMMITTED state: run remaining ops in
// topo order, recompute dropped inputs (recursively), evict on pressure,
// drop after last use. Deterministic. May FAIL (that is the point of greedy).
struct ExecResult {
  bool ok;
  std::vector<Act> acts;
  size_t steps = 0;
};

ExecResult greedy_execute(const Graph& g, State s, const Request& r,
                          const LastUse& lu) {
  ExecResult out;
  out.ok = false;
  auto emit_drop = [&](TensorId t) {
    out.acts.push_back({ActKind::Drop, t});
    s = apply_drop(g, s, t);
  };

  // Async frontiers: drain outstanding completions first (WAIT is free and
  // memory was reserved at LAUNCH, so no legality risk; SPEC §5).
  for (OpId v = 0; v < (OpId)g.ops.size(); ++v) {
    if (s.inflight & (OpMask(1) << v)) {
      out.acts.push_back({ActKind::Wait, v});
      s = apply_wait(g, s, v);
      ++out.steps;
    }
  }

  // position of each op in topo order
  std::vector<int> pos(g.ops.size(), -1);
  for (size_t i = 0; i < g.topo.size(); ++i) pos[g.topo[i]] = (int)i;

  for (size_t i = 0; i < g.topo.size(); ++i) {
    OpId v = g.topo[i];
    if (s.done & (OpMask(1) << v)) continue;
    // Recursively ensure inputs are resident (recompute dropped ones).
    // Iterative post-order over the producer DAG of missing inputs.
    std::vector<OpId> rerun;
    OpMask rerun_seen = 0;
    std::vector<TensorId> stack;
    for (TensorId t : g.ops[v].inputs)
      if (!(s.resident & (TensorMask(1) << t))) stack.push_back(t);
    bool feasible = true;
    while (!stack.empty()) {
      TensorId t = stack.back();
      stack.pop_back();
      if (s.resident & (TensorMask(1) << t)) continue;
      OpId p = g.tensors[t].producer;
      if (!(s.done & (OpMask(1) << p))) {
        // producer not run yet: it appears later in topo order; the main loop
        // will reach it. Defer: this input will be produced naturally.
        continue;
      }
      if (!g.ops[p].replayable) {
        feasible = false;
        break;
      }
      if (!(rerun_seen & (OpMask(1) << p))) {
        rerun_seen |= (OpMask(1) << p);
        rerun.push_back(p);
      }
      for (TensorId u : g.ops[p].inputs)
        if (!(s.resident & (TensorMask(1) << u))) stack.push_back(u);
    }
    if (!feasible) return out;  // permanently lost input: continuation fails

    // Make room for an op: evict largest droppable non-required tensors,
    // never the op's own inputs. Returns false when nothing fits.
    auto make_room = [&](OpId op, TensorMask protect_extra) {
      TensorMask protect = g.inputs_mask(op) | protect_extra;
      while (true) {
        Legality L = can_run(g, s, r, op);
        if (L.ok) return true;
        if (L.err != Err::OVER_CEILING) return false;
        std::vector<TensorId> evict;
        for (int t = 0; t < (int)g.tensors.size(); ++t) {
          TensorMask bit = TensorMask(1) << t;
          if ((s.resident & bit) && !(protect & bit) && !g.tensors[t].required &&
              can_drop(g, s, t).ok)
            evict.push_back(t);
        }
        if (evict.empty()) return false;
        TensorId victim = *std::max_element(
            evict.begin(), evict.end(), [&](TensorId a, TensorId b) {
              return g.tensors[a].size < g.tensors[b].size;
            });
        emit_drop(victim);
        ++out.steps;
      }
    };

    // Re-run producers in reverse discovery order (producer-side first).
    for (auto it = rerun.rbegin(); it != rerun.rend(); ++it) {
      OpId p = *it;
      // Protect inputs of ALL pending re-runs plus v: they coexist at peak.
      TensorMask prot = g.inputs_mask(v);
      for (OpId q : rerun)
        if (q != p) prot |= g.inputs_mask(q);
      if (!make_room(p, prot)) return out;
      out.acts.push_back({ActKind::Run, p});
      s = apply_run(g, s, r, p);
      ++out.steps;
    }

    if (!make_room(v, 0)) return out;
    out.acts.push_back({ActKind::Run, v});
    s = apply_run(g, s, r, v);
    ++out.steps;

    // Drop after last use (never required outputs).
    for (int t = 0; t < (int)g.tensors.size(); ++t)
      if ((s.resident & (TensorMask(1) << t)) && !g.tensors[t].required &&
          lu[t] >= 0 && lu[t] <= (int)i && can_drop(g, s, t).ok)
        emit_drop(t);
  }

  // Final sweep: ensure all required outputs are resident (they are, by the
  // no-required-drop discipline, unless a re-derivation eviction... we never
  // evict required, so goal holds iff committed and nothing in flight).
  out.ok = is_goal(g, s);
  return out;
}

namespace {
int64_t now_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
}  // namespace

// Validates a candidate trace with the independent checker, recording the
// invocation and its wall-clock into the outcome (audit finding 4).
CheckResult validate_counted(const Graph& g, const State& s0, const Request& r,
                             const Witness& w, PolicyOutcome& o) {
  int64_t t0 = now_us();
  CheckResult c = check_witness(g, s0, r, w);
  ++o.checker_invocations;
  o.checker_us += now_us() - t0;
  return c;
}

void fill_metrics(PolicyOutcome& o, const CheckResult& c) {
  o.released_bytes = c.released_bytes;
  o.prep_compute = c.prep_compute;
  o.cont_compute = c.cont_compute;
  o.recompute_compute = c.recompute_compute;
}

PolicyOutcome finish_accepted(const Graph& g, const State& s0, const Request& r,
                              std::vector<Act> acts,
                              const PolicyOutcome& work,
                              const std::string& note) {
  PolicyOutcome o = work;
  o.status = PolicyStatus::Accept;
  o.witness = Witness{std::move(acts)};
  o.planning_nodes = work.outer_expansions + work.cont_expansions_total;
  o.note = note;
  CheckResult c = validate_counted(g, s0, r, o.witness, o);
  if (!c.ok || !c.reached_goal) {
    // Should never happen for honest policies; surface as a defect.
    o.status = PolicyStatus::Failed;
    o.note += " [DEFECT: policy-emitted witness failed validation: " +
              (c.error.empty() ? "goal not reached" : c.error) + "]";
  }
  fill_metrics(o, c);
  return o;
}

}  // namespace

// ---------------------------------------------------------------------------
// Conservative: zero-compute prep; prove the continuation with a bounded
// exact search; any failure or UNKNOWN -> REJECT (SPEC §10).

PolicyOutcome conservative(const Graph& g, const State& s0, const Request& r,
                           const PolicyConfig& cfg) {
  PolicyOutcome o;
  LastUse lu = compute_last_use(g);
  Ladder L = build_ladder(g, s0, r, 0, lu);
  if (!L.commitable) {
    o.note = "drop-only prep cannot reach the release target";
    o.planning_nodes = (size_t)L.acts.size();
    return o;
  }
  State committed = apply_commit(g, L.state, r);
  OracleConfig oc;
  oc.max_expansions = cfg.continuation_cap;
  oc.allow_async = committed.inflight != 0;
  OracleResult cont = solve_continuation(g, committed, r, oc);
  o.cont_expansions_total = cont.expansions;
  o.candidates_checked = 1;
  o.planning_nodes = cont.expansions;
  if (cont.status == OracleStatus::Feasible) {
    std::vector<Act> acts = L.acts;
    acts.push_back({ActKind::Commit, -1});
    for (const Act& a : cont.witness.acts) acts.push_back(a);
    PolicyOutcome work;
    work.outer_expansions = 0;
    work.cont_expansions_total = cont.expansions;
    work.candidates_checked = 1;
    return finish_accepted(g, s0, r, std::move(acts), work,
                           "drop-only prep + exact continuation");
  }
  o.note = cont.status == OracleStatus::Infeasible
               ? "continuation proven infeasible under Bn"
               : "continuation search budget exhausted";
  return o;
}

// ---------------------------------------------------------------------------
// Greedy: commit the largest-first ladder, then run the greedy continuation.
// May FAIL after committing — reported, never hidden.

PolicyOutcome greedy(const Graph& g, const State& s0, const Request& r,
                     const PolicyConfig& cfg, int order) {
  (void)cfg;  // greedy is constructive: no search budget to enforce
  PolicyOutcome o;
  LastUse lu = compute_last_use(g);
  Ladder L = build_ladder(g, s0, r, order, lu);
  if (!L.commitable) {
    o.note = "greedy prep cannot reach the release target";
    return o;
  }
  State committed = apply_commit(g, L.state, r);
  ExecResult ex = greedy_execute(g, committed, r, lu);
  o.executor_steps = ex.steps;
  std::vector<Act> acts = L.acts;
  acts.push_back({ActKind::Commit, -1});
  for (const Act& a : ex.acts) acts.push_back(a);
  o.witness = Witness{std::move(acts)};
  CheckResult c = validate_counted(g, s0, r, o.witness, o);
  fill_metrics(o, c);
  if (c.ok && c.reached_goal) {
    o.status = PolicyStatus::Accept;
    o.note = "greedy ladder + greedy continuation";
  } else if (c.ok) {
    o.status = PolicyStatus::Failed;  // committed, then could not finish
    o.note = "committed but continuation failed (unguarded reclamation)";
  } else {
    o.status = PolicyStatus::Failed;
    o.note = "trace illegal: " + c.error;  // defect; harness flags it
  }
  return o;
}

// ---------------------------------------------------------------------------
// Guarded greedy: three deterministic orders; validate before acknowledging.

PolicyOutcome guarded_greedy(const Graph& g, const State& s0, const Request& r,
                             const PolicyConfig& cfg) {
  (void)cfg;  // guarded greedy is constructive: no search budget to enforce
  PolicyOutcome o;
  LastUse lu = compute_last_use(g);
  const char* names[3] = {"largest-first", "end-of-life-first",
                          "smallest-first"};
  PolicyOutcome work;
  for (int order = 0; order < 3; ++order) {
    Ladder L = build_ladder(g, s0, r, order, lu);
    if (!L.commitable) continue;
    State committed = apply_commit(g, L.state, r);
    ExecResult ex = greedy_execute(g, committed, r, lu);
    work.executor_steps += ex.steps + L.acts.size();
    if (!ex.ok) continue;
    std::vector<Act> acts = L.acts;
    acts.push_back({ActKind::Commit, -1});
    for (const Act& a : ex.acts) acts.push_back(a);
    // Cheap pre-validation to decide acceptance (counted separately from the
    // final validation inside finish_accepted).
    CheckResult c = validate_counted(g, s0, r, Witness{acts}, work);
    if (c.ok && c.reached_goal) {
      work.candidates_checked = work.checker_invocations;
      return finish_accepted(g, s0, r, std::move(acts), work,
                             std::string("guarded(") + names[order] + ")");
    }
  }
  o.status = PolicyStatus::Reject;
  o.executor_steps = work.executor_steps;
  o.checker_invocations = work.checker_invocations;
  o.checker_us = work.checker_us;
  o.note = "no deterministic order yields a validated handover";
  return o;
}

// ---------------------------------------------------------------------------
// KEEL bounded planner: best-first search over prep states (score = released
// bytes minus prep compute), each COMMIT-able candidate continuation-checked
// with a bounded exact search. Prep may RUN ops to unlock releases — the
// capability drop-only baselines lack. Statuses per SPEC §10: never claims
// proven infeasibility.

namespace {

// Candidate-state deduplication (SPEC §8 canonical key, planner-side): the
// same prep state reached via different action orders must be expanded once,
// or duplicate paths flood the budget (observed: 20k expansions over 729
// distinct states on a 6-tensor instance).
struct PrepKey {
  uint64_t done, resident, inflight;
  int64_t released, prep_key;
  bool committed;
  bool operator==(const PrepKey& o) const {
    return done == o.done && resident == o.resident && inflight == o.inflight &&
           released == o.released && prep_key == o.prep_key &&
           committed == o.committed;
  }
};
struct PrepKeyHash {
  size_t operator()(const PrepKey& k) const noexcept {
    uint64_t h = 0x9E3779B97F4A7C15ull;
    auto mix = [&h](uint64_t x) {
      h ^= x + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    };
    mix(k.done);
    mix(k.resident);
    mix(k.inflight);
    mix((uint64_t)k.released);
    mix((uint64_t)k.prep_key);
    mix(k.committed ? 1u : 0u);
    return (size_t)h;
  }
};
PrepKey prep_key_of(const State& s, const Request& r) {
  return PrepKey{s.done, s.resident, s.inflight, s.released_bytes,
                 s.committed ? 0 : std::min(s.prep_used, r.K), s.committed};
}

}  // namespace

PolicyOutcome keel_plan(const Graph& g, const State& s0, const Request& r,
                        const PolicyConfig& cfg) {
  PolicyOutcome o;
  LastUse lu = compute_last_use(g);

  struct Cand {
    int64_t score;
    uint64_t seq;
    State s;
    std::vector<Act> acts;
    int64_t prep_cost;
  };
  auto cmp = [](const Cand& a, const Cand& b) {
    if (a.score != b.score) return a.score < b.score;  // max-heap on score
    return a.seq > b.seq;
  };
  std::priority_queue<Cand, std::vector<Cand>, decltype(cmp)> frontier(cmp);
  std::unordered_map<PrepKey, int64_t, PrepKeyHash> best_score;
  best_score.reserve(1024);
  uint64_t seq = 0;
  size_t expansions = 0, evals = 0, cont_total = 0;

  auto push_candidate = [&](const Cand& c) {
    if (cfg.keel_dedup_candidates) {
      PrepKey k = prep_key_of(c.s, r);
      auto it = best_score.find(k);
      if (it != best_score.end() && it->second >= c.score) return;
      best_score[k] = c.score;
    }
    frontier.push(c);
  };

  push_candidate(Cand{0, seq++, s0, {}, 0});
  while (!frontier.empty() && evals < cfg.keel_max_candidates &&
         expansions < cfg.keel_expansion_budget) {
    Cand c = frontier.top();
    frontier.pop();
    ++expansions;

    if (can_commit(g, c.s, r).ok) {
      ++evals;
      State committed = apply_commit(g, c.s, r);
      OracleConfig oc;
      oc.max_expansions = cfg.continuation_cap;
      oc.allow_async = committed.inflight != 0;
      OracleResult cont = solve_continuation(g, committed, r, oc);
      // Audit finding 4: count EVERY continuation search, not just the one
      // that succeeds.
      cont_total += cont.expansions;
      if (cont.status == OracleStatus::Feasible) {
        std::vector<Act> acts = c.acts;
        acts.push_back({ActKind::Commit, -1});
        for (const Act& a : cont.witness.acts) acts.push_back(a);
        PolicyOutcome work;
        work.outer_expansions = expansions;
        work.cont_expansions_total = cont_total;
        work.candidates_checked = evals;
        return finish_accepted(g, s0, r, std::move(acts), work,
                               "keel: searched prep + exact continuation");
      }
      // Infeasible candidate: discarded; keep searching.
    }

    // Generate prep successors (pre-COMMIT only; RELEASE/DROP/RUN).
    const int nts = (int)g.tensors.size();
    for (TensorId t = 0; t < nts; ++t) {
      if (!(c.s.resident & (TensorMask(1) << t))) continue;
      if (can_release(g, c.s, t).ok) {
        State ns = apply_release(g, c.s, t);
        std::vector<Act> na = c.acts;
        na.push_back({ActKind::Release, t});
        push_candidate(Cand{ns.released_bytes - c.prep_cost, seq++, ns,
                            std::move(na), c.prep_cost});
      }
      if (can_drop(g, c.s, t).ok) {
        State ns = apply_drop(g, c.s, t);
        std::vector<Act> na = c.acts;
        na.push_back({ActKind::Drop, t});
        push_candidate(Cand{c.score, seq++, ns, std::move(na), c.prep_cost});
      }
    }
    const int nops = (int)g.ops.size();
    for (OpId v = 0; v < nops; ++v) {
      // Async frontiers: completing in-flight work during preparation can
      // unlock releases before COMMIT (the oracle's plans use this). WAIT is
      // independent of RUN legality — an in-flight op cannot RUN.
      if (c.s.inflight & (OpMask(1) << v)) {
        State nw = apply_wait(g, c.s, v);
        std::vector<Act> wa = c.acts;
        wa.push_back({ActKind::Wait, v});
        frontier.push(Cand{c.score, seq++, nw, std::move(wa), c.prep_cost});
      }
      if (!can_run(g, c.s, r, v).ok) continue;
      State ns = apply_run(g, c.s, r, v);
      std::vector<Act> na = c.acts;
      na.push_back({ActKind::Run, v});
      int64_t pc = c.prep_cost + g.ops[v].cost;
      push_candidate(
          Cand{c.s.released_bytes - pc, seq++, ns, std::move(na), pc});
    }
  }

  o.outer_expansions = expansions;
  o.cont_expansions_total = cont_total;
  o.candidates_checked = evals;
  o.planning_nodes = expansions + cont_total;
  if (evals >= cfg.keel_max_candidates ||
      expansions >= cfg.keel_expansion_budget) {
    o.status = PolicyStatus::Unknown;  // budget exhausted: no feasibility claim
    o.note = "keel: search budget exhausted";
    return o;
  }
  o.status = PolicyStatus::Reject;
  o.note = "keel: no validated candidate found within budget";
  return o;
}

// ---------------------------------------------------------------------------
// Audit finding 5A: preparation-aware greedy.

namespace {

int64_t releasable_bytes(const Graph& g, const State& s) {
  int64_t sum = 0;
  for (int t = 0; t < (int)g.tensors.size(); ++t) {
    TensorMask bit = TensorMask(1) << t;
    if ((s.resident & bit) && !g.tensors[t].required && !g.tensors[t].pinned &&
        can_drop(g, s, t).ok)
      sum += g.tensors[t].size;
  }
  return sum;
}

}  // namespace

PolicyOutcome prep_greedy(const Graph& g, const State& s0, const Request& r,
                          const PolicyConfig& cfg) {
  (void)cfg;  // unlock-runs are bounded by K from the request, not a budget
  PolicyOutcome o;
  LastUse lu = compute_last_use(g);
  const char* names[3] = {"largest-first", "end-of-life-first",
                          "smallest-first"};
  PolicyOutcome work;
  const int nops = (int)g.ops.size();
  for (int order = 0; order < 3; ++order) {
    State s = s0;
    std::vector<Act> prep;
    int64_t prep_used = s0.prep_used;
    // Unlock phase: run ready ops that strictly increase releasable bytes,
    // best delta-per-cost first, bounded by 2*nops runs and the prep budget K.
    for (int runs = 0; runs < 2 * nops; ++runs) {
      int64_t before = releasable_bytes(g, s);
      OpId best = -1;
      int64_t best_ratio_num = 0, best_ratio_den = 1;
      for (OpId v = 0; v < nops; ++v) {
        if (!can_run(g, s, r, v).ok) continue;
        if (!s.committed && prep_used + g.ops[v].cost > r.K) continue;
        State ns = apply_run(g, s, r, v);
        int64_t delta = releasable_bytes(g, ns) - before;
        if (delta <= 0) continue;
        if (best == -1 ||
            delta * best_ratio_den > best_ratio_num * g.ops[v].cost) {
          best = v;
          best_ratio_num = delta;
          best_ratio_den = g.ops[v].cost;
        }
      }
      if (best == -1) break;
      prep.push_back({ActKind::Run, best});
      if (!s.committed) prep_used += g.ops[best].cost;
      s = apply_run(g, s, r, best);
      work.executor_steps += 1;
    }
    Ladder L = build_ladder(g, s, r, order, lu);
    if (!L.commitable) continue;
    State committed = apply_commit(g, L.state, r);
    ExecResult ex = greedy_execute(g, committed, r, lu);
    work.executor_steps += ex.steps + L.acts.size();
    if (!ex.ok) continue;
    std::vector<Act> acts = prep;
    for (const Act& a : L.acts) acts.push_back(a);
    acts.push_back({ActKind::Commit, -1});
    for (const Act& a : ex.acts) acts.push_back(a);
    CheckResult c = validate_counted(g, s0, r, Witness{acts}, work);
    if (c.ok && c.reached_goal) {
      work.candidates_checked = work.checker_invocations;
      return finish_accepted(g, s0, r, std::move(acts), work,
                             std::string("prep_greedy(") + names[order] + ")");
    }
  }
  o.status = PolicyStatus::Reject;
  o.executor_steps = work.executor_steps;
  o.checker_invocations = work.checker_invocations;
  o.checker_us = work.checker_us;
  o.note = "prep_greedy: no unlocked order yields a validated handover";
  return o;
}

// ---------------------------------------------------------------------------
// Audit finding 5B: limited-lookahead planner (keel machinery, tight budget).

PolicyOutcome limited_lookahead(const Graph& g, const State& s0,
                                const Request& r, const PolicyConfig& cfg) {
  PolicyConfig tight = cfg;
  tight.keel_expansion_budget =
      std::min<size_t>(cfg.keel_expansion_budget, 2000);
  tight.keel_max_candidates = std::min<size_t>(cfg.keel_max_candidates, 100);
  PolicyOutcome o = keel_plan(g, s0, r, tight);
  o.note = "limited_lookahead: " + o.note;
  return o;
}

// ---------------------------------------------------------------------------
// Audit finding 5: hybrid — guarded reclamation first, KEEL on failure.

PolicyOutcome hybrid(const Graph& g, const State& s0, const Request& r,
                     const PolicyConfig& cfg) {
  PolicyOutcome fast = guarded_greedy(g, s0, r, cfg);
  if (fast.status == PolicyStatus::Accept) {
    fast.note = "hybrid: guarded fast path (" + fast.note + ")";
    return fast;
  }
  PolicyOutcome full = keel_plan(g, s0, r, cfg);
  PolicyOutcome o = full;
  o.outer_expansions = fast.outer_expansions + full.outer_expansions;
  o.cont_expansions_total =
      fast.cont_expansions_total + full.cont_expansions_total;
  o.candidates_checked = fast.candidates_checked + full.candidates_checked;
  o.checker_invocations =
      fast.checker_invocations + full.checker_invocations;
  o.checker_us = fast.checker_us + full.checker_us;
  o.executor_steps = fast.executor_steps + full.executor_steps;
  o.planning_nodes = full.planning_nodes;  // search work: keel stage only
  o.note = "hybrid: escalated to keel (fast path: " + fast.note + ")";
  return o;
}

}  // namespace keel
