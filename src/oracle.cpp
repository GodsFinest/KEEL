#include <keel/oracle.hpp>

#include <algorithm>
#include <functional>
#include <queue>
#include <unordered_map>
#include <vector>

namespace keel {

const char* oracle_status_name(OracleStatus s) {
  switch (s) {
    case OracleStatus::Feasible: return "FEASIBLE";
    case OracleStatus::Infeasible: return "INFEASIBLE";
    case OracleStatus::Unknown: return "UNKNOWN";
  }
  return "?";
}

namespace {

// Canonical memo key (SPEC §8): two states with equal keys have identical
// legal futures. prep_used collapses to 0 once committed (it is frozen and no
// rule reads it afterwards) and is capped at K before that.
struct Key {
  uint64_t done = 0;
  uint64_t resident = 0;
  uint64_t inflight = 0;
  int64_t released = 0;
  int64_t prep_key = 0;
  bool committed = false;

  bool operator==(const Key& o) const {
    return done == o.done && resident == o.resident && inflight == o.inflight &&
           released == o.released && prep_key == o.prep_key &&
           committed == o.committed;
  }
};

struct KeyHash {
  size_t operator()(const Key& k) const noexcept {
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

Key key_of(const State& s, const Request& r) {
  Key k;
  k.done = s.done;
  k.resident = s.resident;
  k.inflight = s.inflight;
  k.released = s.released_bytes;
  k.committed = s.committed;
  k.prep_key = s.committed ? 0 : std::min(s.prep_used, r.K);
  return k;
}

// Settled dominance profile (SPEC §9). Cost is implicit: UCS pops in
// nondecreasing cost order, so any settled point has cost <= the popped node's.
struct DomPoint {
  uint64_t done;
  uint64_t resident;
  int64_t released;
  int64_t prep_key;
  bool committed;
};

// A dominates B iff (same phase, both quiescent) done_A ⊇ done_B with only
// replayable extras, resident_A ⊇ resident_B, released_A >= released_B,
// prep_A <= prep_B. Mimic: from A, drop the extra residents (free) to land on
// B's exact memory situation, then replay B's sequence; legality carries over
// because released_A >= released_B keeps the ceiling no smaller relative to
// usage after the drops, and prep_A <= prep_B keeps K satisfied.
bool dominates(const DomPoint& a, const Graph& g, const State& b, int64_t prep_b,
               bool committed_b) {
  if (a.committed != committed_b) return false;
  if (b.inflight != 0) return false;  // dominance requires quiescence
  if ((a.done & b.done) != b.done) return false;
  uint64_t extra_done = a.done & ~b.done;
  while (extra_done) {
    int v = __builtin_ctzll(extra_done);
    if (!g.ops[v].replayable) return false;
    extra_done &= extra_done - 1;
  }
  // Residency must match exactly: with a strict superset, coverage depends on
  // expanding the dominator's drop-closure (itself dominated -> pruned), which
  // loses completeness (docs/DECISIONS.md D11).
  if (a.resident != b.resident) return false;
  // Released bytes are two-dimensional for dominance: more released helps
  // COMMIT's threshold but shrinks the ceiling the mimic replays under. No
  // monotone direction is sound; require equality.
  if (a.released != b.released_bytes) return false;
  // Mimic needs the dominator to have at least as much K headroom left:
  // prep_A <= prep_B (SPEC §9).
  if (a.prep_key > prep_b) return false;
  return true;
}

struct QNode {
  int64_t cost;
  uint64_t seq;  // FIFO tiebreak among equal costs (determinism)
  State s;
};

struct Parent {
  Key prev{};
  bool has_prev = false;
  Act act{ActKind::Run, -1};
};

}  // namespace

OracleResult solve_transition(const Graph& g, const State& s0, const Request& r,
                              const OracleConfig& cfg) {
  OracleResult out;

  auto cmp = [](const QNode& a, const QNode& b) {
    if (a.cost != b.cost) return a.cost > b.cost;
    return a.seq > b.seq;
  };
  std::priority_queue<QNode, std::vector<QNode>, decltype(cmp)> pq(cmp);
  std::unordered_map<Key, int64_t, KeyHash> best;
  std::unordered_map<Key, Parent, KeyHash> parent;
  std::vector<DomPoint> dominators;
  if (cfg.use_dominance) dominators.reserve(1024);
  best.reserve(4096);
  parent.reserve(4096);

  uint64_t seq = 0;

  Key k0 = key_of(s0, r);
  best[k0] = 0;
  parent[k0] = Parent{Key{}, false, Act{ActKind::Run, -1}};
  pq.push(QNode{0, seq++, s0});

  while (!pq.empty()) {
    QNode n = pq.top();
    pq.pop();
    {
      Key k = key_of(n.s, r);
      auto it = best.find(k);
      if (it == best.end() || it->second < n.cost) continue;  // stale entry
    }
    ++out.expansions;
    if (out.expansions > cfg.max_expansions) {
      out.status = OracleStatus::Unknown;
      out.memo_size = best.size();
      return out;  // budget exhausted: NO feasibility claim (SPEC §7)
    }

    if (is_goal(g, n.s)) {
      // Reconstruct the witness by walking parents.
      std::vector<Act> rev;
      Key k = key_of(n.s, r);
      while (true) {
        auto it = parent.find(k);
        if (it == parent.end() || !it->second.has_prev) break;
        rev.push_back(it->second.act);
        k = it->second.prev;
      }
      std::reverse(rev.begin(), rev.end());
      out.status = OracleStatus::Feasible;
      out.witness = Witness{std::move(rev)};
      out.total_compute = n.cost;
      out.memo_size = best.size();
      return out;
    }

    if (cfg.use_dominance && n.s.inflight == 0) {
      bool dominated = false;
      DomPoint matched{};
      for (const DomPoint& d : dominators) {
        if (dominates(d, g, n.s, key_of(n.s, r).prep_key, n.s.committed)) {
          dominated = true;
          matched = d;
          break;
        }
      }
      if (dominated) {
        ++out.dominance_skips;
#ifdef KEEL_DEBUG_DOM
        fprintf(stderr, "DOMSKIP popped done=%lx res=%lx rel=%ld prep=%ld comm=%d | dom done=%lx res=%lx rel=%ld prep=%ld comm=%d\n",
                (unsigned long)n.s.done, (unsigned long)n.s.resident,
                (long)n.s.released_bytes, (long)key_of(n.s,r).prep_key,
                (int)n.s.committed, (unsigned long)matched.done,
                (unsigned long)matched.resident, (long)matched.released,
                (long)matched.prep_key, (int)matched.committed);
#endif
        continue;
      }
      dominators.push_back(DomPoint{n.s.done, n.s.resident, n.s.released_bytes,
                                    key_of(n.s, r).prep_key, n.s.committed});
    }

    // Successor generation (SPEC §5 action set).
    auto consider = [&](const State& ns, int64_t step_cost, Act act) {
      int64_t c;
      if (__builtin_add_overflow(n.cost, step_cost, &c)) return;
      Key k = key_of(ns, r);
      auto it = best.find(k);
      if (it != best.end() && it->second <= c) return;
      best[k] = c;
      parent[k] = Parent{key_of(n.s, r), true, act};
      pq.push(QNode{c, seq++, ns});
    };

    const int nops = (int)g.ops.size();
    for (OpId v = 0; v < nops; ++v) {
      if (cfg.allow_async && can_launch(g, n.s, r, v).ok)
        consider(apply_launch(g, n.s, r, v), g.ops[v].cost, Act{ActKind::Launch, v});
      if (can_run(g, n.s, r, v).ok)
        consider(apply_run(g, n.s, r, v), g.ops[v].cost, Act{ActKind::Run, v});
    }
    if (cfg.allow_async) {
      for (OpId v = 0; v < nops; ++v)
        if (can_wait(g, n.s, v).ok)
          consider(apply_wait(g, n.s, v), 0, Act{ActKind::Wait, v});
    }
    const int nts = (int)g.tensors.size();
    for (TensorId t = 0; t < nts; ++t) {
      if (!(n.s.resident & (TensorMask(1) << t))) continue;
      if (can_drop(g, n.s, t).ok)
        consider(apply_drop(g, n.s, t), 0, Act{ActKind::Drop, t});
      if (can_release(g, n.s, t).ok)
        consider(apply_release(g, n.s, t), 0, Act{ActKind::Release, t});
    }
    if (can_commit(g, n.s, r).ok)
      consider(apply_commit(g, n.s, r), 0, Act{ActKind::Commit, -1});
  }

  out.status = OracleStatus::Infeasible;  // space exhausted: a proof (SPEC §7)
  out.memo_size = best.size();
  return out;
}

}  // namespace keel
