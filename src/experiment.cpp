// KEEL Phase 1 experiment runner (docs/EXPERIMENTS.md frozen grid).
//
// Emits JSON lines: one per (instance × policy) with the oracle label attached.
// Every accepted or failed policy trace is checker-validated here; validation
// failures are recorded as policy defects (they must be zero).

#include <keel/checker.hpp>
#include <keel/gen.hpp>
#include <keel/oracle.hpp>
#include <keel/policy.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace keel;

namespace {

PolicyOutcome greedy0(const Graph& g, const State& s, const Request& r,
                      const PolicyConfig& c) {
  return greedy(g, s, r, c, 0);
}

int64_t now_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::string jstr(const std::string& s) { return "\"" + s + "\""; }

struct Row {
  std::string family, dist;
  int seed = 0, frontier_id = 0, rep = 0;
  bool async = false;
  double bn_frac = 0, k_frac = 0;
  int64_t bo = 0, bn = 0, K = 0;
  std::string policy;
  std::string status;
  bool valid = false;
  int64_t released = 0, prep = 0, cont = 0, recompute = 0, total = 0;
  size_t nodes = 0;
  size_t outer_nodes = 0, cont_nodes = 0, candidates = 0, checker_calls = 0,
         executor_steps = 0;
  int64_t checker_us = 0;
  int64_t plan_us = 0;
  std::string oracle_status;
  int64_t oracle_total = 0;
  bool oracle_label_ok = false;  // witness validated (for oracle rows)
};

void emit(const Row& r) {
  printf("{\"family\":%s,\"dist\":%s,\"seed\":%d,\"rep\":%d,\"frontier\":%d,\"async\":%s,"
         "\"bn_frac\":%.2f,\"k_frac\":%.2f,\"bo\":%ld,\"bn\":%ld,\"K\":%ld,"
         "\"policy\":%s,\"status\":%s,\"valid\":%s,\"released\":%ld,"
         "\"prep\":%ld,\"cont\":%ld,\"recompute\":%ld,\"total\":%ld,"
         "\"nodes\":%zu,\"outer_nodes\":%zu,\"cont_nodes\":%zu,"
         "\"candidates\":%zu,\"checker_calls\":%zu,\"checker_us\":%ld,"
         "\"executor_steps\":%zu,\"plan_us\":%ld,\"oracle_status\":%s,"
         "\"oracle_total\":%ld,\"oracle_label_ok\":%s}\n",
         jstr(r.family).c_str(), jstr(r.dist).c_str(), r.seed, r.rep,
         r.frontier_id,
         r.async ? "true" : "false", r.bn_frac, r.k_frac, (long)r.bo,
         (long)r.bn, (long)r.K, jstr(r.policy).c_str(), jstr(r.status).c_str(),
         r.valid ? "true" : "false", (long)r.released, (long)r.prep,
         (long)r.cont, (long)r.recompute, (long)r.total, r.nodes,
         r.outer_nodes, r.cont_nodes, r.candidates, r.checker_calls,
         (long)r.checker_us, r.executor_steps,
         (long)r.plan_us, jstr(r.oracle_status).c_str(),
         (long)r.oracle_total, r.oracle_label_ok ? "true" : "false");
}

void record(Row base, const char* policy, const PolicyOutcome& o,
            const Graph& g, const State& s0, const Request& req,
            const std::string& oracle_status, int64_t oracle_total) {
  base.policy = policy;
  base.status = policy_status_name(o.status);
  base.nodes = o.planning_nodes;
  base.outer_nodes = o.outer_expansions;
  base.cont_nodes = o.cont_expansions_total;
  base.candidates = o.candidates_checked;
  base.checker_calls = o.checker_invocations;
  base.checker_us = o.checker_us;
  base.executor_steps = o.executor_steps;
  base.released = o.released_bytes;
  base.prep = o.prep_compute;
  base.cont = o.cont_compute;
  base.recompute = o.recompute_compute;
  base.total = o.prep_compute + o.cont_compute;
  // Independent re-validation for Accept/Failed rows (defense in depth; the
  // policies already self-validate — a mismatch here is a defect).
  if (o.status == PolicyStatus::Accept || o.status == PolicyStatus::Failed) {
    CheckResult c = check_witness(g, s0, req, o.witness);
    base.valid = c.ok && (o.status == PolicyStatus::Failed ? true : c.reached_goal);
  }
  base.oracle_status = oracle_status;
  base.oracle_total = oracle_total;
  emit(base);
}

}  // namespace

int main(int argc, char** argv) {
  int seed_lo = 1, seed_hi = 20;
  bool held_out = false;
  const char* out_path = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--seed-lo") && i + 1 < argc) seed_lo = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seed-hi") && i + 1 < argc) seed_hi = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--held-out")) held_out = true;
    else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
  }
  if (out_path) freopen(out_path, "w", stdout);
  fprintf(stderr, "held_out=%d seeds=[%d,%d]\n", held_out, seed_lo, seed_hi);

  const Family families[] = {
      Family::Chain,        Family::Diamond,         Family::SharedAncestor,
      Family::ForkJoin,     Family::Residual,        Family::BigWorkspace,
      Family::Heterogeneous};
  const Family traps[] = {Family::TrapRecompute, Family::TrapNonreplayable,
                          Family::TrapMemory, Family::PrepUnlock,
                          Family::TrapConcurrentLaunch};
  const double bn_fracs[] = {0.50, 0.65, 0.80};
  const double k_fracs[] = {0.0, 0.10, 0.25};
  PolicyConfig cfg;

  auto run_instance = [&](const Graph& g, const State& s0, const Row& meta) {
    for (double bf : bn_fracs) {
      int64_t Bn = (int64_t)(meta.bo * bf);
      if (Bn <= 0) continue;
      for (double kf : k_fracs) {
        int64_t K = (int64_t)(remaining_work(g, s0) * kf);
        Request r{meta.bo, Bn, K};
        Row row = meta;
        row.bn_frac = bf;
        row.k_frac = kf;
        row.bn = Bn;
        row.K = K;

        // Ground truth first.
        int64_t t0 = now_us();
        OracleConfig ocfg{};
        ocfg.allow_async = s0.inflight != 0;  // async slice needs WAIT actions
        OracleResult oracle = solve_transition(g, s0, r, ocfg);
        int64_t oracle_us = now_us() - t0;
        std::string ostr = oracle_status_name(oracle.status);
        bool oracle_valid = false;
        if (oracle.status == OracleStatus::Feasible) {
          CheckResult c = check_witness(g, s0, r, oracle.witness);
          oracle_valid = c.ok && c.reached_goal;
        }
        Row orow = row;
        orow.policy = "oracle_assisted";
        orow.status = ostr == "FEASIBLE" ? "ACCEPT" : "REJECT";
        orow.valid = oracle_valid;
        orow.nodes = oracle.expansions;
        orow.plan_us = oracle_us;
        orow.oracle_status = ostr;
        orow.oracle_total = oracle.total_compute;
        orow.oracle_label_ok = oracle_valid;
        if (oracle.status == OracleStatus::Feasible) {
          CheckResult c = check_witness(g, s0, r, oracle.witness);
          orow.released = c.released_bytes;
          orow.prep = c.prep_compute;
          orow.cont = c.cont_compute;
          orow.recompute = c.recompute_compute;
          orow.total = c.prep_compute + c.cont_compute;
        }
        emit(orow);

        auto run_policy = [&](const char* name,
                              PolicyOutcome (*fn)(const Graph&, const State&,
                                                  const Request&,
                                                  const PolicyConfig&)) {
          int64_t t1 = now_us();
          PolicyOutcome o = fn(g, s0, r, cfg);
          int64_t us = now_us() - t1;
          row.plan_us = us;
          record(row, name, o, g, s0, r, ostr, oracle.total_compute);
        };
        run_policy("conservative", conservative);
        run_policy("greedy", greedy0);
        run_policy("guarded_greedy", guarded_greedy);
        run_policy("prep_greedy", prep_greedy);
        run_policy("limited_lookahead", limited_lookahead);
        run_policy("hybrid", hybrid);
        run_policy("keel", keel_plan);
      }
    }
  };

  auto run_family = [&](Family f) {
    // Traps have fixed structure: one replica, one distribution label.
    std::vector<CostDist> ds;
    if (family_is_adversarial(f)) ds = {CostDist::Uniform};
    else ds = {CostDist::Uniform, CostDist::HeavyTail, CostDist::Decorrelated};
    for (CostDist d : ds) {
      int n_rep = family_is_adversarial(f) ? 1 : 3;
      for (int rep = 0; rep < n_rep; ++rep) {
        for (int seed = seed_lo; seed <= seed_hi; ++seed) {
          Generated gen = generate(f, d, (uint64_t)seed * 131u + rep);
          const Graph& g = gen.graph;
          for (int fseed = 0; fseed < 2; ++fseed) {
            Rng rng((uint64_t)seed * 977u + fseed * 31u + rep * 7u);
            State s0 =
                sample_frontier(g, gen.bo, rng, 1,
                                (int)g.ops.size() - 1, false, 0);
            if (s0.resident == 0) continue;  // empty frontier: nothing to hand over
            Row meta;
            meta.family = family_name(f);
            meta.dist = dist_name(d);
            meta.seed = seed;
            meta.frontier_id = fseed;
            meta.rep = rep;
            meta.async = false;
            meta.bo = gen.bo;
            run_instance(g, s0, meta);
          }
          // Async slice: chain and fork_join only (frozen grid).
          if (f == Family::Chain || f == Family::ForkJoin) {
            Rng rng((uint64_t)seed * 31u + 999u);
            State s0 = sample_frontier(g, gen.bo, rng, 1,
                                       (int)g.ops.size() - 1, true, 2);
            Row meta;
            meta.family = family_name(f);
            meta.dist = dist_name(d);
            meta.seed = seed;
            meta.frontier_id = 2;
            meta.rep = rep;
            meta.async = true;
            meta.bo = gen.bo;
            run_instance(g, s0, meta);
          }
        }
      }
    }
  };

  for (Family f : families) run_family(f);
  for (Family f : traps) run_family(f);  // 1 rep, fixed structure
  fprintf(stderr, "done\n");
  return 0;
}
