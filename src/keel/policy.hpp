// KEEL Phase 1 — Baseline policies and the bounded planner (SPEC §10).
//
// Every policy either REJECTs (the task continues legally under Bo — each prep
// action is individually legal, so stopping anywhere is safe) or emits a full
// witness. The HARNESS checker-validates every witness; `greedy` may emit a
// trace that commits and then fails to reach the goal (status Failed) — that
// is the unguarded-reclamation defect the mandate says must never be hidden.

#pragma once

#include <keel/model.hpp>
#include <keel/oracle.hpp>

#include <string>

namespace keel {

enum class PolicyStatus { Accept, Reject, Failed, Unknown };
const char* policy_status_name(PolicyStatus s);
PolicyStatus policy_status_from_oracle(OracleStatus s);

// Work accounting (audit finding 4). Normalized definitions:
//  - outer_expansions: node pops of the preparation search (search policies
//    only; constructive policies have none).
//  - cont_expansions_total: expansions summed over ALL continuation-search
//    invocations, including unsuccessful candidates (v1.0 counted only the
//    successful one).
//  - candidates_checked: continuation searches invoked.
//  - checker_invocations / checker_us: independent-validation workload.
//  - executor_steps: constructive executor steps (greedy family). NOT
//    comparable to UCS expansions and never summed with them; comparisons
//    across policy families use wall-clock plus these labeled fields.
//  - planning_nodes: total search work = outer + cont_expansions_total
//    (0 for purely constructive policies).
struct PolicyOutcome {
  PolicyStatus status = PolicyStatus::Reject;
  Witness witness;              // Accept: full plan. Failed: committed trace.
  int64_t released_bytes = 0;   // from checker validation (0 if rejected)
  int64_t prep_compute = 0;
  int64_t cont_compute = 0;
  int64_t recompute_compute = 0;
  size_t outer_expansions = 0;
  size_t cont_expansions_total = 0;
  size_t candidates_checked = 0;
  size_t checker_invocations = 0;
  int64_t checker_us = 0;
  size_t executor_steps = 0;
  size_t planning_nodes = 0;    // = outer_expansions + cont_expansions_total
  std::string note;             // human-readable reason / which order won
};

struct PolicyConfig {
  size_t continuation_cap = 20000;   // UCS node cap for continuation checks
  size_t keel_max_candidates = 2000; // candidate continuation checks
  size_t keel_expansion_budget = 20000;
  // Phase 8 ablation flag: canonical-key deduplication of prep candidates.
  // With it off, duplicate states reached via different action orders flood
  // the expansion budget (measured: 20,000 expansions over 729 distinct
  // states on a 6-tensor instance).
  bool keel_dedup_candidates = true;
};

using LastUse = std::vector<int>;  // per tensor: topo index of last consumer

LastUse compute_last_use(const Graph& g);

// -- Baseline 1: conservative (drop-only prep, then prove the continuation) --
PolicyOutcome conservative(const Graph& g, const State& s0, const Request& r,
                           const PolicyConfig& cfg);

// -- Baseline 2: greedy (unguarded reclamation + greedy continuation) --------
PolicyOutcome greedy(const Graph& g, const State& s0, const Request& r,
                     const PolicyConfig& cfg, int order = 0);

// -- Baseline 3: guarded greedy (3 deterministic orders, checker-validated) --
PolicyOutcome guarded_greedy(const Graph& g, const State& s0, const Request& r,
                             const PolicyConfig& cfg);

// -- KEEL: bounded planner (greedy seed + best-first prep search with
//    checked continuations; may spend prep compute to unlock releases) -------
PolicyOutcome keel_plan(const Graph& g, const State& s0, const Request& r,
                        const PolicyConfig& cfg);

// -- Audit finding 5A: preparation-aware greedy. Before each release ladder it
//    runs "unlock" ops — ready operations whose execution strictly increases
//    the releasable byte set — then performs the same guarded continuation
//    checking as guarded_greedy.
PolicyOutcome prep_greedy(const Graph& g, const State& s0, const Request& r,
                          const PolicyConfig& cfg);

// -- Audit finding 5B: limited-lookahead planner. Identical legality and
//    continuation validation as keel_plan, with a much tighter search budget
//    (prep expansions <= 2000, candidate checks <= 100).
PolicyOutcome limited_lookahead(const Graph& g, const State& s0,
                                const Request& r, const PolicyConfig& cfg);

// -- Audit finding 5: hybrid. Cheap guarded reclamation first; full KEEL
//    planner only if that fails. Work accounting is the sum of both stages.
PolicyOutcome hybrid(const Graph& g, const State& s0, const Request& r,
                     const PolicyConfig& cfg);

}  // namespace keel
