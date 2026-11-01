#!/usr/bin/env python3
"""Phase 1 aggregation (docs/EXPERIMENTS.md frozen metrics).

Reads one or more JSONL files from the runner and prints per-family and
per-(Bn, K) tables plus FULL DISTRIBUTIONS (min/median/p90/max). Never pools
adversarial families into organic headline rates.

Usage: python3 aggregate.py results/raw/dev.jsonl [...]
"""
import collections
import json
import statistics
import sys

ADVERSARIAL = {"trap_recompute", "trap_nonreplayable", "trap_memory",
               "prep_unlock", "trap_concurrent_launch"}
POLICIES = ["conservative", "greedy", "guarded_greedy", "prep_greedy",
            "limited_lookahead", "hybrid", "keel", "oracle_assisted"]


def key(r):
    return (r["family"], r["dist"], r["rep"], r["seed"], r["frontier"],
            r["async"], r["bn_frac"], r["k_frac"])


def dist(vals):
    if not vals:
        return "n/a"
    vs = sorted(vals)
    def q(p):
        i = min(len(vs) - 1, int(p * (len(vs) - 1)))
        return vs[i]
    return (f"min={vs[0]} med={statistics.median(vs)} "
            f"p90={q(0.9)} max={vs[-1]} n={len(vs)}")


def main(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    print(f"rows: {len(rows)}")

    inst = {key(r): r for r in rows if r["policy"] == "oracle_assisted"}
    org = [r for r in inst.values() if r["family"] not in ADVERSARIAL]
    adv = [r for r in inst.values() if r["family"] in ADVERSARIAL]

    for name, group in (("ORGANIC", org), ("ADVERSARIAL (separate)", adv)):
        c = collections.Counter(r["oracle_status"] for r in group)
        print(f"\n== {name} oracle labels: {dict(c)} "
              f"(feasible rate {c['FEASIBLE'] / max(1, len(group)):.1%})")
        byk = collections.defaultdict(collections.Counter)
        for r in group:
            byk[(r["bn_frac"], r["k_frac"])][r["oracle_status"]] += 1
        for k in sorted(byk):
            c2 = byk[k]
            tot = sum(c2.values())
            print(f"   bn={k[0]} K={k[1]:<4}: FEASIBLE {c2['FEASIBLE']:5d}/{tot:5d}"
                  f"  UNKNOWN {c2.get('UNKNOWN', 0)}")

    feas = {k for k, r in inst.items() if r["oracle_status"] == "FEASIBLE"
            and r["family"] not in ADVERSARIAL}
    feas_adv = {k for k, r in inst.items() if r["oracle_status"] == "FEASIBLE"
                and r["family"] in ADVERSARIAL}

    print("\n== Policy behavior on ORGANIC oracle-FEASIBLE instances")
    for p in POLICIES:
        c = collections.Counter()
        for r in rows:
            if r["policy"] == p and key(r) in feas:
                c[r["status"]] += 1
        print(f"   {p:16s} {dict(c)}")

    print("\n== Policy behavior on ORGANIC oracle-INFEASIBLE instances "
          "(ACCEPT here = defect)")
    ins = {k for k, r in inst.items() if r["oracle_status"] == "INFEASIBLE"
           and r["family"] not in ADVERSARIAL}
    for p in POLICIES:
        c = collections.Counter()
        for r in rows:
            if r["policy"] == p and key(r) in ins:
                c[r["status"]] += 1
        print(f"   {p:16s} {dict(c)}")

    print("\n== Defect audit (all organic+adversarial, all statuses)")
    for p in POLICIES:
        bad = [r for r in rows if r["policy"] == p and r["status"] == "ACCEPT"
               and not r["valid"]]
        print(f"   {p:16s} accepted-but-invalid: {len(bad)}")

    print("\n== H1: oracle-feasible instances missed by drop-only policies")
    drop_only = {}
    for p in ("conservative", "guarded_greedy"):
        acc = {key(r) for r in rows if r["policy"] == p
               and r["status"] == "ACCEPT"}
        drop_only[p] = acc
    missed = feas - drop_only["conservative"] - drop_only["guarded_greedy"]
    print(f"   organic feasible: {len(feas)}; missed by BOTH drop-only: "
          f"{len(missed)} ({len(missed) / max(1, len(feas)):.1%})")
    keel_acc = {key(r) for r in rows if r["policy"] == "keel"
                and r["status"] == "ACCEPT"}
    print(f"   of those, keel accepts: {len(missed & keel_acc)} "
          f"({len(missed & keel_acc) / max(1, len(missed)):.1%})")
    # prep-compute evidence: keel rows on missed instances with prep > 0
    preps = [r["prep"] for r in rows if r["policy"] == "keel"
             and key(r) in (missed & keel_acc)]
    print(f"   keel prep-compute on missed: {dist(preps)}")

    print("\n== Distributions on ORGANIC oracle-FEASIBLE (per policy)")
    for p in POLICIES:
        rs = [r for r in rows if r["policy"] == p and key(r) in feas]
        if not rs:
            continue
        print(f"   {p}:")
        for f in ("released", "prep", "recompute", "total", "nodes",
                  "plan_us"):
            print(f"     {f:10s} {dist([r[f] for r in rs])}")

    print("\n== Per-family policy acceptance (organic, oracle-feasible only)")
    fams = sorted({r["family"] for r in org})
    print(f"   {'family':18s}" + "".join(f"{p:>17s}" for p in POLICIES))
    for f in fams:
        line = f"   {f:18s}"
        for p in POLICIES:
            n_feas = len([1 for k in feas if k[0] == f])
            n_acc = len([1 for r in rows if r["policy"] == p
                         and r["status"] == "ACCEPT" and key(r) in feas
                         and key(r)[0] == f])
            line += f"{n_acc}/{n_feas:<15d}"
        print(line)

    print("\n== Async slice (order-abstract in-flight frontiers)")
    fa = {key(r) for r in inst.values() if r["async"]
          and r["oracle_status"] == "FEASIBLE"}
    for p in POLICIES:
        c = collections.Counter()
        for r in rows:
            if r["policy"] == p and key(r) in fa:
                c[r["status"]] += 1
        if c:
            print(f"   {p:16s} {dict(c)}")


if __name__ == "__main__":
    main(sys.argv[1:])
