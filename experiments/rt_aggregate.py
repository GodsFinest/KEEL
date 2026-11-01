#!/usr/bin/env python3
"""Phase 7 real-runtime benchmark aggregation (mandate Part VII).

Reads rt_bench JSONL and prints per-policy accept rates, latency
distributions (min/median/p90/max), correctness/invariant audits, and
per-family breakdowns. Includes bad outcomes; nothing is averaged away.
"""
import collections
import json
import statistics
import sys

POLICIES = ["conservative", "greedy", "guarded_greedy", "prep_greedy",
            "limited_lookahead", "keel", "hybrid", "oracle"]


def dist(vals):
    if not vals:
        return "n/a"
    vs = sorted(vals)

    def q(p):
        return vs[min(len(vs) - 1, int(p * (len(vs) - 1)))]
    return (f"min={vs[0]} med={statistics.median(vs)} "
            f"p90={q(0.9)} max={vs[-1]} n={len(vs)}")


def main(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    baseline = [r for r in rows if r.get("policy") == "none"]
    handover = [r for r in rows if r.get("policy") != "none"]
    print(f"trials: {len(rows)} (baseline={len(baseline)}, handover={len(handover)})")

    print("\n== global audits")
    print("  A reference-correct:", collections.Counter(r.get("a_correct", r.get("correct"))
                                                       for r in handover))
    print("  B reference-correct:", collections.Counter(r.get("b_correct")
                                                       for r in handover))
    print("  invariants hold:  ", collections.Counter(r.get("invariants")
                                                      for r in handover))
    print("  task A completed: ", collections.Counter(r.get("a_ok")
                                                      for r in handover))
    print("  task B completed: ", collections.Counter(r.get("b_ok")
                                                      for r in handover))
    print("  joint success:    ", collections.Counter(r.get("joint_success")
                                                       for r in handover))
    print("  overlap observed: ", collections.Counter(r.get("overlap_observed")
                                                       for r in handover))
    print("  baseline A-only correctness:", collections.Counter(
        r.get("a_correct", r.get("correct")) for r in baseline))
    bad = [r for r in handover if
           (not r.get("joint_success", False) or
            not r.get("invariants", True))]
    print("  failed trials (truthfully reported):")
    for r in bad:
        print(f"    {r['family']} seed={r['seed']} {r['policy']} "
              f"p={r['pressure']} phase={r['phase']} released={r['released']}")

    print("\n== per-policy outcomes")
    pol = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["policy"] == "none":
            continue
        pol[r["policy"]][r["phase"]] += 1
    for p in sorted(pol):
        total = sum(pol[p].values())
        committed = sum(1 for r in rows
                        if r["policy"] == p and r.get("committed", False))
        joint = sum(1 for r in rows
                    if r["policy"] == p and r.get("joint_success", False))
        print(f"  {p:18s} committed {committed:4d}/{total} ({committed/total:.1%})  "
              f"joint {joint:4d}/{total} ({joint/total:.1%})  {dict(pol[p])}")

    print("\n== handover latency (COMMITTED trials, us)")
    for p in sorted(pol):
        vs = [r["handover_us"] for r in rows
              if r["policy"] == p and r.get("committed", False)]
        print(f"  {p:18s} {dist(vs)}")

    print("\n== planning latency (plan_us) and checker (checker_us)")
    for p in sorted(pol):
        vs = [r["plan_us"] for r in rows if r["policy"] == p]
        cs = [r["checker_us"] for r in rows if r["policy"] == p]
        print(f"  {p:18s} plan {dist(vs)}")
        print(f"  {'':18s} check {dist(cs)}")

    print("\n== released bytes (COMMITTED trials)")
    for p in sorted(pol):
        vs = [r["released"] for r in rows
              if r["policy"] == p and r.get("committed", False)]
        print(f"  {p:18s} {dist(vs)}")

    print("\n== Task A completion time (us, per policy incl. failures)")
    for p in sorted(pol):
        vs = [r["a_us"] for r in rows if r["policy"] == p]
        print(f"  {p:18s} {dist(vs)}")

    print("\n== physical overlap and B reuse (committed trials)")
    for p in sorted(pol):
        committed_rows = [r for r in rows
                          if r["policy"] == p and r.get("committed", False)]
        overlap = sum(1 for r in committed_rows
                      if r.get("overlap_observed", False))
        reuse = sum(1 for r in committed_rows
                    if r.get("b_used_transferred", False))
        print(f"  {p:18s} overlap {overlap:4d}/{len(committed_rows):<4d}  "
              f"B-reused {reuse:4d}/{len(committed_rows):<4d}".rstrip())
        print(f"  {'':18s} B first-use {dist([r['b_first_use_us'] for r in committed_rows
                                           if r.get('b_first_use_us', 0) > 0])}")

    print("\n== per-family x policy JOINT-SUCCESS rate")
    fams = sorted({r["family"] for r in rows})
    print("  " + " ".join(f"{p[:12]:>14s}" for p in POLICIES))
    for f in fams:
        line = f"  {f[:14]:16s}"
        for p in POLICIES:
            tr = [r for r in rows if r["family"] == f and r["policy"] == p]
            acc = sum(1 for r in tr if r.get("joint_success", False))
            line += f"{acc:>6d}/{len(tr):<7d}"
        print(line.rstrip())

    print("\n== unconstrained baseline (policy=none)")
    nb = [r for r in rows if r["policy"] == "none"]
    print("  a_us:", dist([r["a_us"] for r in nb]))
    per_f = {r["family"]: r["a_us"] for r in nb}
    for f in sorted(per_f):
        print(f"    {f:18s} {per_f[f]}")


if __name__ == "__main__":
    main(sys.argv[1])
