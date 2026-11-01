#!/usr/bin/env python3
"""Clustered, paired analysis for the KEEL CPU-runtime evaluation.

The independent sampling unit is workload_instance_id. Rows within one graph
instance remain together during bootstrap resampling, while policy/factor
matching is preserved inside each cluster. Repeated timing rows are therefore
not treated as independent graph observations.
"""

import argparse
import collections
import gzip
import json
import math
import random
import statistics
from pathlib import Path


BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 20261101


def read_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def cluster_id(row):
    return row.get("workload_instance_id") or (
        f"{row.get('family')}:{row.get('seed')}:{row.get('graph_fingerprint', 'legacy')}"
    )


def quantiles(values):
    values = sorted(values)
    if not values:
        return {"n": 0}

    def q(p):
        if len(values) < 20 and p >= 0.99:
            return None
        if len(values) == 1:
            return values[0]
        pos = p * (len(values) - 1)
        lo, hi = math.floor(pos), math.ceil(pos)
        if lo == hi:
            return values[lo]
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)

    out = {"n": len(values), "min": values[0], "median": statistics.median(values),
           "p90": q(0.90), "p95": q(0.95), "p99": q(0.99), "max": values[-1]}
    if len(values) < 20:
        out["p99_note"] = "suppressed for n<20"
    return out


def mean_ci(cluster_values, seed=BOOTSTRAP_SEED, iterations=BOOTSTRAP_ITERATIONS):
    values = [float(value) for value in cluster_values]
    if not values:
        return {"clusters": 0, "mean": None, "ci95": None}
    mean = statistics.fmean(values)
    if len(values) == 1:
        return {"clusters": 1, "mean": mean, "ci95": [mean, mean]}
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        samples.append(statistics.fmean(values[rng.randrange(len(values))]
                                        for _ in values))
    samples.sort()
    return {"clusters": len(values), "mean": mean,
            "ci95": [samples[int(0.025 * (len(samples) - 1))],
                     samples[int(0.975 * (len(samples) - 1))]]}


def rate(rows, field):
    values = [float(bool(row.get(field))) for row in rows]
    return {"rows": len(values), "count": int(sum(values)),
            "rate": statistics.fmean(values) if values else None}


def clustered_rate(rows, field):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[cluster_id(row)].append(float(bool(row.get(field))))
    return {**rate(rows, field),
            "cluster_estimate": mean_ci([statistics.fmean(v) for v in grouped.values()])}


def counterfactual_completion(row, baseline):
    return float(_completion(row, baseline))


def clustered_counterfactual_rate(rows, baseline):
    values = [counterfactual_completion(row, baseline) for row in rows]
    grouped = collections.defaultdict(list)
    for row, value in zip(rows, values):
        grouped[cluster_id(row)].append(value)
    return {
        "rows": len(values),
        "count": int(sum(values)),
        "rate": statistics.fmean(values) if values else None,
        "cluster_estimate": mean_ci([statistics.fmean(v) for v in grouped.values()]),
    }


def _completion(row, baseline):
    if baseline == "handover":
        return bool(row.get("committed") and row.get("b_ok") and
                    row.get("b_correct") and row.get("invariants"))
    return bool(row.get(f"{baseline}_b_ok") and row.get(f"{baseline}_b_correct") and
                row.get(f"{baseline}_joint_completion"))


def metric_value(row, field):
    """Return None when a timing/value is only meaningful after COMMIT."""
    if field == "a_slowdown_ratio" and not (row.get("committed") and row.get("a_ok")):
        return None
    if field == "reused_bytes" and not row.get("committed"):
        return None
    return row.get(field)


def paired_cluster_delta(rows, policy, reference, field):
    """Policy-reference effect, first averaged within each graph cluster."""
    match_fields = ("grant_profile", "pressure", "budget_fraction",
                    "frontier_fraction", "repeat")
    by_cluster = collections.defaultdict(lambda: collections.defaultdict(dict))
    for row in rows:
        if row.get("policy") not in (policy, reference):
            continue
        key = tuple(row.get(field_name) for field_name in match_fields)
        by_cluster[cluster_id(row)][key][row["policy"]] = row
    cluster_effects = []
    for matches in by_cluster.values():
        effects = []
        for pair in matches.values():
            if policy not in pair or reference not in pair:
                continue
            left, right = pair[policy], pair[reference]
            if field == "joint_success":
                effects.append(float(bool(left.get(field))) -
                               float(bool(right.get(field))))
            elif field == "fair_wait_completion_delta":
                effects.append(_completion(left, "wait") -
                               _completion(right, "wait"))
            elif field == "fair_drain_completion_delta":
                effects.append(_completion(left, "drain") -
                               _completion(right, "drain"))
            else:
                lv, rv = metric_value(left, field), metric_value(right, field)
                if lv is not None and rv is not None:
                    effects.append(float(lv) - float(rv))
        if effects:
            cluster_effects.append(statistics.fmean(effects))
    return mean_ci(cluster_effects)


def svg_bar(path, title, labels, values, y_label, upper=1.0):
    width, height = 1000, 560
    left, top, chart_w, chart_h = 100, 90, 820, 360
    bottom = top + chart_h
    upper = max(float(upper), 1.0e-12)
    scale = chart_h / upper
    bar_w = chart_w / max(1, len(labels)) * 0.65
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '<style>text{font-family:monospace;font-size:13px}.title{font-size:18px;font-weight:bold}</style>',
        f'<text x="{width/2}" y="30" text-anchor="middle" class="title">{title}</text>',
        f'<text x="20" y="{top + chart_h/2}" transform="rotate(-90 20 {top + chart_h/2})" text-anchor="middle">{y_label}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{left+chart_w}" y2="{bottom}" stroke="black"/>',
    ]
    for tick in range(0, 6):
        value = upper * tick / 5
        y = bottom - value * scale
        parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left+chart_w}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end">{value:.3g}</text>')
    for i, (label, value) in enumerate(zip(labels, values)):
        value = float(value or 0.0)
        x = left + chart_w * (i + 0.5) / len(labels)
        h = max(0.0, min(upper, value)) * scale
        parts.append(f'<rect x="{x-bar_w/2:.1f}" y="{bottom-h:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#4472c4"/>')
        parts.append(f'<text x="{x:.1f}" y="{bottom+22}" text-anchor="middle" transform="rotate(25 {x:.1f} {bottom+22})">{label}</text>')
        parts.append(f'<text x="{x:.1f}" y="{max(top+12, bottom-h-6):.1f}" text-anchor="middle">{value:.3g}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def svg_lines(path, title, x_labels, series, y_label, upper=1.0):
    width, height = 1000, 600
    left, top, chart_w, chart_h = 100, 90, 760, 380
    bottom = top + chart_h
    upper = max(float(upper), 1.0e-12)
    colors = ["#4472c4", "#ed7d31", "#70ad47", "#a64d79", "#8064a2", "#00a6a6", "#c00000"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '<style>text{font-family:monospace;font-size:12px}.title{font-size:18px;font-weight:bold}</style>',
        f'<text x="{width/2}" y="30" text-anchor="middle" class="title">{title}</text>',
        f'<text x="20" y="{top + chart_h/2}" transform="rotate(-90 20 {top + chart_h/2})" text-anchor="middle">{y_label}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{left+chart_w}" y2="{bottom}" stroke="black"/>',
    ]
    for tick in range(0, 6):
        value = upper * tick / 5
        y = bottom - value * chart_h / upper
        parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left+chart_w}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end">{value:.3g}</text>')
    for i, label in enumerate(x_labels):
        x = left + chart_w * (i / max(1, len(x_labels) - 1))
        parts.append(f'<text x="{x:.1f}" y="{bottom+22}" text-anchor="middle">{label}</text>')
    for series_index, (label, values) in enumerate(series):
        points = []
        for i, value in enumerate(values):
            x = left + chart_w * (i / max(1, len(x_labels) - 1))
            y = bottom - float(value or 0.0) * chart_h / upper
            points.append(f"{x:.1f},{y:.1f}")
        color = colors[series_index % len(colors)]
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>')
        lx, ly = left + chart_w + 15, top + 20 + series_index * 20
        parts.append(f'<line x1="{lx}" y1="{ly-4}" x2="{lx+20}" y2="{ly-4}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx+26}" y="{ly}">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def svg_scatter(path, title, points, x_label, y_label):
    width, height = 1000, 600
    left, top, chart_w, chart_h = 110, 90, 700, 380
    bottom = top + chart_h
    if points:
        xs = [float(point[1]) for point in points]
        ys = [float(point[2]) for point in points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
    else:
        xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    xpad = max((xmax - xmin) * 0.10, 1.0)
    ypad = max((ymax - ymin) * 0.10, 1.0)
    xmin, xmax = xmin - xpad, xmax + xpad
    ymin, ymax = ymin - ypad, ymax + ypad
    def sx(value):
        return left + (float(value) - xmin) * chart_w / (xmax - xmin)
    def sy(value):
        return bottom - (float(value) - ymin) * chart_h / (ymax - ymin)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '<style>text{font-family:monospace;font-size:12px}.title{font-size:18px;font-weight:bold}</style>',
        f'<text x="{width/2}" y="30" text-anchor="middle" class="title">{title}</text>',
        f'<text x="{left + chart_w/2}" y="{bottom+48}" text-anchor="middle">{x_label}</text>',
        f'<text x="20" y="{top + chart_h/2}" transform="rotate(-90 20 {top + chart_h/2})" text-anchor="middle">{y_label}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{left+chart_w}" y2="{bottom}" stroke="black"/>',
    ]
    for point_index, (label, x_value, y_value) in enumerate(points):
        cx, cy = sx(x_value), sy(y_value)
        color = ["#4472c4", "#ed7d31", "#70ad47", "#a64d79"][point_index % 4]
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{color}"/>')
        parts.append(f'<text x="{cx+8:.1f}" y="{max(top+12, cy-8):.1f}">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference-policy", default="hybrid")
    args = parser.parse_args()
    rows = read_rows(args.input)
    if not rows:
        raise SystemExit("input contains no rows")
    out = args.out_dir
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    policies = sorted({row["policy"] for row in rows})
    clusters = sorted({cluster_id(row) for row in rows})

    by_policy = {}
    for policy in policies:
        subset = [row for row in rows if row["policy"] == policy]
        committed = [row for row in subset if row.get("committed")]
        valid_wait_benefit = [row["wait_b_benefit_us"] for row in subset
                              if row.get("wait_b_benefit_us") is not None]
        valid_drain_benefit = [row["drain_b_benefit_us"] for row in subset
                               if row.get("drain_b_benefit_us") is not None]
        by_policy[policy] = {
            "attempted": len(subset),
            "clusters": len({cluster_id(row) for row in subset}),
            "planning_status": dict(collections.Counter(row.get("planning_status")
                                                         for row in subset)),
            "committed": clustered_rate(subset, "committed"),
            "joint_success": clustered_rate(subset, "joint_success"),
            "a_correct": clustered_rate([r for r in subset if r.get("a_ok")], "a_correct"),
            "b_correct": clustered_rate([r for r in subset if r.get("b_ok")], "b_correct"),
            "invariants": clustered_rate(subset, "invariants"),
            "overlap": clustered_rate(committed, "overlap_observed"),
            "physical_reuse": clustered_rate(committed, "reused_bytes"),
            "fair_wait_completion": clustered_counterfactual_rate(subset, "wait"),
            "fair_drain_completion": clustered_counterfactual_rate(subset, "drain"),
            "fair_wait_b_benefit_us": quantiles(valid_wait_benefit),
            "fair_drain_b_benefit_us": quantiles(valid_drain_benefit),
            "failure_stages": dict(collections.Counter(
                row.get("failure_stage", "unknown") for row in subset
                if not row.get("joint_success"))),
            "plan_us": quantiles([row["plan_us"] for row in subset]),
            "checker_us": quantiles([row["checker_us"] for row in subset]),
            "end_to_end_us": quantiles([row["end_to_end_us"] for row in subset]),
            "a_slowdown_ratio": quantiles([row["a_slowdown_ratio"] for row in committed
                                            if row.get("a_ok")]),
            "reused_bytes": quantiles([row["reused_bytes"] for row in committed]),
            "fragmentation_before": quantiles([row["fragmentation_before"] for row in subset]),
        }

    paired = {}
    if args.reference_policy in policies:
        fields = ("joint_success", "fair_wait_completion_delta",
                  "fair_drain_completion_delta", "a_slowdown_ratio",
                  "fair_wait_b_benefit_us", "fair_drain_b_benefit_us",
                  "end_to_end_us", "plan_us", "reused_bytes")
        for policy in policies:
            if policy == args.reference_policy:
                continue
            paired[policy] = {field: paired_cluster_delta(
                rows, policy, args.reference_policy, field) for field in fields}

    resource_fields = (
        "graph_scratch_bytes", "graph_peak_scratch_bytes",
        "required_output_bytes", "a_live_bytes_before", "b_live_bytes_before",
        "a_peak_live_bytes", "b_peak_live_bytes", "b_free_blocks_before",
        "b_largest_free_run_before", "remaining_work_request",
    )
    resource_distributions = {}
    for field in resource_fields:
        values = [row[field] for row in rows
                  if isinstance(row.get(field), (int, float))]
        resource_distributions[field] = quantiles(values)
    resource_by_profile = {}
    for profile in sorted({row.get("grant_profile") for row in rows}):
        resource_by_profile[profile] = {}
        profile_rows = [row for row in rows if row.get("grant_profile") == profile]
        for field in resource_fields:
            resource_by_profile[profile][field] = quantiles(
                [row[field] for row in profile_rows
                 if isinstance(row.get(field), (int, float))])

    summary = {
        "input": str(args.input), "rows": len(rows),
        "independent_workload_instances": len(clusters),
        "unique_graph_fingerprints": len({row.get("graph_fingerprint") for row in rows}),
        "unique_resource_configurations": len({(
            row.get("grant_profile"), row.get("old_grant_bytes"),
            row.get("b_grant_bytes"), row.get("pressure"),
            row.get("budget_fraction"), row.get("frontier_fraction")) for row in rows}),
        "timing_repetitions": len({row.get("repeat") for row in rows}),
        "independent_seeds": len({(row.get("family"), row.get("seed")) for row in rows}),
        "bootstrap": {"unit": "workload_instance_id", "seed": BOOTSTRAP_SEED,
                       "iterations": BOOTSTRAP_ITERATIONS},
        "policies": policies, "reference_policy": args.reference_policy,
        "by_policy": by_policy, "paired_cluster_delta_vs_reference": paired,
        "resource_distributions": resource_distributions,
        "resource_by_profile": resource_by_profile,
        "failure_stage": dict(collections.Counter(row.get("failure_stage", "unknown")
                                                   for row in rows
                                                   if not row.get("joint_success"))),
        "family_joint_success": {},
    }
    for family in sorted({row["family"] for row in rows}):
        summary["family_joint_success"][family] = {}
        for policy in policies:
            subset = [row for row in rows if row["family"] == family and
                      row["policy"] == policy]
            summary["family_joint_success"][family][policy] = clustered_rate(
                subset, "joint_success")

    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out / "failure_cases.jsonl").open("w") as stream:
        fields = ("family", "workload_instance_id", "graph_fingerprint", "seed",
                  "repeat", "grant_profile", "policy", "pressure",
                  "budget_fraction", "frontier_fraction", "planning_status",
                  "phase", "failure_stage", "committed", "joint_success",
                  "b_ok", "b_correct", "wait_b_ok", "drain_b_ok",
                  "wait_b_benefit_us", "drain_b_benefit_us", "plan_us",
                  "end_to_end_us")
        for row in rows:
            if not row.get("joint_success"):
                stream.write(json.dumps({key: row.get(key) for key in fields},
                                        sort_keys=True) + "\n")

    md = ["# KEEL Clustered Evaluation Analysis", "",
          f"Input: `{args.input}`", f"Rows: **{len(rows)}**.",
          f"Independent workload instances: **{len(clusters)}**; unique graph fingerprints: **{summary['unique_graph_fingerprints']}**.",
          "Confidence intervals resample workload instances, not trial rows.", "",
          "## Policy outcomes", "",
          "| policy | rows | instances | committed | joint success | B completion (wait) | B completion (drain) | A correct | B correct |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for policy in policies:
        info = by_policy[policy]
        md.append("| {} | {} | {} | {}/{} ({}) | {}/{} ({}) | {}/{} ({}) | {}/{} ({}) | {}/{} ({}) | {}/{} ({}) |".format(
            policy, info["attempted"], info["clusters"],
            info["committed"]["count"], info["committed"]["rows"], fmt(info["committed"]["rate"]),
            info["joint_success"]["count"], info["joint_success"]["rows"], fmt(info["joint_success"]["rate"]),
            info["fair_wait_completion"]["count"], info["fair_wait_completion"]["rows"], fmt(info["fair_wait_completion"]["rate"]),
            info["fair_drain_completion"]["count"], info["fair_drain_completion"]["rows"], fmt(info["fair_drain_completion"]["rate"]),
            info["a_correct"]["count"], info["a_correct"]["rows"], fmt(info["a_correct"]["rate"]),
            info["b_correct"]["count"], info["b_correct"]["rows"], fmt(info["b_correct"]["rate"])))
    md += ["", "## Distributions", "", "| policy | plan us | checker us | end-to-end us | A slowdown | fair wait B benefit us | fair drain B benefit us |", "|---|---|---|---|---|---|---|"]
    for policy in policies:
        info = by_policy[policy]
        md.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            policy, fmt(info["plan_us"]), fmt(info["checker_us"]),
            fmt(info["end_to_end_us"]), fmt(info["a_slowdown_ratio"]),
            fmt(info["fair_wait_b_benefit_us"]), fmt(info["fair_drain_b_benefit_us"])))
    md += ["", "## Failure stages", "", "| stage | count |", "|---|---:|"]
    for stage, count in sorted(summary["failure_stage"].items()):
        md.append(f"| {stage} | {count} |")
    md += ["", "## Clustered paired deltas vs " + args.reference_policy, "",
           "Each graph instance contributes one within-instance mean after matching policy/factor/repeat keys. Bootstrap resamples those instance means.", "",
           "| policy | joint success | wait completion | drain completion | A slowdown | wait B benefit us | drain B benefit us | plan us |", "|---|---|---|---|---|---|---|---|"]
    for policy, fields in paired.items():
        md.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            policy, *(fmt(fields[field]) for field in ("joint_success", "fair_wait_completion_delta", "fair_drain_completion_delta", "a_slowdown_ratio", "fair_wait_b_benefit_us", "fair_drain_b_benefit_us", "plan_us"))))
    md += ["", "## Per-family joint success", "", "| family | " + " | ".join(policies) + " |", "|---|" + "---:|" * len(policies)]
    for family, values in summary["family_joint_success"].items():
        md.append("| {} | {} |".format(family, " | ".join(
            f"{v['count']}/{v['rows']} ({fmt(v['rate'])})" for v in (values[p] for p in policies))))
    md += ["", "## Residency and resource measurements", "",
           "These distributions are measured at the request frontier or during the completed trial. The calibration tier uses the same fields to justify grant profiles; held-out values remain descriptive diagnostics.",
           "", "| metric | n | min | median | p90 | max |", "|---|---:|---:|---:|---:|---:|"]
    for field, values in summary["resource_distributions"].items():
        md.append("| {} | {} | {} | {} | {} | {} |".format(
            field, values.get("n", 0), fmt(values.get("min")),
            fmt(values.get("median")), fmt(values.get("p90")),
            fmt(values.get("max"))))
    (out / "tables.md").write_text("\n".join(md) + "\n")

    labels = policies
    joint_values = [by_policy[p]["joint_success"]["rate"] or 0.0 for p in labels]
    svg_bar(plots / "joint_success_by_policy.svg", "Joint success by policy", labels,
            joint_values, "rate", 1.0)
    wait_values = [by_policy[p]["fair_wait_completion"]["rate"] or 0.0
                   for p in labels]
    svg_bar(plots / "task_b_completion_vs_wait.svg",
            "Task B completion in handover rows vs wait baseline", labels,
            wait_values, "completion rate", 1.0)
    overlap_values = [by_policy[p]["overlap"]["cluster_estimate"]["mean"] or 0.0
                      for p in labels]
    svg_bar(plots / "overlap_rate_by_policy.svg",
            "Observed post-COMMIT overlap among committed rows", labels,
            overlap_values, "overlap rate", 1.0)
    latencies = [by_policy[p]["end_to_end_us"].get("median", 0) or 0 for p in labels]
    svg_bar(plots / "median_end_to_end_us.svg", "Median request-to-completion", labels,
            latencies, "microseconds", max(latencies + [1]))
    plan_p90 = [by_policy[p]["plan_us"].get("p90", 0) or 0 for p in labels]
    svg_bar(plots / "planning_latency_p90.svg", "Planning latency p90", labels,
            plan_p90, "microseconds", max(plan_p90 + [1]))
    budgets = sorted({row.get("budget_fraction") for row in rows})
    budget_series = []
    for policy in policies:
        budget_series.append((policy, [
            (lambda subset: (clustered_rate(subset, "joint_success")
                             ["cluster_estimate"]["mean"] or 0.0)
             if subset else 0.0)([
                 r for r in rows if r.get("policy") == policy and
                 r.get("budget_fraction") == budget
             ]) for budget in budgets]))
    svg_lines(plots / "preparation_budget_sensitivity.svg", "Joint success vs preparation budget",
              [f"{b:.2g}" for b in budgets], budget_series, "rate", 1.0)
    pressures = sorted({row.get("pressure") for row in rows})
    pressure_series = []
    for policy in policies:
        pressure_series.append((policy, [
            (lambda subset: (clustered_rate(subset, "joint_success")
                             ["cluster_estimate"]["mean"] or 0.0)
             if subset else 0.0)([
                 r for r in rows if r.get("policy") == policy and
                 r.get("pressure") == pressure
             ]) for pressure in pressures]))
    svg_lines(plots / "memory_pressure_sensitivity.svg", "Joint success vs grant pressure",
              [f"{p:.2g}" for p in pressures], pressure_series, "rate", 1.0)
    families = sorted({row["family"] for row in rows})
    family_series = []
    for policy in policies:
        family_series.append((policy, [
            (lambda subset: (clustered_rate(subset, "joint_success")
                             ["cluster_estimate"]["mean"] or 0.0)
             if subset else 0.0)([
                 r for r in rows if r.get("policy") == policy and
                 r.get("family") == family
             ]) for family in families]))
    svg_lines(plots / "joint_success_by_family.svg", "Joint success by family",
              families, family_series, "rate", 1.0)
    scatter_points = []
    for row in rows:
        if not (row.get("committed") and row.get("a_ok") and
                row.get("a_slowdown_ratio") is not None and
                row.get("wait_b_benefit_us") is not None):
            continue
        scatter_points.append((
            f"{row['policy']}:{row['seed']}:{row['grant_profile']}",
            row["a_slowdown_ratio"], row["wait_b_benefit_us"]))
    svg_scatter(plots / "a_slowdown_vs_b_benefit.svg",
                "A slowdown vs fair-wait B benefit", scatter_points,
                "A slowdown ratio", "B benefit (microseconds)")
    calibration_labels = ["a_live", "b_live", "a_peak", "b_peak", "required_output"]
    calibration_keys = ["a_live_bytes_before", "b_live_bytes_before",
                        "a_peak_live_bytes", "b_peak_live_bytes",
                        "required_output_bytes"]
    calibration_values = [summary["resource_distributions"][key].get("median", 0)
                          or 0 for key in calibration_keys]
    svg_bar(plots / "residency_median.svg",
            "Median residency and required output", calibration_labels,
            calibration_values, "bytes", max(calibration_values + [1]))
    stages = summary["failure_stage"]
    stage_labels, stage_values = zip(*sorted(stages.items())) if stages else ([], [])
    svg_bar(plots / "failure_stages.svg", "Non-joint outcome stages", list(stage_labels),
            list(stage_values), "rows", max(stage_values or [1]))


if __name__ == "__main__":
    main()
