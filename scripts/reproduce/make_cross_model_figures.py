#!/usr/bin/env python3
"""Create paper-ready figures for cross-model localization robustness.

The script intentionally uses only the Python standard library so it can run on
minimal training machines where matplotlib or seaborn may not be installed.
It writes SVG figures plus companion CSVs with the plotted statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from html import escape
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Sequence, Tuple


OBJECTIVES = [
    ("lexical_binding", "Lexical"),
    ("factual_association", "Factual"),
    ("behavioral_policy", "Behavioral"),
    ("causal_mapping", "Causal"),
    ("procedural_reasoning", "Procedural"),
]

MODEL_ORDER = [
    ("llama_3_1_8b_instruct", "Llama-3.1\n8B"),
    ("mistral_7b_instruct_v0_3", "Mistral\n7B"),
    ("gemma_2_9b_it", "Gemma-2\n9B"),
    ("olmo_2_1124_7b_instruct", "OLMo-2\n7B"),
    ("qwen2_5_14b_instruct", "Qwen2.5\n14B"),
]

METRICS = [
    ("acquisition", "Acquisition"),
    ("transfer", "Transfer"),
    ("boundedness", "Boundedness"),
]

LOCALIZED = ["early", "middle", "late"]
CONDITION_LABEL = {"full": "Full", "early": "Early", "middle": "Middle", "late": "Late"}

REGION_COLORS = {"early": "#8dd3c7", "middle": "#80b1d3", "late": "#fb8072"}
MODEL_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#9c755f"]
MODEL_COLOR_BY_SLUG = {slug: MODEL_COLORS[i] for i, (slug, _label) in enumerate(MODEL_ORDER)}
TEXT = "#1f2933"
MUTED = "#5f6b7a"
GRID = "#d7dee8"
PAPER = "#ffffff"

T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def f(row: Dict[str, Any], key: str) -> float:
    return float(row[key])


def t_critical_95(n: int) -> float:
    if n <= 1:
        return float("nan")
    return T_CRITICAL_95.get(n - 1, 1.96)


def summarize_values(values: Sequence[float]) -> Dict[str, Any]:
    vals = [float(v) for v in values]
    n = len(vals)
    avg = mean(vals) if vals else float("nan")
    sd = stdev(vals) if n > 1 else 0.0
    sem = sd / math.sqrt(n) if n > 1 else 0.0
    ci = t_critical_95(n) * sem if n > 1 else 0.0
    return {
        "n": n,
        "mean": avg,
        "sd": sd,
        "sem": sem,
        "ci95": ci,
        "ci95_low": avg - ci,
        "ci95_high": avg + ci,
        "ci95_excludes_zero": (avg - ci > 0) or (avg + ci < 0),
    }


def binomial_one_sided_p(k: int, n: int) -> float | None:
    if n <= 0:
        return None
    return sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n)


def fmt_p(value: float | None) -> str:
    if value is None:
        return ""
    if value < 0.001:
        return "p<0.001"
    return f"p={value:.3f}"


def hex_to_rgb(color: str) -> Tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#" + "".join(f"{max(0, min(255, int(v))):02x}" for v in rgb)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def blend(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = hex_to_rgb(c1)
    r2, g2, b2 = hex_to_rgb(c2)
    return rgb_to_hex((lerp(r1, r2, t), lerp(g1, g2, t), lerp(b1, b2, t)))


def diverging_color(value: float, limit: float) -> str:
    if limit <= 0:
        return "#f7f9fc"
    x = max(-1.0, min(1.0, value / limit))
    if x >= 0:
        return blend("#f7f9fc", "#2166ac", x)
    return blend("#f7f9fc", "#b2182b", -x)


def agreement_color(rate: float) -> str:
    return blend("#f7f9fc", "#4e79a7", max(0.0, min(1.0, rate)))


def performance_color(value: float) -> str:
    value = max(0.0, min(1.0, value))
    if value < 0.5:
        return blend("#f7f9fc", "#b2182b", (0.5 - value) / 0.5)
    return blend("#f7f9fc", "#2166ac", (value - 0.5) / 0.5)


def text(
    x: float,
    y: float,
    value: str | Sequence[str],
    size: int = 13,
    anchor: str = "start",
    weight: int = 400,
    fill: str = TEXT,
    line_gap: int = 15,
) -> str:
    lines = [value] if isinstance(value, str) else list(value)
    attrs = (
        f'x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'text-anchor="{anchor}" font-weight="{weight}" fill="{fill}"'
    )
    if len(lines) == 1:
        return f"<text {attrs}>{escape(str(lines[0]))}</text>"
    spans = [f'<tspan x="{x:.1f}" dy="0">{escape(str(lines[0]))}</tspan>']
    for line in lines[1:]:
        spans.append(f'<tspan x="{x:.1f}" dy="{line_gap}">{escape(str(line))}</tspan>')
    return f"<text {attrs}>{''.join(spans)}</text>"


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", rx: float = 0) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" fill="{fill}" stroke="{stroke}"/>'


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = GRID, width: float = 1) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width:.1f}"/>'


def circle(cx: float, cy: float, r: float, fill: str, stroke: str = PAPER) -> str:
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1"/>'


def svg(width: int, height: int, body: Iterable[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
            "<style>text{font-family:Arial,Helvetica,sans-serif}.small{font-size:11px}</style>",
            rect(0, 0, width, height, PAPER),
            *body,
            "</svg>",
            "",
        ]
    )


def objective_label(objective: str) -> str:
    return dict(OBJECTIVES).get(objective, objective)


def metric_label(metric: str) -> str:
    return dict(METRICS).get(metric, metric)


def model_label(slug: str) -> str:
    return dict(MODEL_ORDER).get(slug, slug)


def load_summary(path: Path) -> List[Dict[str, Any]]:
    rows = read_csv(path)
    for row in rows:
        for key in ["budget", "seed", "rank"]:
            row[key] = int(row[key])
        for key in ["id_eval", "paraphrase_eval", "acquisition", "transfer", "generalization", "boundedness"]:
            row[key] = float(row[key])
        row["condition"] = row["condition"].lower()
    return rows


def delta_geometry_stats(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by = {(r["model_slug"], r["objective"], r["seed"], r["condition"]): r for r in rows}
    triples = sorted({(r["model_slug"], r["objective"], r["seed"]) for r in rows})
    values: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for model, objective, seed in triples:
        full = by.get((model, objective, seed, "full"))
        if not full:
            continue
        for condition in LOCALIZED:
            row = by.get((model, objective, seed, condition))
            if not row:
                continue
            for metric, _label in METRICS:
                values[(objective, metric, condition)].append(float(row[metric]) - float(full[metric]))

    out: List[Dict[str, Any]] = []
    for objective, _ in OBJECTIVES:
        for metric, _label in METRICS:
            for condition in LOCALIZED:
                stats = summarize_values(values.get((objective, metric, condition), []))
                out.append(
                    {
                        "objective": objective,
                        "metric": metric,
                        "condition": condition,
                        **stats,
                    }
                )
    return out


def full_stack_performance(summary: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    full_rows = [r for r in summary if r["condition"] == "full"]
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in full_rows:
        grouped[(row["model_slug"], row["objective"])].append(row)

    perf_rows: List[Dict[str, Any]] = []
    for model_slug_value, _model_label in MODEL_ORDER:
        for objective, _objective_label in OBJECTIVES:
            vals = grouped.get((model_slug_value, objective), [])
            if not vals:
                continue
            perf_rows.append(
                {
                    "model_slug": model_slug_value,
                    "model_label": model_label(model_slug_value).replace("\n", " "),
                    "objective": objective,
                    "budget": vals[0]["budget"],
                    "n_seeds": len({v["seed"] for v in vals}),
                    "full_acquisition": mean([float(v["acquisition"]) for v in vals]),
                    "full_transfer": mean([float(v["transfer"]) for v in vals]),
                    "full_boundedness": mean([float(v["boundedness"]) for v in vals]),
                }
            )

    llama = {r["objective"]: r for r in perf_rows if r["model_slug"] == "llama_3_1_8b_instruct"}
    diagnostics: List[Dict[str, Any]] = []
    for row in perf_rows:
        base = llama.get(row["objective"])
        acquisition_gap = float(row["full_acquisition"]) - float(base["full_acquisition"]) if base else None
        transfer_gap = float(row["full_transfer"]) - float(base["full_transfer"]) if base else None
        reasons: List[str] = []
        high = False
        consider = False
        if float(row["full_acquisition"]) < 0.60:
            high = True
            reasons.append("full acquisition < 0.60")
        elif float(row["full_acquisition"]) < 0.75:
            consider = True
            reasons.append("full acquisition < 0.75")
        if float(row["full_transfer"]) < 0.35:
            high = True
            reasons.append("full transfer < 0.35")
        elif float(row["full_transfer"]) < 0.50:
            consider = True
            reasons.append("full transfer < 0.50")
        if acquisition_gap is not None:
            if acquisition_gap <= -0.25:
                high = True
                reasons.append("acquisition >= 0.25 below Llama")
            elif acquisition_gap <= -0.15:
                consider = True
                reasons.append("acquisition >= 0.15 below Llama")
        if transfer_gap is not None:
            if transfer_gap <= -0.35:
                high = True
                reasons.append("transfer >= 0.35 below Llama")
            elif transfer_gap <= -0.20:
                consider = True
                reasons.append("transfer >= 0.20 below Llama")
        if float(row["full_boundedness"]) < 0.35:
            consider = True
            reasons.append("boundedness < 0.35")
        recommendation = "recalibrate_high_priority" if high else "recalibrate_consider" if consider else "budget_fit_ok"
        diagnostics.append(
            {
                **row,
                "llama_full_acquisition": base["full_acquisition"] if base else None,
                "llama_full_transfer": base["full_transfer"] if base else None,
                "acquisition_gap_vs_llama": acquisition_gap,
                "transfer_gap_vs_llama": transfer_gap,
                "recommendation": recommendation,
                "rationale": "; ".join(reasons) if reasons else "no severe mismatch by screening rule",
            }
        )
    return perf_rows, diagnostics


def draw_full_stack_performance(perf_rows: List[Dict[str, Any]], diagnostics: List[Dict[str, Any]], out: Path) -> None:
    width, height = 1220, 790
    left, top = 126, 140
    facet_w, cell_w, cell_h = 330, 58, 70
    facet_gap = 38
    body: List[str] = []
    body.append(text(28, 36, "Full-stack performance at Llama-selected budgets", 22, weight=500))
    body.append(text(28, 60, "Cells show mean score across seeds. Bold border marks model-objective pairs flagged for model-specific budget calibration.", 12, fill=MUTED))
    perf = {(r["objective"], r["model_slug"]): r for r in perf_rows}
    diag = {(r["objective"], r["model_slug"]): r for r in diagnostics}
    for fi, (metric, label) in enumerate(METRICS):
        source = "full_transfer" if metric == "transfer" else f"full_{metric}"
        x0 = left + fi * (facet_w + facet_gap)
        body.append(text(x0 + 2.5 * cell_w, top - 36, label, 15, anchor="middle", weight=500))
        for mi, (model_slug_value, label_lines) in enumerate(MODEL_ORDER):
            body.append(text(x0 + mi * cell_w + cell_w / 2, top - 18, label_lines.split("\n"), 10, anchor="middle", fill=MUTED, line_gap=11))
        for ri, (objective, obj_label) in enumerate(OBJECTIVES):
            y = top + ri * cell_h
            if fi == 0:
                body.append(text(28, y + 33, obj_label, 13, weight=500))
                budget = next((r["budget"] for r in perf_rows if r["objective"] == objective), "")
                body.append(text(28, y + 51, f"B={budget}", 11, fill=MUTED))
            for mi, (model_slug_value, _label) in enumerate(MODEL_ORDER):
                x = x0 + mi * cell_w
                row = perf.get((objective, model_slug_value))
                if not row:
                    continue
                value = float(row[source])
                rec = diag[(objective, model_slug_value)]["recommendation"]
                stroke = TEXT if rec == "recalibrate_high_priority" else "#8a95a5" if rec == "recalibrate_consider" else GRID
                sw = 2.2 if rec == "recalibrate_high_priority" else 1.5 if rec == "recalibrate_consider" else 1.0
                body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w - 5:.1f}" height="{cell_h - 8:.1f}" rx="5" fill="{performance_color(value)}" stroke="{stroke}" stroke-width="{sw}"/>')
                body.append(text(x + (cell_w - 5) / 2, y + 37, f"{value * 100:.1f}", 12, anchor="middle", weight=500))
    legend_x, legend_y = left, height - 74
    body.append(text(28, legend_y + 14, "Score", 12, fill=MUTED))
    for i in range(41):
        v = i / 40
        body.append(rect(legend_x + i * 9, legend_y, 9, 14, performance_color(v)))
    body.append(text(legend_x, legend_y + 34, "0", 11, fill=MUTED))
    body.append(text(legend_x + 184, legend_y + 34, "0.5", 11, anchor="middle", fill=MUTED))
    body.append(text(legend_x + 369, legend_y + 34, "1", 11, anchor="end", fill=MUTED))
    body.append(text(width - 28, height - 39, "Screening rule: high priority if full acquisition <0.60, full transfer <0.35, or >=0.25 acquisition / >=0.35 transfer below Llama.", 11, anchor="end", fill=MUTED))
    write_text(out, svg(width, height, body))


def draw_delta_heatmap(stats: List[Dict[str, Any]], out: Path) -> None:
    width, height = 1180, 730
    margin_l, top = 138, 128
    facet_w, cell_w, cell_h = 315, 90, 70
    row_gap, facet_gap = 8, 34
    body: List[str] = []
    body.append(text(28, 36, "Cross-model localized-minus-full geometry", 22, weight=500))
    body.append(text(28, 60, "Cells show mean paired Delta in percentage points; +/- is 95% CI across model-seed pairs; * marks CI excluding 0.", 12, fill=MUTED))

    max_abs = max(abs(float(r["mean"])) for r in stats if r["n"]) or 1.0
    max_abs = max(max_abs, 0.01)
    by = {(r["objective"], r["metric"], r["condition"]): r for r in stats}

    for fi, (metric, label) in enumerate(METRICS):
        x0 = margin_l + fi * (facet_w + facet_gap)
        body.append(text(x0 + 1.5 * cell_w, top - 28, label, 15, anchor="middle", weight=500))
        for ci, condition in enumerate(LOCALIZED):
            body.append(text(x0 + ci * cell_w + cell_w / 2, top - 8, CONDITION_LABEL[condition], 12, anchor="middle", fill=MUTED))
        for ri, (objective, obj_label) in enumerate(OBJECTIVES):
            y = top + ri * (cell_h + row_gap)
            if fi == 0:
                body.append(text(28, y + 30, obj_label, 13, weight=500))
            for ci, condition in enumerate(LOCALIZED):
                x = x0 + ci * cell_w
                row = by[(objective, metric, condition)]
                fill = diverging_color(float(row["mean"]), max_abs)
                body.append(rect(x, y, cell_w - 5, cell_h, fill, stroke=GRID, rx=5))
                value = f"{float(row['mean']) * 100:+.1f}"
                ci_value = f"+/-{float(row['ci95']) * 100:.1f}"
                star = "*" if row["ci95_excludes_zero"] else ""
                body.append(text(x + (cell_w - 5) / 2, y + 27, value + star, 14, anchor="middle", weight=500))
                body.append(text(x + (cell_w - 5) / 2, y + 47, ci_value, 11, anchor="middle", fill=MUTED))

    legend_x, legend_y = margin_l, height - 82
    body.append(text(28, legend_y + 16, "Delta scale", 12, fill=MUTED))
    for i in range(41):
        v = -max_abs + (2 * max_abs) * i / 40
        body.append(rect(legend_x + i * 10, legend_y, 10, 14, diverging_color(v, max_abs)))
    body.append(text(legend_x, legend_y + 34, f"{-max_abs * 100:.0f} pp", 11, fill=MUTED))
    body.append(text(legend_x + 205, legend_y + 34, "0", 11, anchor="middle", fill=MUTED))
    body.append(text(legend_x + 410, legend_y + 34, f"+{max_abs * 100:.0f} pp", 11, anchor="end", fill=MUTED))
    n_values = sorted({int(r["n"]) for r in stats if r["n"]})
    body.append(text(width - 28, height - 48, f"n={','.join(map(str, n_values))} paired model-seed deltas per cell", 11, anchor="end", fill=MUTED))
    write_text(out, svg(width, height, body))


def draw_best_region_agreement(rows: List[Dict[str, str]], out: Path) -> None:
    width, height = 920, 560
    left, top, cell_w, cell_h = 160, 118, 190, 62
    body: List[str] = []
    body.append(text(28, 36, "Best localized region agreement across models", 22, weight=500))
    body.append(text(28, 60, "Each cell shows modal best localized window and k/N model agreement; fill intensity encodes agreement rate.", 12, fill=MUTED))
    by = {(r["objective"], r["metric"]): r for r in rows}
    for ci, (metric, label) in enumerate(METRICS):
        body.append(text(left + ci * cell_w + cell_w / 2, top - 16, label, 13, anchor="middle", weight=500))
    for ri, (objective, obj_label) in enumerate(OBJECTIVES):
        y = top + ri * cell_h
        body.append(text(28, y + 35, obj_label, 13, weight=500))
        for ci, (metric, _label) in enumerate(METRICS):
            x = left + ci * cell_w
            row = by.get((objective, metric))
            if not row:
                body.append(rect(x, y, cell_w - 8, cell_h - 8, "#f7f9fc", stroke=GRID, rx=5))
                body.append(text(x + cell_w / 2, y + 33, "missing", 12, anchor="middle", fill=MUTED))
                continue
            rate = float(row["agreement_rate"])
            fill = agreement_color(rate)
            region = CONDITION_LABEL.get(row["most_common_best_region"], row["most_common_best_region"])
            score = f"{row['agreement_count']}/{row['n_models']}"
            body.append(rect(x, y, cell_w - 8, cell_h - 8, fill, stroke=GRID, rx=5))
            body.append(text(x + (cell_w - 8) / 2, y + 25, region, 14, anchor="middle", weight=500))
            body.append(text(x + (cell_w - 8) / 2, y + 44, score, 12, anchor="middle", fill=MUTED))
    legend_x, legend_y = left, height - 58
    body.append(text(28, legend_y + 13, "Agreement", 12, fill=MUTED))
    for i in range(31):
        rate = i / 30
        body.append(rect(legend_x + i * 11, legend_y, 11, 14, agreement_color(rate)))
    body.append(text(legend_x, legend_y + 34, "0", 11, fill=MUTED))
    body.append(text(legend_x + 341, legend_y + 34, "1", 11, anchor="end", fill=MUTED))
    write_text(out, svg(width, height, body))


def parse_model_deltas(value: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in value.split(";"):
        if "=" not in part:
            continue
        model, val = part.strip().split("=", 1)
        out[model] = float(val)
    return out


def draw_mislocation(rows: List[Dict[str, str]], out: Path) -> None:
    width, height = 1100, 680
    left, right, top, row_h = 265, 1000, 130, 72
    body: List[str] = []
    body.append(text(28, 36, "Primary mislocation penalties replicate across models", 22, weight=500))
    body.append(text(28, 60, "Bars show mean preferred-minus-mislocated Delta across model means; colored points are individual models; p_sign is one-sided sign-test p under H0=0.5.", 12, fill=MUTED))

    parsed = []
    for row in rows:
        deltas = parse_model_deltas(row["model_level_deltas"])
        vals = list(deltas.values())
        stats = summarize_values(vals)
        k, n = int(row["n_positive"]), int(row["n_models"])
        parsed.append((row, deltas, stats, binomial_one_sided_p(k, n)))
    min_x = min(min(d.values()) for _r, d, _s, _p in parsed)
    max_x = max(max(d.values()) for _r, d, _s, _p in parsed)
    lim = max(abs(min_x), abs(max_x), 0.02) * 1.15

    def sx(v: float) -> float:
        return left + (v + lim) / (2 * lim) * (right - left)

    zero_x = sx(0)
    body.append(line(left, top - 8, right, top - 8, GRID))
    for tick in [-lim, -lim / 2, 0, lim / 2, lim]:
        x = sx(tick)
        body.append(line(x, top - 14, x, top + row_h * len(parsed) - 20, "#edf1f6"))
        body.append(text(x, top - 22, f"{tick * 100:+.0f}", 10, anchor="middle", fill=MUTED))
    body.append(text((left + right) / 2, top - 44, "Preferred - mislocated (percentage points)", 12, anchor="middle", fill=MUTED))
    body.append(line(zero_x, top - 14, zero_x, top + row_h * len(parsed) - 20, "#8a95a5", 1.2))

    for i, (row, deltas, stats, pval) in enumerate(parsed):
        y = top + i * row_h
        label = objective_label(row["objective"])
        contrast = f"{CONDITION_LABEL[row['preferred_condition']]} > {CONDITION_LABEL[row['mislocated_condition']]}"
        body.append(text(28, y + 12, label, 13, weight=500))
        body.append(text(28, y + 31, f"{metric_label(row['primary_metric'])}: {contrast}", 11, fill=MUTED))
        mean_x = sx(float(stats["mean"]))
        bar_x = min(zero_x, mean_x)
        body.append(rect(bar_x, y + 26, abs(mean_x - zero_x), 14, "#4e79a7", rx=2))
        ci_l, ci_h = sx(float(stats["ci95_low"])), sx(float(stats["ci95_high"]))
        body.append(line(ci_l, y + 33, ci_h, y + 33, TEXT, 1.2))
        body.append(line(ci_l, y + 27, ci_l, y + 39, TEXT, 1.2))
        body.append(line(ci_h, y + 27, ci_h, y + 39, TEXT, 1.2))
        ordered_models = [(slug, label) for slug, label in MODEL_ORDER if slug in deltas]
        for j, (model_slug_value, _label) in enumerate(ordered_models):
            jitter = (j - (len(ordered_models) - 1) / 2) * 2.2
            body.append(circle(sx(deltas[model_slug_value]), y + 54 + jitter, 4.2, MODEL_COLOR_BY_SLUG[model_slug_value]))
        p_text = "" if pval is None else f"p_sign{fmt_p(pval).removeprefix('p')}"
        body.append(text(width - 28, y + 37, f"{row['n_positive']}/{row['n_models']}; {p_text}", 12, anchor="end", fill=MUTED))

    legend_y = height - 86
    body.append(text(28, legend_y, "Legend", 12, weight=500))
    body.append(rect(92, legend_y - 10, 28, 10, "#4e79a7", rx=2))
    body.append(text(127, legend_y, "Mean Delta", 11, fill=MUTED))
    body.append(line(218, legend_y - 5, 262, legend_y - 5, TEXT, 1.2))
    body.append(text(270, legend_y, "95% CI across models", 11, fill=MUTED))
    model_x = 410
    body.append(text(model_x, legend_y, "Model points:", 11, fill=MUTED))
    x = model_x + 82
    for model_slug_value, label in MODEL_ORDER:
        body.append(circle(x, legend_y - 5, 4.2, MODEL_COLOR_BY_SLUG[model_slug_value]))
        body.append(text(x + 8, legend_y, label.replace("\n", " "), 11, fill=MUTED))
        x += 116
    write_text(out, svg(width, height, body))


def draw_similarity_variance(sim_rows: List[Dict[str, str]], variance: Dict[str, Any], perm: Dict[str, Any], out: Path) -> None:
    width, height = 1020, 600
    body: List[str] = []
    body.append(text(28, 36, "Cross-model geometry similarity and variance components", 22, weight=500))
    body.append(text(28, 60, "Cosine bars show mean +/- SD across model pairs. Variance components are descriptive ratios over DeltaG dimensions.", 12, fill=MUTED))

    left, top, plot_w, row_h = 210, 140, 360, 58
    body.append(text(left + plot_w / 2, top - 42, "Pairwise cosine similarity", 14, anchor="middle", weight=500))
    body.append(line(left, top - 4, left + plot_w, top - 4, GRID))
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        x = left + tick * plot_w
        body.append(line(x, top - 10, x, top + row_h * len(OBJECTIVES) - 8, "#edf1f6"))
        body.append(text(x, top - 18, f"{tick:.2g}", 10, anchor="middle", fill=MUTED))
    sim_by_obj = {r["objective"]: r for r in sim_rows}
    for i, (objective, label) in enumerate(OBJECTIVES):
        y = top + i * row_h
        row = sim_by_obj.get(objective)
        body.append(text(28, y + 21, label, 13, weight=500))
        if not row:
            continue
        m = float(row["cosine_mean"])
        sd = float(row["cosine_sd"] or 0)
        x0, x1 = left, left + m * plot_w
        body.append(rect(x0, y + 8, x1 - x0, 18, "#4e79a7", rx=2))
        lo, hi = max(0, m - sd), min(1, m + sd)
        body.append(line(left + lo * plot_w, y + 17, left + hi * plot_w, y + 17, TEXT, 1.2))
        body.append(text(left + plot_w + 12, y + 22, f"{m:.2f} +/- {sd:.2f}", 11, fill=MUTED))

    comp_x, comp_y, comp_w = 640, 125, 315
    body.append(text(comp_x + comp_w / 2, comp_y - 48, "Variance decomposition", 14, anchor="middle", weight=500))
    components = [
        ("Objective", float(variance.get("objective_explained_ratio") or 0), "#59a14f"),
        ("Model", float(variance.get("model_explained_ratio") or 0), "#4e79a7"),
        ("Residual", max(0.0, 1.0 - float(variance.get("objective_explained_ratio") or 0) - float(variance.get("model_explained_ratio") or 0)), "#bab0ab"),
    ]
    y = comp_y
    x = comp_x
    for label, ratio, color in components:
        w = ratio * comp_w
        body.append(rect(x, y, w, 34, color, rx=2 if x == comp_x else 0))
        if w > 54:
            body.append(text(x + w / 2, y + 22, f"{ratio * 100:.0f}%", 12, anchor="middle", fill=PAPER, weight=500))
        x += w
    y += 60
    for label, ratio, color in components:
        body.append(rect(comp_x, y - 11, 12, 12, color, rx=2))
        body.append(text(comp_x + 20, y, f"{label}: {ratio:.3f}", 12, fill=MUTED))
        y += 24

    obj_p = perm.get("objective_label_permutation", {}).get("p_value")
    model_p = perm.get("model_label_permutation", {}).get("p_value")
    body.append(text(comp_x, y + 12, [f"Permutation objective {fmt_p(obj_p)}", f"Permutation model {fmt_p(model_p)}", f"Larger component: {variance.get('key_question_answer')}"], 12, fill=MUTED, line_gap=18))
    write_text(out, svg(width, height, body))


def svg_dimensions(path: Path) -> Tuple[int, int]:
    raw = path.read_text(encoding="utf-8")
    width = re.search(r'<svg[^>]*\bwidth="([0-9.]+)"', raw)
    height = re.search(r'<svg[^>]*\bheight="([0-9.]+)"', raw)
    if not width or not height:
        raise ValueError(f"Could not read SVG dimensions from {path}")
    return int(float(width.group(1))), int(float(height.group(1)))


def find_chrome(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    program_files = os.environ.get("ProgramFiles")
    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    candidates = [
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("chrome"),
        shutil.which("msedge"),
        str(Path(program_files) / "Google/Chrome/Application/chrome.exe") if program_files else None,
        str(Path(program_files_x86) / "Google/Chrome/Application/chrome.exe") if program_files_x86 else None,
        str(Path(program_files) / "Microsoft/Edge/Application/msedge.exe") if program_files else None,
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def html_wrapper(svg_path: Path, width: int, height: int, scale: float, for_pdf: bool) -> str:
    svg_markup = svg_path.read_text(encoding="utf-8")
    css_width = width * scale
    css_height = height * scale
    page_rule = f"@page {{ size: {width}px {height}px; margin: 0; }}" if for_pdf else ""
    return "\n".join(
        [
            "<!doctype html>",
            "<html>",
            "<head>",
            '<meta charset="utf-8">',
            "<style>",
            page_rule,
            "html, body { margin: 0; padding: 0; background: white; }",
            f"svg {{ width: {css_width:.0f}px; height: {css_height:.0f}px; display: block; }}",
            "</style>",
            "</head>",
            "<body>",
            svg_markup,
            "</body>",
            "</html>",
        ]
    )


def export_with_chrome(svg_paths: Sequence[Path], formats: Sequence[str], chrome_exe: str | None, png_scale: float) -> List[Path]:
    requested = {fmt.lower() for fmt in formats}
    if not ({"png", "pdf"} & requested):
        return []
    chrome = find_chrome(chrome_exe)
    if chrome is None:
        raise RuntimeError("PNG/PDF export requires Chrome/Chromium. Pass --chrome_exe or run with --formats svg.")

    exported: List[Path] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        user_data = tmpdir / "chrome-profile"
        for svg_path in svg_paths:
            width, height = svg_dimensions(svg_path)
            if "png" in requested:
                html_path = tmpdir / f"{svg_path.stem}_png.html"
                html_path.write_text(html_wrapper(svg_path, width, height, png_scale, for_pdf=False), encoding="utf-8")
                png_path = svg_path.with_suffix(".png")
                cmd = [
                    chrome,
                    "--headless=new",
                    "--disable-gpu",
                    "--no-sandbox",
                    "--hide-scrollbars",
                    "--no-first-run",
                    f"--user-data-dir={user_data}",
                    f"--window-size={int(width * png_scale)},{int(height * png_scale)}",
                    f"--screenshot={png_path.resolve()}",
                    html_path.resolve().as_uri(),
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                exported.append(png_path)
            if "pdf" in requested:
                html_path = tmpdir / f"{svg_path.stem}_pdf.html"
                html_path.write_text(html_wrapper(svg_path, width, height, 1.0, for_pdf=True), encoding="utf-8")
                pdf_path = svg_path.with_suffix(".pdf")
                cmd = [
                    chrome,
                    "--headless=new",
                    "--disable-gpu",
                    "--no-sandbox",
                    "--no-first-run",
                    f"--user-data-dir={user_data}",
                    f"--print-to-pdf={pdf_path.resolve()}",
                    "--print-to-pdf-no-header",
                    html_path.resolve().as_uri(),
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                exported.append(pdf_path)
    return exported


def write_manifest(out_dir: Path, files: Sequence[Path]) -> None:
    rows = [{"figure": path.name, "path": str(path), "format": path.suffix.lstrip(".")} for path in files]
    write_csv(out_dir / "cross_model_figure_manifest.csv", rows, ["figure", "path", "format"])
    tex_lines = [
        "% Auto-generated figure include snippets.\n",
        "% Prefer PDF when present for LaTeX workflows.\n",
    ]
    stems = []
    for path in files:
        if path.suffix == ".svg" and path.stem not in stems:
            stems.append(path.stem)
    for stem in stems:
        pdf = out_dir / f"{stem}.pdf"
        svg_path = out_dir / f"{stem}.svg"
        include = pdf if pdf.exists() else svg_path
        tex_lines.append(f"% {include.name}\n")
        tex_lines.append(f"\\includegraphics[width=\\linewidth]{{{include.as_posix()}}}\n\n")
    write_text(out_dir / "cross_model_figure_includes.tex", "".join(tex_lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Make cross-model localization figures.")
    parser.add_argument("--summary", type=Path, default=Path("artifacts/localization/cross_model/summary_cross_model_by_seed.csv"))
    parser.add_argument("--geometry_dir", type=Path, default=Path("artifacts/localization/cross_model/geometry_analysis"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/release_verification/figures/localization/cross_model"))
    parser.add_argument("--formats", nargs="+", default=["svg", "png", "pdf"], choices=["svg", "png", "pdf"], help="Figure formats to write. SVG is always generated as the source format.")
    parser.add_argument("--chrome_exe", default=None, help="Optional Chrome/Chromium executable for PNG/PDF export.")
    parser.add_argument("--png_scale", type=float, default=2.0, help="Raster scale factor for PNG export.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = load_summary(args.summary)
    delta_stats = delta_geometry_stats(summary)
    write_csv(
        args.output_dir / "cross_model_delta_geometry_stats.csv",
        delta_stats,
        ["objective", "metric", "condition", "n", "mean", "sd", "sem", "ci95", "ci95_low", "ci95_high", "ci95_excludes_zero"],
    )

    best = read_csv(args.geometry_dir / "cross_model_best_region_agreement.csv")
    sign = read_csv(args.geometry_dir / "cross_model_mislocation_sign_agreement.csv")
    sim = read_csv(args.geometry_dir / "cross_model_geometry_similarity_summary.csv")
    variance = json.loads((args.geometry_dir / "cross_model_variance_decomposition.json").read_text(encoding="utf-8"))
    perm = json.loads((args.geometry_dir / "cross_model_permutation_tests.json").read_text(encoding="utf-8"))

    figure_paths = [
        args.output_dir / "cross_model_full_stack_performance.svg",
        args.output_dir / "cross_model_delta_geometry_heatmap.svg",
        args.output_dir / "cross_model_best_region_agreement.svg",
        args.output_dir / "cross_model_primary_mislocation.svg",
        args.output_dir / "cross_model_similarity_variance.svg",
    ]
    perf_rows, diagnostics = full_stack_performance(summary)
    write_csv(
        args.output_dir / "cross_model_full_stack_performance.csv",
        perf_rows,
        ["model_slug", "model_label", "objective", "budget", "n_seeds", "full_acquisition", "full_transfer", "full_boundedness"],
    )
    write_csv(
        args.output_dir / "cross_model_calibration_mismatch_diagnostics.csv",
        diagnostics,
        [
            "model_slug",
            "model_label",
            "objective",
            "budget",
            "n_seeds",
            "full_acquisition",
            "full_transfer",
            "full_boundedness",
            "llama_full_acquisition",
            "llama_full_transfer",
            "acquisition_gap_vs_llama",
            "transfer_gap_vs_llama",
            "recommendation",
            "rationale",
        ],
    )

    draw_full_stack_performance(perf_rows, diagnostics, figure_paths[0])
    draw_delta_heatmap(delta_stats, figure_paths[1])
    draw_best_region_agreement(best, figure_paths[2])
    draw_mislocation(sign, figure_paths[3])
    draw_similarity_variance(sim, variance, perm, figure_paths[4])
    exported_paths = export_with_chrome(figure_paths, args.formats, args.chrome_exe, args.png_scale)
    write_manifest(args.output_dir, [*figure_paths, *exported_paths])

    print(f"Wrote {len(figure_paths)} SVG figures and {len(exported_paths)} exported files to {args.output_dir}")


if __name__ == "__main__":
    main()
