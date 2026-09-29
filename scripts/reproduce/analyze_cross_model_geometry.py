#!/usr/bin/env python3
"""Analyze cross-model adaptation-geometry robustness."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Sequence, Tuple


METRICS = ["acquisition", "transfer", "boundedness"]
LOCALIZED = ["early", "middle", "late"]
PROFILE_DIMS = [
    "acquisition_early_minus_full",
    "acquisition_middle_minus_full",
    "acquisition_late_minus_full",
    "transfer_early_minus_full",
    "transfer_middle_minus_full",
    "transfer_late_minus_full",
    "boundedness_early_minus_full",
    "boundedness_middle_minus_full",
    "boundedness_late_minus_full",
]

MISLOCATION_CONTRASTS = [
    ("lexical_binding", "lexical_early_vs_late", "early", "late"),
    ("lexical_binding", "lexical_early_vs_middle", "early", "middle"),
    ("factual_association", "factual_late_vs_early", "late", "early"),
    ("factual_association", "factual_late_vs_middle", "late", "middle"),
    ("behavioral_policy", "behavioral_gating_middle_vs_late", "middle", "late"),
    ("behavioral_policy", "behavioral_acquisition_late_vs_middle", "late", "middle"),
    ("causal_mapping", "causal_middle_vs_early", "middle", "early"),
    ("causal_mapping", "causal_middle_vs_late", "middle", "late"),
    ("procedural_reasoning", "procedural_middle_vs_early", "middle", "early"),
    ("procedural_reasoning", "procedural_middle_vs_late", "middle", "late"),
]

PRIMARY_SIGN_DIAGNOSTICS = [
    ("lexical_binding", "acquisition", "early", "late"),
    ("factual_association", "transfer", "late", "early"),
    ("behavioral_policy", "boundedness", "middle", "late"),
    ("causal_mapping", "transfer", "middle", "early"),
    ("procedural_reasoning", "transfer", "middle", "early"),
]


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def f(row: Dict[str, Any], key: str) -> float:
    return float(row[key])


def sd(values: List[float]) -> float | None:
    return stdev(values) if len(values) > 1 else None


def cosine(a: Sequence[float], b: Sequence[float]) -> float | None:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else None


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def row_key(row: Dict[str, str]) -> Tuple[str, str, str, str]:
    return (row["model_slug"], row["objective"], str(row["seed"]), row["condition"].lower())


def compute_profiles(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    by_cond = {row_key(row): row for row in rows}
    triples = sorted({(r["model_name"], r["model_slug"], r["objective"], str(r["seed"])) for r in rows})
    profiles: List[Dict[str, Any]] = []
    for model_name, model_slug, objective, seed in triples:
        full = by_cond.get((model_slug, objective, seed, "full"))
        if full is None:
            continue
        vec: Dict[str, Any] = {"model_name": model_name, "model_slug": model_slug, "objective": objective, "seed": seed}
        complete = True
        for metric in METRICS:
            for condition in LOCALIZED:
                row = by_cond.get((model_slug, objective, seed, condition))
                if row is None:
                    complete = False
                    break
                vec[f"{metric}_{condition}_minus_full"] = f(row, metric) - f(full, metric)
            if not complete:
                break
        if complete:
            profiles.append(vec)
    return profiles


def vector(profile: Dict[str, Any]) -> List[float]:
    return [float(profile[dim]) for dim in PROFILE_DIMS]


def mean_vector(profiles: List[Dict[str, Any]]) -> List[float]:
    return [mean([float(p[dim]) for p in profiles]) for dim in PROFILE_DIMS]


def best_regions(rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str, str, str, str], List[float]] = defaultdict(list)
    model_names: Dict[str, str] = {}
    for row in rows:
        model_names[row["model_slug"]] = row["model_name"]
        for metric in METRICS:
            grouped[(row["model_slug"], row["objective"], row["condition"].lower(), metric, "value")].append(f(row, metric))

    best_rows: List[Dict[str, Any]] = []
    for model_slug in sorted(model_names):
        objectives = sorted({k[1] for k in grouped if k[0] == model_slug})
        for objective in objectives:
            for metric in METRICS:
                full_vals = grouped.get((model_slug, objective, "full", metric, "value"), [])
                loc_means = {}
                for condition in LOCALIZED:
                    vals = grouped.get((model_slug, objective, condition, metric, "value"), [])
                    if vals:
                        loc_means[condition] = mean(vals)
                if not full_vals or not loc_means:
                    continue
                best = max(loc_means, key=loc_means.get)
                best_rows.append(
                    {
                        "model_slug": model_slug,
                        "objective": objective,
                        "metric": metric,
                        "best_localized_condition": best,
                        "best_localized_value": loc_means[best],
                        "full_value": mean(full_vals),
                        "localized_advantage": loc_means[best] - mean(full_vals),
                    }
                )

    agreement_rows: List[Dict[str, Any]] = []
    for objective in sorted({r["objective"] for r in best_rows}):
        for metric in METRICS:
            sub = [r for r in best_rows if r["objective"] == objective and r["metric"] == metric]
            if not sub:
                continue
            regions = [r["best_localized_condition"] for r in sub]
            counts = Counter(regions)
            most_common, count = counts.most_common(1)[0]
            agreement_rows.append(
                {
                    "objective": objective,
                    "metric": metric,
                    "most_common_best_region": most_common,
                    "agreement_count": count,
                    "n_models": len(sub),
                    "agreement_rate": count / len(sub),
                    "all_model_regions": "; ".join(f"{r['model_slug']}={r['best_localized_condition']}" for r in sub),
                }
            )
    return best_rows, agreement_rows


def geometry_similarity(profiles: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pair_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for objective in sorted({p["objective"] for p in profiles}):
        by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for p in profiles:
            if p["objective"] == objective:
                by_model[p["model_slug"]].append(p)
        model_vecs = {m: mean_vector(ps) for m, ps in by_model.items()}
        models = sorted(model_vecs)
        for i, model_i in enumerate(models):
            for model_j in models[i + 1:]:
                c = cosine(model_vecs[model_i], model_vecs[model_j])
                d = euclidean(model_vecs[model_i], model_vecs[model_j])
                pair_rows.append(
                    {
                        "objective": objective,
                        "model_i": model_i,
                        "model_j": model_j,
                        "cosine_similarity": c,
                        "euclidean_distance": d,
                    }
                )
        cos_vals = [float(r["cosine_similarity"]) for r in pair_rows if r["objective"] == objective and r["cosine_similarity"] is not None]
        dist_vals = [float(r["euclidean_distance"]) for r in pair_rows if r["objective"] == objective]
        summary_rows.append(
            {
                "objective": objective,
                "cosine_mean": mean(cos_vals) if cos_vals else None,
                "cosine_sd": sd(cos_vals),
                "cosine_min": min(cos_vals) if cos_vals else None,
                "cosine_max": max(cos_vals) if cos_vals else None,
                "euclidean_mean": mean(dist_vals) if dist_vals else None,
                "euclidean_sd": sd(dist_vals),
                "euclidean_min": min(dist_vals) if dist_vals else None,
                "euclidean_max": max(dist_vals) if dist_vals else None,
            }
        )
    return pair_rows, summary_rows


def between_group_ss(X: List[List[float]], labels: List[str]) -> float:
    if not X:
        return 0.0
    grand = [mean(col) for col in zip(*X)]
    total = 0.0
    for label in sorted(set(labels)):
        group = [x for x, lab in zip(X, labels) if lab == label]
        mu = [mean(col) for col in zip(*group)]
        total += len(group) * sum((a - b) ** 2 for a, b in zip(mu, grand))
    return total


def variance_decomposition(profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    X = [vector(p) for p in profiles]
    if not X:
        return {"note": "No complete profiles."}
    grand = [mean(col) for col in zip(*X)]
    total_ss = sum(sum((v - g) ** 2 for v, g in zip(x, grand)) for x in X)
    obj_ss = between_group_ss(X, [p["objective"] for p in profiles])
    model_ss = between_group_ss(X, [p["model_slug"] for p in profiles])
    residual = total_ss - obj_ss - model_ss
    return {
        "n_profiles": len(profiles),
        "total_variance_ss": total_ss,
        "objective_explained_ss": obj_ss,
        "model_explained_ss": model_ss,
        "residual_interaction_proxy_ss": residual,
        "objective_explained_ratio": obj_ss / total_ss if total_ss else None,
        "model_explained_ratio": model_ss / total_ss if total_ss else None,
        "key_question_answer": "objective" if obj_ss > model_ss else "model",
        "note": "Descriptive additive sum-of-squares proxy over DeltaG profile dimensions.",
    }


def permutation_test(profiles: List[Dict[str, Any]], label_key: str, n_perm: int, rng: random.Random) -> Dict[str, Any]:
    X = [vector(p) for p in profiles]
    labels = [str(p[label_key]) for p in profiles]
    observed = between_group_ss(X, labels)
    if not X or len(set(labels)) < 2:
        return {"observed_stat": observed, "p_value": None, "n_perm": n_perm, "note": "Insufficient groups."}
    hits = 0
    for _ in range(n_perm):
        shuffled = labels[:]
        rng.shuffle(shuffled)
        if between_group_ss(X, shuffled) >= observed:
            hits += 1
    return {"observed_stat": observed, "p_value": (hits + 1) / (n_perm + 1), "n_perm": n_perm}


def mislocation(rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    by = {(r["model_slug"], r["objective"], str(r["seed"]), r["condition"].lower()): r for r in rows}
    by_seed: List[Dict[str, Any]] = []
    for objective, contrast_id, preferred, mislocated in MISLOCATION_CONTRASTS:
        keys = sorted({(r["model_name"], r["model_slug"], str(r["seed"])) for r in rows if r["objective"] == objective})
        for model_name, model_slug, seed in keys:
            pref_row = by.get((model_slug, objective, seed, preferred))
            mis_row = by.get((model_slug, objective, seed, mislocated))
            if not pref_row or not mis_row:
                continue
            for metric in METRICS:
                by_seed.append(
                    {
                        "model_name": model_name,
                        "model_slug": model_slug,
                        "objective": objective,
                        "contrast_id": contrast_id,
                        "metric": metric,
                        "preferred_condition": preferred,
                        "mislocated_condition": mislocated,
                        "seed": seed,
                        "preferred_value": f(pref_row, metric),
                        "mislocated_value": f(mis_row, metric),
                        "delta_preferred_minus_mislocated": f(pref_row, metric) - f(mis_row, metric),
                    }
                )

    grouped: Dict[Tuple[str, str, str, str, str, str], List[float]] = defaultdict(list)
    for r in by_seed:
        grouped[(r["model_slug"], r["objective"], r["contrast_id"], r["metric"], r["preferred_condition"], r["mislocated_condition"])].append(
            float(r["delta_preferred_minus_mislocated"])
        )
    summary = [
        {
            "model_slug": k[0],
            "objective": k[1],
            "contrast_id": k[2],
            "metric": k[3],
            "preferred_condition": k[4],
            "mislocated_condition": k[5],
            "delta_mean": mean(v),
            "delta_sd": sd(v),
            "n_seeds": len(v),
        }
        for k, v in sorted(grouped.items())
    ]

    sign_rows: List[Dict[str, Any]] = []
    for objective, metric, preferred, mislocated in PRIMARY_SIGN_DIAGNOSTICS:
        model_deltas: Dict[str, List[float]] = defaultdict(list)
        for r in by_seed:
            if r["objective"] == objective and r["metric"] == metric and r["preferred_condition"] == preferred and r["mislocated_condition"] == mislocated:
                model_deltas[r["model_slug"]].append(float(r["delta_preferred_minus_mislocated"]))
        model_means = {m: mean(v) for m, v in model_deltas.items() if v}
        n_positive = sum(1 for v in model_means.values() if v > 0)
        n_models = len(model_means)
        sign_rows.append(
            {
                "objective": objective,
                "primary_metric": metric,
                "preferred_condition": preferred,
                "mislocated_condition": mislocated,
                "n_models": n_models,
                "n_positive": n_positive,
                "agreement_rate": n_positive / n_models if n_models else None,
                "model_level_deltas": "; ".join(f"{m}={v:.4f}" for m, v in sorted(model_means.items())),
            }
        )
    return by_seed, summary, sign_rows


def latex_escape(x: Any) -> str:
    return str(x).replace("_", r"\_")


def write_latex_tables(out: Path, best_agreement: List[Dict[str, Any]], sign_rows: List[Dict[str, Any]], sim_summary: List[Dict[str, Any]]) -> None:
    lines = ["\\begin{tabular}{llll}\n\\toprule\nObjective & Metric & Most common best localized region & Agreement \\\\\n\\midrule\n"]
    for r in best_agreement:
        lines.append(f"{latex_escape(r['objective'])} & {latex_escape(r['metric'])} & {latex_escape(r['most_common_best_region'])} & {r['agreement_count']}/{r['n_models']} \\\\\n")
    lines.append("\\bottomrule\n\\end{tabular}\n")
    (out / "cross_model_best_region_agreement_table.tex").write_text("".join(lines), encoding="utf-8")

    lines = ["\\begin{tabular}{llll}\n\\toprule\nObjective & Primary metric & Preferred $>$ Mislocated & Models replicated \\\\\n\\midrule\n"]
    for r in sign_rows:
        contrast = f"{r['preferred_condition']} > {r['mislocated_condition']}"
        lines.append(f"{latex_escape(r['objective'])} & {latex_escape(r['primary_metric'])} & {latex_escape(contrast)} & {r['n_positive']}/{r['n_models']} \\\\\n")
    lines.append("\\bottomrule\n\\end{tabular}\n")
    (out / "cross_model_mislocation_sign_agreement_table.tex").write_text("".join(lines), encoding="utf-8")

    lines = ["\\begin{tabular}{lrr}\n\\toprule\nObjective & Mean cosine similarity & Mean Euclidean distance \\\\\n\\midrule\n"]
    for r in sim_summary:
        cos_v = "" if r["cosine_mean"] is None else f"{float(r['cosine_mean']):.3f}"
        dist_v = "" if r["euclidean_mean"] is None else f"{float(r['euclidean_mean']):.3f}"
        lines.append(f"{latex_escape(r['objective'])} & {cos_v} & {dist_v} \\\\\n")
    lines.append("\\bottomrule\n\\end{tabular}\n")
    (out / "cross_model_geometry_similarity_table.tex").write_text("".join(lines), encoding="utf-8")


def write_report(out: Path, rows: List[Dict[str, str]], best_agreement: List[Dict[str, Any]], sim_summary: List[Dict[str, Any]], variance: Dict[str, Any], perm: Dict[str, Any], sign_rows: List[Dict[str, Any]]) -> None:
    models = sorted({r["model_slug"] for r in rows})
    objectives = sorted({r["objective"] for r in rows})
    seeds = sorted({str(r["seed"]) for r in rows})
    lines = [
        "# Cross-Model Geometry Report\n\n",
        f"- Models: {len(models)} ({', '.join(models)})\n",
        f"- Objectives: {len(objectives)} ({', '.join(objectives)})\n",
        f"- Seeds: {len(seeds)} ({', '.join(seeds)})\n\n",
        "## Best Region Agreement\n\n",
    ]
    for r in best_agreement:
        lines.append(f"- {r['objective']} / {r['metric']}: {r['most_common_best_region']} ({r['agreement_count']}/{r['n_models']})\n")
    lines.append("\n## Geometry Similarity\n\n")
    for r in sim_summary:
        lines.append(f"- {r['objective']}: mean cosine={r['cosine_mean']}, mean Euclidean={r['euclidean_mean']}\n")
    lines.append("\n## Variance Decomposition\n\n")
    lines.append(f"- Objective explained ratio: {variance.get('objective_explained_ratio')}\n")
    lines.append(f"- Model explained ratio: {variance.get('model_explained_ratio')}\n")
    lines.append(f"- Larger descriptive component: {variance.get('key_question_answer')}\n\n")
    lines.append("## Permutation Tests\n\n")
    lines.append(f"- Objective-label permutation p: {perm['objective_label_permutation'].get('p_value')}\n")
    lines.append(f"- Model-label permutation p: {perm['model_label_permutation'].get('p_value')}\n\n")
    lines.append("## Primary Mislocation Sign Agreement\n\n")
    for r in sign_rows:
        lines.append(f"- {r['objective']} / {r['primary_metric']}: {r['n_positive']}/{r['n_models']} positive for {r['preferred_condition']} > {r['mislocated_condition']}\n")
    lines.append("\n## Interpretation Notes\n\n")
    larger = variance.get("key_question_answer")
    if larger == "objective":
        lines.append("High best-region agreement and high geometry cosine similarity suggest stable adaptation-geometry profiles across model families. In this descriptive analysis, the objective variance component is larger than the model component, supporting the claim that task type explains more geometry variation than architecture family.\n")
    elif larger == "model":
        lines.append("Best-region agreement and geometry similarity summarize which localization signatures replicate across model families. In this descriptive analysis, the model variance component is larger than the objective component, so the current partial run does not by itself support the claim that task type explains more geometry variation than architecture family.\n")
    else:
        lines.append("Best-region agreement and geometry similarity summarize which localization signatures replicate across model families. The variance-decomposition result is unavailable or inconclusive for this input.\n")
    (out / "cross_model_geometry_report.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts/localization/cross_model/summary_cross_model_by_seed.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/release_verification/localization/cross_model/geometry_analysis"))
    parser.add_argument("--n_perm", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.input)
    if not rows:
        print(f"[warn] No rows found in {args.input}")

    profiles = compute_profiles(rows)
    write_csv(args.output_dir / "cross_model_geometry_profiles_by_seed.csv", profiles, ["model_name", "model_slug", "objective", "seed", *PROFILE_DIMS])

    best_rows, best_agreement = best_regions(rows)
    write_csv(args.output_dir / "cross_model_best_regions.csv", best_rows, ["model_slug", "objective", "metric", "best_localized_condition", "best_localized_value", "full_value", "localized_advantage"])
    write_csv(args.output_dir / "cross_model_best_region_agreement.csv", best_agreement, ["objective", "metric", "most_common_best_region", "agreement_count", "n_models", "agreement_rate", "all_model_regions"])

    sim_rows, sim_summary = geometry_similarity(profiles)
    write_csv(args.output_dir / "cross_model_geometry_similarity.csv", sim_rows, ["objective", "model_i", "model_j", "cosine_similarity", "euclidean_distance"])
    write_csv(args.output_dir / "cross_model_geometry_similarity_summary.csv", sim_summary, ["objective", "cosine_mean", "cosine_sd", "cosine_min", "cosine_max", "euclidean_mean", "euclidean_sd", "euclidean_min", "euclidean_max"])

    variance = variance_decomposition(profiles)
    (args.output_dir / "cross_model_variance_decomposition.json").write_text(json.dumps(variance, indent=2), encoding="utf-8")

    perm = {
        "objective_label_permutation": permutation_test(profiles, "objective", args.n_perm, rng),
        "model_label_permutation": permutation_test(profiles, "model_slug", args.n_perm, rng),
    }
    (args.output_dir / "cross_model_permutation_tests.json").write_text(json.dumps(perm, indent=2), encoding="utf-8")

    penalty_by_seed, penalty_summary, sign_rows = mislocation(rows)
    write_csv(args.output_dir / "cross_model_mislocation_penalties_by_seed.csv", penalty_by_seed, ["model_name", "model_slug", "objective", "contrast_id", "metric", "preferred_condition", "mislocated_condition", "seed", "preferred_value", "mislocated_value", "delta_preferred_minus_mislocated"])
    write_csv(args.output_dir / "cross_model_mislocation_penalty_summary.csv", penalty_summary, ["model_slug", "objective", "contrast_id", "metric", "preferred_condition", "mislocated_condition", "delta_mean", "delta_sd", "n_seeds"])
    write_csv(args.output_dir / "cross_model_mislocation_sign_agreement.csv", sign_rows, ["objective", "primary_metric", "preferred_condition", "mislocated_condition", "n_models", "n_positive", "agreement_rate", "model_level_deltas"])

    write_latex_tables(args.output_dir, best_agreement, sign_rows, sim_summary)
    write_report(args.output_dir, rows, best_agreement, sim_summary, variance, perm, sign_rows)
    print(f"Wrote cross-model geometry analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
