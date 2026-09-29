#!/usr/bin/env python3
"""Appendix Experiment 5: ranking fidelity under LOFO generalization."""

from __future__ import annotations

import argparse
import json
import math
import os
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from plot_style import (
    OBJECTIVE_LABELS,
    OBJECTIVE_ORDER,
    OUTCOMES,
    PROGRAM_ORDER,
    SELECTOR_COLORS,
    ensure_dirs,
    json_script_metadata,
    read_csv,
    read_json,
    to_builtin,
    write_summary,
)


EXPECTED_REPRESENTED = {
    "behavioral_policy": {"spearman": 0.900000, "pairwise": 0.916667},
    "causal_mapping": {"spearman": 0.940000, "pairwise": 0.958333},
    "factual_association": {"spearman": 0.603246, "pairwise": 0.758333},
    "lexical_binding": {"spearman": 0.630000, "pairwise": 0.783333},
    "procedural_reasoning": {"spearman": 0.940000, "pairwise": 0.958333},
}
EXPECTED_LOFO = {
    "behavioral_policy": {"spearman": 0.520000, "pairwise": 0.708333},
    "causal_mapping": {"spearman": 0.470000, "pairwise": 0.691667},
    "factual_association": {"spearman": -0.100000, "pairwise": 0.483333},
    "lexical_binding": {"spearman": -0.550000, "pairwise": 0.258333},
    "procedural_reasoning": {"spearman": -0.050000, "pairwise": 0.475000},
}
EXPECTED_LOFO_MACRO = {"spearman": 0.058, "pairwise": 0.523}
PRIMARY_METHOD = "primary"
TIE_TOLERANCE = 1e-12


def assert_close(name: str, observed: float, expected: float, tolerance: float) -> None:
    if not math.isfinite(float(observed)) or abs(float(observed) - float(expected)) > tolerance:
        raise ValueError(f"{name} mismatch: observed={observed}, expected approximately={expected}")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes"}


def sign_with_tolerance(value: float, tolerance: float = TIE_TOLERANCE) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def validate_balanced_utility(predictions: pd.DataFrame, label: str) -> Dict[str, int]:
    required = ["episode_id", "learning_type", "meta_split", "config_id", "observed_utility", "primary_predicted_utility"]
    for outcome in OUTCOMES:
        required.extend([f"observed_{outcome}", f"primary_predicted_{outcome}"])
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise ValueError(f"{label} prediction table missing required columns: {missing}")

    observed_expected = predictions[[f"observed_{outcome}" for outcome in OUTCOMES]].astype(float).mean(axis=1)
    predicted_expected = predictions[[f"primary_predicted_{outcome}" for outcome in OUTCOMES]].astype(float).mean(axis=1)
    observed_diff = (predictions["observed_utility"].astype(float) - observed_expected).abs().max()
    predicted_diff = (predictions["primary_predicted_utility"].astype(float) - predicted_expected).abs().max()
    if observed_diff > 1e-12:
        raise ValueError(f"{label} observed utility is not mean(A,T,B,P); max diff={observed_diff}")
    if predicted_diff > 1e-12:
        raise ValueError(f"{label} primary predicted utility is not mean(A,T,B,P); max diff={predicted_diff}")

    observed_ties = 0
    predicted_ties = 0
    total_pairs = 0
    for _, group in predictions.groupby("episode_id", sort=True):
        if set(group["config_id"].astype(str)) != set(PROGRAM_ORDER):
            raise ValueError(f"{label} episode lacks four candidate programs.")
        indexed = group.set_index("config_id")
        for a, b in combinations(PROGRAM_ORDER, 2):
            total_pairs += 1
            observed_ties += int(
                sign_with_tolerance(float(indexed.loc[a, "observed_utility"]) - float(indexed.loc[b, "observed_utility"])) == 0
            )
            predicted_ties += int(
                sign_with_tolerance(
                    float(indexed.loc[a, "primary_predicted_utility"]) - float(indexed.loc[b, "primary_predicted_utility"])
                )
                == 0
            )
    return {"total_pairs": total_pairs, "observed_ties": observed_ties, "predicted_ties": predicted_ties}


def aggregate_primary_ranking(ranking: pd.DataFrame, expected_objectives: Sequence[str], context: str) -> Dict[str, Dict[str, float]]:
    required = {"episode_id", "learning_type", "method", "spearman", "pairwise_ranking_accuracy"}
    missing = required - set(ranking.columns)
    if missing:
        raise ValueError(f"{context} ranking table missing columns: {sorted(missing)}")
    primary = ranking[ranking["method"].astype(str) == PRIMARY_METHOD].copy()
    if primary.empty:
        raise ValueError(f"{context} ranking table has no primary rows.")

    observed_objectives = set(primary["learning_type"].astype(str))
    expected_set = set(expected_objectives)
    if observed_objectives != expected_set:
        raise ValueError(f"{context} objectives mismatch: observed={sorted(observed_objectives)}, expected={sorted(expected_set)}")
    duplicates = int(primary.duplicated(["episode_id", "method"]).sum())
    if duplicates:
        raise ValueError(f"{context} has duplicate primary episode rows: {duplicates}")

    out: Dict[str, Dict[str, float]] = {}
    for objective, group in primary.groupby("learning_type", sort=False):
        objective = str(objective)
        out[objective] = {
            "spearman": float(group["spearman"].astype(float).mean()),
            "pairwise": float(group["pairwise_ranking_accuracy"].astype(float).mean()),
            "n_episodes": int(group["episode_id"].nunique()),
        }
    return out


def validate_represented(represented_root: Path, sanity_tolerance: float) -> Tuple[Dict[str, Dict[str, float]], List[Path], Dict[str, Any]]:
    ranking_path = represented_root / "ranking_metrics_by_episode.csv"
    predictions_path = represented_root / "predictions_test.csv"
    metadata_path = represented_root / "model_metadata.json"
    ranking_json_path = represented_root / "ranking_metrics.json"
    for path in [ranking_path, predictions_path, metadata_path, ranking_json_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    metadata = read_json(metadata_path)
    if metadata.get("experiment") != "experiment2_geometry_prediction":
        raise ValueError(f"{metadata_path} is not the ordinary Experiment 2 predictor metadata.")
    if metadata.get("feature_set") != "full":
        raise ValueError(f"{metadata_path} is not the final ordinary represented-family full representation.")
    if metadata.get("objective_identity_used_in_primary_predictor") is not False:
        raise ValueError(f"{metadata_path} indicates objective identity was used.")
    if list(metadata.get("config_ids", [])) != list(PROGRAM_ORDER):
        raise ValueError(f"{metadata_path} does not use the four primary candidate programs.")
    if int(metadata.get("test_episodes", -1)) != 100 or int(metadata.get("test_rows", -1)) != 400:
        raise ValueError(f"{metadata_path} has unexpected ordinary test coverage.")

    predictions = read_csv(predictions_path)
    if set(predictions["meta_split"].astype(str)) != {"test"}:
        raise ValueError("Represented-family predictions are not ordinary held-out test rows only.")
    if predictions["episode_id"].nunique() != 100 or len(predictions) != 400:
        raise ValueError("Represented-family predictions do not contain 100 episodes x 4 programs.")
    counts = predictions.groupby("learning_type")["episode_id"].nunique().to_dict()
    bad = {objective: counts.get(objective, 0) for objective in OBJECTIVE_ORDER if counts.get(objective, 0) != 20}
    if bad:
        raise ValueError(f"Represented-family objective counts are not 20 each: {bad}")
    tie_counts = validate_balanced_utility(predictions, "represented-family")

    metrics = aggregate_primary_ranking(read_csv(ranking_path), OBJECTIVE_ORDER, "represented-family")
    ranking_json = read_json(ranking_json_path)
    primary_json = ranking_json["primary"]
    assert_close(
        "represented macro Spearman",
        float(primary_json["mean_episode_spearman"]),
        float(np.mean([metrics[obj]["spearman"] for obj in OBJECTIVE_ORDER])),
        1e-12,
    )
    assert_close(
        "represented macro pairwise",
        float(primary_json["pairwise_ranking_accuracy"]),
        float(np.mean([metrics[obj]["pairwise"] for obj in OBJECTIVE_ORDER])),
        1e-12,
    )
    for objective, expected in EXPECTED_REPRESENTED.items():
        assert_close(f"represented {objective} Spearman", metrics[objective]["spearman"], expected["spearman"], sanity_tolerance)
        assert_close(f"represented {objective} pairwise", metrics[objective]["pairwise"], expected["pairwise"], sanity_tolerance)
        if metrics[objective]["n_episodes"] != 20:
            raise ValueError(f"Represented {objective} has n_episodes={metrics[objective]['n_episodes']}, expected 20.")

    checks = {
        "source": "ordinary held-out Experiment 2 final full_auto primary predictor",
        "feature_set": metadata.get("feature_set"),
        "test_episodes": int(metadata["test_episodes"]),
        "test_rows": int(metadata["test_rows"]),
        "balanced_utility": "observed_utility and primary_predicted_utility equal mean(A,T,B,P)",
        "pair_tie_counts": tie_counts,
    }
    return metrics, [ranking_path, predictions_path, metadata_path, ranking_json_path], checks


def validate_lofo(lofo_root: Path, sanity_tolerance: float) -> Tuple[Dict[str, Dict[str, float]], List[Path], Dict[str, Any]]:
    summary_path = lofo_root / "lofo_summary.csv"
    summary_json_path = lofo_root / "lofo_summary.json"
    for path in [summary_path, summary_json_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    summary = read_csv(summary_path)
    summary_json = read_json(summary_json_path)
    if summary_json.get("experiment") != "experiment2_lofo_generalization":
        raise ValueError(f"{summary_json_path} is not the final LOFO evaluator summary.")
    if summary_json.get("feature_set") != "episode":
        raise ValueError(f"{summary_json_path} is not the final episode-only LOFO run.")
    if list(summary_json.get("config_ids", [])) != list(PROGRAM_ORDER):
        raise ValueError(f"{summary_json_path} does not use the four primary candidate programs.")

    metrics: Dict[str, Dict[str, float]] = {}
    source_files = [summary_path, summary_json_path]
    fold_tie_counts: Dict[str, Any] = {}
    for objective in OBJECTIVE_ORDER:
        fold_root = lofo_root / objective
        ranking_path = fold_root / "ranking_metrics_by_episode.csv"
        predictions_path = fold_root / "predictions_test.csv"
        metadata_path = fold_root / "model_metadata.json"
        ranking_json_path = fold_root / "ranking_metrics.json"
        for path in [ranking_path, predictions_path, metadata_path, ranking_json_path]:
            if not path.exists():
                raise FileNotFoundError(path)
            source_files.append(path)

        metadata = read_json(metadata_path)
        if metadata.get("experiment") != "experiment2_lofo_generalization":
            raise ValueError(f"{metadata_path} is not LOFO metadata.")
        if metadata.get("held_out_learning_type") != objective:
            raise ValueError(f"{metadata_path} held_out_learning_type mismatch.")
        if metadata.get("feature_set") != "episode":
            raise ValueError(f"{metadata_path} does not use episode-only LOFO representation.")
        if metadata.get("held_out_objective_seen_during_training") is not False:
            raise ValueError(f"{metadata_path} indicates held-out objective was seen during training.")
        if metadata.get("held_out_objective_seen_during_validation") is not False:
            raise ValueError(f"{metadata_path} indicates held-out objective was seen during validation.")
        if metadata.get("test_rows_used_for_model_fitting_or_selection") is not False:
            raise ValueError(f"{metadata_path} indicates test rows were used for fitting or selection.")
        if objective in set(metadata.get("training_learning_types", [])) or objective in set(metadata.get("validation_learning_types", [])):
            raise ValueError(f"{objective} appears in LOFO train/validation metadata.")
        if set(metadata.get("test_learning_types", [])) != {objective}:
            raise ValueError(f"{metadata_path} test_learning_types mismatch.")
        if list(metadata.get("config_ids", [])) != list(PROGRAM_ORDER):
            raise ValueError(f"{metadata_path} does not use the four primary candidate programs.")

        predictions = read_csv(predictions_path)
        if set(predictions["meta_split"].astype(str)) != {"test"}:
            raise ValueError(f"{objective} LOFO predictions are not held-out test rows only.")
        if predictions["episode_id"].nunique() != 20 or len(predictions) != 80:
            raise ValueError(f"{objective} LOFO predictions do not contain 20 episodes x 4 programs.")
        if set(predictions["learning_type"].astype(str)) != {objective}:
            raise ValueError(f"{objective} LOFO prediction learning_type mismatch.")
        fold_tie_counts[objective] = validate_balanced_utility(predictions, f"LOFO {objective}")

        fold_metrics = aggregate_primary_ranking(read_csv(ranking_path), [objective], f"LOFO {objective}")
        metrics[objective] = fold_metrics[objective]
        ranking_json = read_json(ranking_json_path)["primary"]
        assert_close(f"LOFO {objective} ranking-json Spearman", metrics[objective]["spearman"], float(ranking_json["mean_episode_spearman"]), 1e-12)
        assert_close(f"LOFO {objective} ranking-json pairwise", metrics[objective]["pairwise"], float(ranking_json["pairwise_ranking_accuracy"]), 1e-12)
        if metrics[objective]["n_episodes"] != 20:
            raise ValueError(f"LOFO {objective} has n_episodes={metrics[objective]['n_episodes']}, expected 20.")
        expected = EXPECTED_LOFO[objective]
        assert_close(f"LOFO {objective} Spearman", metrics[objective]["spearman"], expected["spearman"], sanity_tolerance)
        assert_close(f"LOFO {objective} pairwise", metrics[objective]["pairwise"], expected["pairwise"], sanity_tolerance)

    summary_lookup = summary.set_index("held_out_learning_type").to_dict(orient="index")
    macro = summary_lookup.get("macro_average")
    if macro is None:
        raise ValueError("LOFO summary missing macro_average row.")
    macro_spearman = float(macro["mean_episode_spearman"])
    macro_pairwise = float(macro["pairwise_ranking_accuracy"])
    computed_spearman = float(np.mean([metrics[obj]["spearman"] for obj in OBJECTIVE_ORDER]))
    computed_pairwise = float(np.mean([metrics[obj]["pairwise"] for obj in OBJECTIVE_ORDER]))
    assert_close("LOFO macro Spearman from folds", macro_spearman, computed_spearman, 1e-12)
    assert_close("LOFO macro pairwise from folds", macro_pairwise, computed_pairwise, 1e-12)
    assert_close("LOFO macro Spearman sanity", macro_spearman, EXPECTED_LOFO_MACRO["spearman"], sanity_tolerance)
    assert_close("LOFO macro pairwise sanity", macro_pairwise, EXPECTED_LOFO_MACRO["pairwise"], sanity_tolerance)

    checks = {
        "source": "final Experiment 5 episode-only LOFO folds",
        "feature_set": "episode",
        "held_out_family_absent_from_training_and_validation": True,
        "macro_spearman": macro_spearman,
        "macro_pairwise": macro_pairwise,
        "balanced_utility": "observed_utility and primary_predicted_utility equal mean(A,T,B,P)",
        "pair_tie_counts_by_fold": fold_tie_counts,
    }
    return metrics, source_files, checks


def build_rows(represented: Mapping[str, Mapping[str, float]], lofo: Mapping[str, Mapping[str, float]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for objective in OBJECTIVE_ORDER:
        represented_spearman = float(represented[objective]["spearman"])
        lofo_spearman = float(lofo[objective]["spearman"])
        represented_pairwise = float(represented[objective]["pairwise"])
        lofo_pairwise = float(lofo[objective]["pairwise"])
        rows.append(
            {
                "objective": OBJECTIVE_LABELS[objective],
                "represented_spearman": represented_spearman,
                "lofo_spearman": lofo_spearman,
                "spearman_delta": lofo_spearman - represented_spearman,
                "represented_pairwise": represented_pairwise,
                "lofo_pairwise": lofo_pairwise,
                "pairwise_delta": lofo_pairwise - represented_pairwise,
            }
        )
    return rows


def draw_figure(
    rows: Sequence[Mapping[str, Any]],
    output_root: Path,
    inputs: Sequence[Path],
    checks: Mapping[str, Any],
    pairwise_chance_valid: bool,
) -> Dict[str, Any]:
    dirs = ensure_dirs(output_root)
    summary_path = dirs["data"] / "exp5_ranking_generalization.csv"
    write_summary(summary_path, rows)
    stem = dirs["appendix"] / "fig_exp5_ranking_generalization"
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    metadata_path = stem.with_suffix(".metadata.json")
    metadata = json_script_metadata(__file__, inputs)
    metadata.update(
        {
            "figure": "fig_exp5_ranking_generalization",
            "summary_csv": str(summary_path),
            "summary": rows,
            "bootstrap_resamples": None,
            "intervals": "none; plotted values are fold/objective means read from final evaluator outputs",
            "data_integrity_checks": checks,
            "pairwise_chance_reference": {
                "shown": bool(pairwise_chance_valid),
                "value": 0.5 if pairwise_chance_valid else None,
                "rationale": (
                    "Pairwise accuracy is sign agreement over unordered program pairs. A random strict ordering "
                    "matches each non-tied observed pair with probability 0.5; observed ties are rare in these "
                    "saved tables and are recorded in pair_tie_counts."
                ),
            },
            "caption": (
                "Program-ranking fidelity under leave-one-family-out generalization. Within-episode agreement "
                "between predicted and observed adaptation-program orderings is compared when the learning family "
                "is represented during meta-training versus excluded entirely under leave-one-family-out (LOFO) "
                "evaluation. (A) Spearman rank correlation. (B) Pairwise ranking accuracy. Ranking fidelity "
                "decreases across all five objectives when the learning family is unseen, with particularly large "
                "degradation for factual, lexical, and procedural episodes."
            ),
        }
    )
    _draw_static_outputs(rows, pdf_path, png_path, pairwise_chance_valid)
    metadata_path.write_text(json.dumps(to_builtin(metadata), indent=2), encoding="utf-8")
    return metadata


def _draw_static_outputs(rows: Sequence[Mapping[str, Any]], pdf_path: Path, png_path: Path, pairwise_chance_valid: bool) -> None:
    """Draw a small vector/raster dumbbell figure without requiring matplotlib."""

    from PIL import Image, ImageDraw, ImageFont
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    width_pt = 6.75 * 72.0
    height_pt = 3.15 * 72.0
    dpi = 300
    scale = dpi / 72.0
    windows_root = os.environ.get("WINDIR")
    font_regular_path = Path(windows_root) / "Fonts/arial.ttf" if windows_root else Path("__missing_font__")
    font_bold_path = Path(windows_root) / "Fonts/arialbd.ttf" if windows_root else Path("__missing_font__")
    if font_regular_path.exists():
        pdfmetrics.registerFont(TTFont("FigureSans", str(font_regular_path)))
        regular_pdf = "FigureSans"
    else:
        regular_pdf = "Helvetica"
    if font_bold_path.exists():
        pdfmetrics.registerFont(TTFont("FigureSansBold", str(font_bold_path)))
        bold_pdf = "FigureSansBold"
    else:
        bold_pdf = "Helvetica-Bold"

    regular_png = str(font_regular_path) if font_regular_path.exists() else None
    bold_png = str(font_bold_path) if font_bold_path.exists() else regular_png

    colors = {
        "text": "#222222",
        "muted": "#555555",
        "grid": "#D9D9D9",
        "axis": "#222222",
        "reference": "#777777",
        "connector": "#9B9B9B",
        "represented": SELECTOR_COLORS["objective_conditioned_mean"],
        "lofo": SELECTOR_COLORS["primary"],
        "white": "#FFFFFF",
    }
    layout = {
        "label_x": 96.0,
        "panel_a": (112.0, 260.0),
        "panel_b": (326.0, 474.0),
        "plot_top": 68.0,
        "plot_bottom": 174.0,
        "row_top": 78.0,
        "row_bottom": 158.0,
        "title_y": 42.0,
        "panel_label_y": 42.0,
        "tick_y": 185.0,
        "axis_label_y": 210.0,
        "legend_y": 18.0,
    }

    def hex_rgb(value: str) -> tuple[int, int, int]:
        value = value.lstrip("#")
        return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))

    def x_scale(value: float, domain: tuple[float, float], panel: tuple[float, float]) -> float:
        return panel[0] + (float(value) - domain[0]) / (domain[1] - domain[0]) * (panel[1] - panel[0])

    def y_pos(index: int) -> float:
        return layout["row_top"] + index * (layout["row_bottom"] - layout["row_top"]) / (len(rows) - 1)

    def render(line: Any, text: Any, circle: Any, diamond: Any) -> None:
        panel_a = layout["panel_a"]
        panel_b = layout["panel_b"]
        top = layout["plot_top"]
        bottom = layout["plot_bottom"]

        text(184, layout["legend_y"], "Represented family", 8.1, anchor="lm", color="text")
        circle(174, layout["legend_y"], 3.3, "represented", "axis", width=0.5)
        text(322, layout["legend_y"], "Held-out family (LOFO)", 8.1, anchor="lm", color="text")
        diamond(312, layout["legend_y"], 3.9, "white", "lofo", width=1.2)

        text(panel_a[0] - 28, layout["panel_label_y"], "A", 11.0, anchor="lm", bold=True, color="text")
        text(panel_a[0], layout["title_y"], "Rank correlation", 10.0, anchor="lm", color="text")
        text(panel_b[0] - 28, layout["panel_label_y"], "B", 11.0, anchor="lm", bold=True, color="text")
        text(panel_b[0], layout["title_y"], "Pairwise ranking", 10.0, anchor="lm", color="text")

        def axis(panel: tuple[float, float], domain: tuple[float, float], ticks: Sequence[float], tick_labels: Sequence[str], xlabel: str) -> None:
            for tick, label in zip(ticks, tick_labels):
                x = x_scale(tick, domain, panel)
                line(x, top, x, bottom, "grid", width=0.55)
                line(x, bottom, x, bottom + 3.2, "axis", width=0.75)
                text(x, layout["tick_y"], label, 7.4, anchor="ma", color="text")
            line(panel[0], bottom, panel[1], bottom, "axis", width=0.85)
            text((panel[0] + panel[1]) / 2, layout["axis_label_y"], xlabel, 9.1, anchor="ma", color="text")

        axis(panel_a, (-0.65, 1.0), [-0.5, 0.0, 0.5, 1.0], ["-0.5", "0", "0.5", "1.0"], "Spearman rho")
        axis(panel_b, (0.0, 1.0), [0.0, 0.5, 1.0], ["0", "0.5", "1.0"], "Pairwise accuracy")
        line(x_scale(0.0, (-0.65, 1.0), panel_a), top, x_scale(0.0, (-0.65, 1.0), panel_a), bottom, "reference", width=0.8)
        if pairwise_chance_valid:
            chance_x = x_scale(0.5, (0.0, 1.0), panel_b)
            line(chance_x, top, chance_x, bottom, "reference", width=0.75, dash=True)
            text(chance_x + 4.0, top - 7.0, "Chance", 7.1, anchor="la", color="muted")

        for i, row in enumerate(rows):
            y = y_pos(i)
            text(layout["label_x"], y, str(row["objective"]), 8.4, anchor="rm", color="text")
            rep_s = x_scale(float(row["represented_spearman"]), (-0.65, 1.0), panel_a)
            lofo_s = x_scale(float(row["lofo_spearman"]), (-0.65, 1.0), panel_a)
            rep_p = x_scale(float(row["represented_pairwise"]), (0.0, 1.0), panel_b)
            lofo_p = x_scale(float(row["lofo_pairwise"]), (0.0, 1.0), panel_b)
            line(rep_s, y, lofo_s, y, "connector", width=0.8)
            line(rep_p, y, lofo_p, y, "connector", width=0.8)
            circle(rep_s, y, 3.4, "represented", "axis", width=0.45)
            diamond(lofo_s, y, 3.9, "white", "lofo", width=1.2)
            circle(rep_p, y, 3.4, "represented", "axis", width=0.45)
            diamond(lofo_p, y, 3.9, "white", "lofo", width=1.2)

    def draw_pdf() -> None:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        c = canvas.Canvas(str(pdf_path), pagesize=(width_pt, height_pt))

        def line(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 0.8, dash: bool = False) -> None:
            c.setStrokeColor(colors[color] if color in colors else color)
            c.setLineWidth(width)
            c.setDash(2.5, 2.5) if dash else c.setDash()
            c.line(x1, height_pt - y1, x2, height_pt - y2)
            c.setDash()

        def text(x: float, y: float, value: str, size: float, anchor: str = "la", bold: bool = False, color: str = "text") -> None:
            font_name = bold_pdf if bold else regular_pdf
            c.setFont(font_name, size)
            c.setFillColor(colors[color] if color in colors else color)
            width = pdfmetrics.stringWidth(value, font_name, size)
            draw_x = x
            if anchor[0] == "m":
                draw_x = x - width / 2
            elif anchor[0] == "r":
                draw_x = x - width
            draw_y = height_pt - y
            if anchor[1] == "m":
                draw_y -= size * 0.35
            elif anchor[1] == "a":
                draw_y -= size
            c.drawString(draw_x, draw_y, value)

        def circle(x: float, y: float, radius: float, fill: str, outline: str, width: float = 0.8) -> None:
            c.setFillColor(colors[fill] if fill in colors else fill)
            c.setStrokeColor(colors[outline] if outline in colors else outline)
            c.setLineWidth(width)
            c.circle(x, height_pt - y, radius, stroke=1, fill=1)

        def diamond(x: float, y: float, radius: float, fill: str, outline: str, width: float = 1.2) -> None:
            c.setFillColor(colors[fill] if fill in colors else fill)
            c.setStrokeColor(colors[outline] if outline in colors else outline)
            c.setLineWidth(width)
            points = [
                (x, height_pt - (y - radius)),
                (x + radius, height_pt - y),
                (x, height_pt - (y + radius)),
                (x - radius, height_pt - y),
            ]
            path = c.beginPath()
            path.moveTo(*points[0])
            for px, py in points[1:]:
                path.lineTo(px, py)
            path.close()
            c.drawPath(path, stroke=1, fill=1)

        render(line, text, circle, diamond)
        c.showPage()
        c.save()

    def draw_pil() -> None:
        image = Image.new("RGB", (int(round(width_pt * scale)), int(round(height_pt * scale))), hex_rgb(colors["white"]))
        draw = ImageDraw.Draw(image)

        def xy(x: float, y: float) -> tuple[float, float]:
            return (x * scale, y * scale)

        def font(size: float, bold: bool = False) -> Any:
            path = bold_png if bold else regular_png
            return ImageFont.truetype(path, int(round(size * scale))) if path else ImageFont.load_default()

        def line(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 0.8, dash: bool = False) -> None:
            if not dash:
                draw.line([xy(x1, y1), xy(x2, y2)], fill=hex_rgb(colors[color]), width=max(1, int(round(width * scale))))
                return
            dash_len = 3.0
            gap = 3.0
            y = y1
            while y < y2:
                end = min(y + dash_len, y2)
                draw.line([xy(x1, y), xy(x2, end)], fill=hex_rgb(colors[color]), width=max(1, int(round(width * scale))))
                y = end + gap

        def text(x: float, y: float, value: str, size: float, anchor: str = "la", bold: bool = False, color: str = "text") -> None:
            draw.text(xy(x, y), value, font=font(size, bold=bold), fill=hex_rgb(colors[color]), anchor=anchor)

        def circle(x: float, y: float, radius: float, fill: str, outline: str, width: float = 0.8) -> None:
            r = radius * scale
            cx, cy = xy(x, y)
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=hex_rgb(colors[fill]), outline=hex_rgb(colors[outline]), width=max(1, int(round(width * scale))))

        def diamond(x: float, y: float, radius: float, fill: str, outline: str, width: float = 1.2) -> None:
            pts = [xy(x, y - radius), xy(x + radius, y), xy(x, y + radius), xy(x - radius, y)]
            draw.polygon(pts, fill=hex_rgb(colors[fill]))
            draw.line([*pts, pts[0]], fill=hex_rgb(colors[outline]), width=max(1, int(round(width * scale))))

        render(line, text, circle, diamond)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(png_path, dpi=(dpi, dpi))

    draw_pdf()
    draw_pil()


def collect_plot_data(represented_root: Path, lofo_root: Path, sanity_tolerance: float) -> Tuple[List[Dict[str, Any]], List[Path], Dict[str, Any], bool]:
    represented, represented_inputs, represented_checks = validate_represented(represented_root, sanity_tolerance)
    lofo, lofo_inputs, lofo_checks = validate_lofo(lofo_root, sanity_tolerance)
    rows = build_rows(represented, lofo)
    checks = {
        "represented": represented_checks,
        "lofo": lofo_checks,
        "same_objectives": [OBJECTIVE_LABELS[objective] for objective in OBJECTIVE_ORDER],
        "candidate_programs": list(PROGRAM_ORDER),
        "ranking_aggregation": "mean over primary per-episode ranking_metrics_by_episode rows within objective",
        "spearman_convention": "evaluator rank_episode Spearman over four candidate programs using tie-aware ranks",
        "pairwise_convention": "evaluator rank_episode pairwise sign agreement over six unordered candidate-program pairs",
    }
    tie_counts = [represented_checks["pair_tie_counts"], *lofo_checks["pair_tie_counts_by_fold"].values()]
    total_tied_observed = sum(int(item["observed_ties"]) for item in tie_counts)
    total_predicted_ties = sum(int(item["predicted_ties"]) for item in tie_counts)
    pairwise_chance_valid = total_predicted_ties == 0
    checks["pairwise_chance_assessment"] = {
        "chance_line_at_0_5_used": bool(pairwise_chance_valid),
        "observed_tied_pairs_total": int(total_tied_observed),
        "predicted_tied_pairs_total": int(total_predicted_ties),
    }
    return rows, [*represented_inputs, *lofo_inputs], checks, pairwise_chance_valid


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Appendix Experiment 5 ranking-generalization figure.")
    parser.add_argument("--represented_root", default="artifacts/llama/prediction")
    parser.add_argument("--lofo_root", default="artifacts/llama/lofo")
    parser.add_argument("--output_root", default="outputs/release_verification/figures/llama")
    parser.add_argument("--sanity_tolerance", type=float, default=5e-4)
    args = parser.parse_args()

    rows, inputs, checks, pairwise_chance_valid = collect_plot_data(
        Path(args.represented_root),
        Path(args.lofo_root),
        sanity_tolerance=args.sanity_tolerance,
    )
    draw_figure(rows, Path(args.output_root), inputs, checks, pairwise_chance_valid)
    print("Experiment 5 ranking generalization:")
    for row in rows:
        print(
            f"  {row['objective']}: "
            f"spearman {row['represented_spearman']:.6f}->{row['lofo_spearman']:.6f} "
            f"(delta {row['spearman_delta']:.6f}); "
            f"pairwise {row['represented_pairwise']:.6f}->{row['lofo_pairwise']:.6f} "
            f"(delta {row['pairwise_delta']:.6f})"
        )
    print(f"Wrote {Path(args.output_root) / 'appendix' / 'fig_exp5_ranking_generalization.pdf'}")
    print(f"Wrote {Path(args.output_root) / 'appendix' / 'fig_exp5_ranking_generalization.png'}")


if __name__ == "__main__":
    main()
