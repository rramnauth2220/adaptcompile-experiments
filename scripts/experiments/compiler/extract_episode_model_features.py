#!/usr/bin/env python3
"""Extract one pre-adaptation feature payload per compiler episode."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.compiler_common import DEFAULT_TARGET_MODULES, read_jsonl  # noqa: E402
from src.extract_episode_features import (  # noqa: E402
    FrozenModelFeatureBackend,
    extract_episode_features,
    feature_path,
    select_episodes,
)


def write_feature_payloads(path: str | Path, payloads: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(payloads)
    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    table_rows = [
        {
            "episode_id": row.get("metadata", {}).get("episode_id"),
            "model_slug": row.get("metadata", {}).get("model_slug"),
            "meta_split": row.get("metadata", {}).get("meta_split"),
            "payload_json": json.dumps(row, ensure_ascii=False),
        }
        for row in rows
    ]
    if path.suffix == ".csv":
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode_id", "model_slug", "meta_split", "payload_json"])
            writer.writeheader()
            writer.writerows(table_rows)
        return
    if path.suffix == ".parquet":
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("Writing parquet requires pandas plus pyarrow or fastparquet.") from exc
        pd.DataFrame(table_rows).to_parquet(path, index=False)
        return
    raise ValueError("Feature output must end in .jsonl, .csv, or .parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Experiment 2 episode-model features.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--episode_manifest", default="data/compiler/episode_manifest.jsonl")
    parser.add_argument("--examples_path", default="data/prompt_examples.jsonl")
    parser.add_argument("--output", default="outputs/compiler/features/episode_features.jsonl")
    parser.add_argument("--output_root", default=None, help="Optional resumable per-episode JSON output root.")
    parser.add_argument("--episode_ids", nargs="+", default=None)
    parser.add_argument("--meta_split", choices=["train", "validation", "test"], default=None)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--target_modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--backend", choices=["frozen_model", "proxy"], default="frozen_model")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--torch_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--max_probe_examples", type=int, default=4, help="Use -1 for all adaptation examples.")
    parser.add_argument("--probe_module_batch_size", type=int, default=1)
    parser.add_argument("--sketch_dim", type=int, default=64)
    parser.add_argument("--sketch_elements", type=int, default=4096)
    parser.add_argument("--episode_embedding_dim", type=int, default=128)
    parser.add_argument("--episode_embedding_layer", type=int, default=-1)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = select_episodes(
        read_jsonl(args.episode_manifest),
        episode_ids=args.episode_ids,
        meta_split=args.meta_split,
        max_episodes=args.max_episodes,
    )
    examples_by_id = {row["example_id"]: row for row in read_jsonl(args.examples_path)}
    max_probe_examples: Optional[int] = None if args.max_probe_examples < 0 else args.max_probe_examples
    output_root = Path(args.output_root) if args.output_root else None
    if output_root is not None and args.skip_existing:
        episodes = [
            episode
            for episode in episodes
            if not feature_path(output_root, args.model_name_or_path, episode["episode_id"]).exists()
        ]

    payloads: List[Dict[str, Any]] = []
    if args.backend == "frozen_model" and episodes:
        print(
            "[load] Initializing frozen model backend once for "
            f"{len(episodes)} episode(s): {args.model_name_or_path}"
        )
        backend = FrozenModelFeatureBackend(
            model_name_or_path=args.model_name_or_path,
            target_modules=args.target_modules,
            n_layers=args.n_layers,
            seed=args.seed,
            max_length=args.max_length,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            max_probe_examples=max_probe_examples,
            probe_module_batch_size=args.probe_module_batch_size,
            sketch_dim=args.sketch_dim,
            sketch_elements=args.sketch_elements,
            episode_embedding_dim=args.episode_embedding_dim,
            episode_embedding_layer=args.episode_embedding_layer,
            benchmark=args.benchmark,
            trust_remote_code=args.trust_remote_code,
        )
        for episode in episodes:
            payload = backend.extract(episode, examples_by_id=examples_by_id)
            if output_root is not None:
                path = feature_path(output_root, args.model_name_or_path, episode["episode_id"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                print(f"Wrote {path}")
            else:
                payloads.append(payload)
    else:
        for episode in episodes:
            payload = extract_episode_features(
                episode,
                examples_by_id=examples_by_id,
                model_name_or_path=args.model_name_or_path,
                target_modules=args.target_modules,
                n_layers=args.n_layers,
                seed=args.seed,
                backend=args.backend,
                max_length=args.max_length,
                torch_dtype=args.torch_dtype,
                device_map=args.device_map,
                max_probe_examples=max_probe_examples,
                probe_module_batch_size=args.probe_module_batch_size,
                sketch_dim=args.sketch_dim,
                sketch_elements=args.sketch_elements,
                episode_embedding_dim=args.episode_embedding_dim,
                episode_embedding_layer=args.episode_embedding_layer,
                benchmark=args.benchmark,
                trust_remote_code=args.trust_remote_code,
            )
            if output_root is not None:
                path = feature_path(output_root, args.model_name_or_path, episode["episode_id"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                print(f"Wrote {path}")
            else:
                payloads.append(payload)

    if output_root is None:
        write_feature_payloads(args.output, payloads)
        print(f"Wrote {len(payloads)} episode feature payloads to {args.output}")
    else:
        print(f"Wrote {len(episodes)} per-episode feature payloads under {output_root}")


if __name__ == "__main__":
    main()
