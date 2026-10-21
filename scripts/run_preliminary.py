#!/usr/bin/env python3
"""Run one teammate's assigned model on all three mineral groups."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from model.config import load_config, smoke_test_config
from model.metrics import priority_weighted_score
from model.trainer import run_group


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["resnet18", "efficientnet_b0", "efficientnet_b3"])
    parser.add_argument("--cache-dir", required=True, help="Drive folder containing high_224.zip, medium_224.zip, and low_224.zip")
    parser.add_argument("--cache-meta", default=ROOT / "notebooks/cache_224/cache_meta.json")
    parser.add_argument(
        "--splits-meta",
        default=ROOT / "notebooks/preliminary_splits/splits_meta.json",
    )
    parser.add_argument("--config", default=ROOT / "configs/preliminary_experiment.json")
    parser.add_argument("--extract-dir", default="/content/mineral_cache")
    parser.add_argument("--results-dir", default=ROOT / "results/preliminary")
    parser.add_argument("--checkpoint-dir", required=True, help="Drive folder for large .pt checkpoint files")
    parser.add_argument("--groups", nargs="+", choices=["high", "medium", "low"])
    parser.add_argument("--smoke-test", action="store_true", help="Run one epoch to verify the pipeline; not an official result")
    return parser.parse_args()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.smoke_test:
        config = smoke_test_config(config)
    groups = args.groups or config["groups"]

    # Each teammate runs the same function three times; only model_name differs
    # between teammates. Resetting the seed inside run_group starts each run from
    # the same initialization policy.
    summaries = {}
    for group in groups:
        summaries[group] = run_group(
            model_name=args.model,
            group=group,
            config=config,
            cache_dir=args.cache_dir,
            cache_meta_path=args.cache_meta,
            splits_meta_path=args.splits_meta,
            extract_root=args.extract_dir,
            results_root=args.results_dir,
            checkpoint_root=args.checkpoint_dir,
        )

    group_f1 = {group: summary["test_metrics"]["macro_f1"] for group, summary in summaries.items()}
    aggregate = {
        "model": args.model,
        "seed": config["seed"],
        "groups_completed": list(summaries),
        "group_macro_f1": group_f1,
    }
    if set(groups) == set(config["groups"]):
        aggregate["priority_weighted_score"] = priority_weighted_score(group_f1, config["priority_weights"])
        aggregate["cross_group_macro_f1_standard_deviation"] = statistics.pstdev(group_f1.values())

    mode = "smoke" if args.smoke_test else "preliminary"
    output = Path(args.results_dir) / args.model / f"aggregate_{mode}.json"
    write_json(output, aggregate)
    print(f"Wrote aggregate results to {output}")


if __name__ == "__main__":
    main()

