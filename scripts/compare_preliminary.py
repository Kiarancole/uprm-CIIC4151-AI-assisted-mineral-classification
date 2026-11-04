#!/usr/bin/env python3
"""Combine the nine official experiment summaries into one team comparison."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


MODELS = ["resnet18", "efficientnet_b0", "efficientnet_b3"]
GROUPS = ["high", "medium", "low"]


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing official result: {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/preliminary")
    parser.add_argument("--output", default="results/preliminary/team_comparison.json")
    args = parser.parse_args()

    root = Path(args.results_dir)
    rows = []
    reference_config = None
    commits = set()
    hardware = set()
    for model in MODELS:
        aggregate = read_json(root / model / "aggregate_preliminary.json")
        seed_directory = f"seed_{aggregate['seed']}"
        summaries = [
            read_json(root / model / group / seed_directory / "summary.json") for group in GROUPS
        ]
        # Refuse to compare results if someone changed the shared protocol. The
        # accuracy numbers would otherwise look comparable when they are not.
        for summary in summaries:
            if summary["model"] != model or summary["group"] not in GROUPS:
                raise ValueError(f"Mislabeled result for {model}: {summary['group']}")
            if reference_config is None:
                reference_config = summary["shared_config"]
            elif summary["shared_config"] != reference_config:
                raise ValueError("Official runs do not share an identical experiment configuration")
            commits.add(summary["git_commit"])
            hardware.add(summary["hardware"]["gpu"] or "CPU")

        high_summary = next(summary for summary in summaries if summary["group"] == "high")
        rows.append(
            {
                "model": model,
                "priority_weighted_score": aggregate["priority_weighted_score"],
                "cross_group_macro_f1_standard_deviation": aggregate[
                    "cross_group_macro_f1_standard_deviation"
                ],
                "mean_accuracy": statistics.fmean(s["test_metrics"]["accuracy"] for s in summaries),
                "mean_top3_accuracy": statistics.fmean(
                    s["test_metrics"]["top3_accuracy"] for s in summaries
                ),
                "mean_expected_calibration_error": statistics.fmean(
                    s["test_metrics"]["expected_calibration_error"] for s in summaries
                ),
                "high_group_worst_class_recall": high_summary["test_metrics"]["worst_class_recall"],
                "total_parameters": summaries[0]["complexity"]["total_parameters"],
                "total_assimilation_time_seconds": sum(
                    s["assimilation_time_seconds"] for s in summaries
                ),
                "mean_cpu_inference_seconds": statistics.fmean(
                    s["cpu_inference_latency"]["mean_seconds"] for s in summaries
                ),
            }
        )

    # PWS is the first selection rule from the proposal. A close result still
    # requires the team to review recall, complexity, and latency manually.
    rows.sort(key=lambda row: row["priority_weighted_score"], reverse=True)
    selection = reference_config["selection"]
    pws_difference = rows[0]["priority_weighted_score"] - rows[1]["priority_weighted_score"]
    comparison = {
        "models": rows,
        "pws_leader": rows[0]["model"],
        "top_two_pws_difference": pws_difference,
        "tie_break_review_required": pws_difference < float(selection["pws_tie_margin"]),
        "same_git_commit": len(commits) == 1,
        "git_commits": sorted(commits),
        "same_training_hardware": len(hardware) == 1,
        "training_hardware": sorted(hardware),
        "selection_note": (
            f"If the top two PWS values differ by less than {selection['pws_tie_margin']}, "
            "review complexity and latency. A candidate with high-group worst-class recall "
            f"below {selection['high_group_minimum_worst_class_recall']} should not be selected "
            "unless no candidate reaches the threshold."
        ),
    }
    write_json(Path(args.output), comparison)
    print(json.dumps(comparison, indent=2))
    print(f"Wrote comparison to {args.output}")


if __name__ == "__main__":
    main()
