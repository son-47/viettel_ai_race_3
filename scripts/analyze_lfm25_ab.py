"""Cluster-aware comparison of grading-spec A/B runs.

The 420 requests are not independent: six turns belong to each of 70
conversations.  This script therefore reports deltas and bootstrap intervals
after aggregating by conversation instead of treating all requests as IID.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def latency_score(value_ms: float, floor_ms: float, ceiling_ms: float) -> float:
    normalized = min(
        1.0,
        max(0.0, (ceiling_ms - value_ms) / (ceiling_ms - floor_ms)),
    )
    return normalized**2


def request_score(item: dict) -> float:
    if item["error"] is not None:
        return 0.0
    return 0.5 * (
        latency_score(float(item["ttft_ms"]), 10.0, 400.0)
        + latency_score(float(item["tpot_ms"]), 1.0, 10.0)
    )


def case_name(payload: dict) -> str:
    stem = Path(payload["summary"]["config"]["out"]).stem
    parts = stem.split("-", 1)
    return parts[1] if len(parts) == 2 else stem


def result_scores(payload: dict) -> dict[tuple[int, int], float]:
    scores = {
        (int(item["conversation_id"]), int(item["turn_index"])): request_score(item)
        for item in payload["results"]
    }
    config = payload["summary"]["config"]
    for conversation_id in range(int(config["num_conversations"])):
        for turn_index in range(int(config["turns"])):
            # A failed turn stops the remaining turns in that conversation.
            # The grader and benchmark summary score those missing turns as 0.
            scores.setdefault((conversation_id, turn_index), 0.0)
    return scores


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    by_rate: dict[float, list[dict]] = defaultdict(list)
    for path in sorted(args.result_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        by_rate[float(payload["summary"]["config"]["rate_scale"])].append(payload)

    rng = random.Random(args.seed)
    print("rate\tcase\ters\tdelta_vs_control_mean\tcluster_bootstrap_95pct")
    for rate, payloads in sorted(by_rate.items()):
        controls = [p for p in payloads if case_name(p).startswith("cold_control-")]
        if not controls:
            continue
        control_maps = [result_scores(payload) for payload in controls]
        keys = sorted(set.intersection(*(set(scores) for scores in control_maps)))
        control_mean = {
            key: statistics.fmean(scores[key] for scores in control_maps)
            for key in keys
        }

        for payload in payloads:
            scores = result_scores(payload)
            per_conversation: dict[int, list[float]] = defaultdict(list)
            for key in keys:
                if key in scores:
                    per_conversation[key[0]].append(scores[key] - control_mean[key])
            cluster_deltas = [
                statistics.fmean(deltas)
                for _, deltas in sorted(per_conversation.items())
            ]
            delta = statistics.fmean(cluster_deltas)
            bootstrap = [
                statistics.fmean(rng.choices(cluster_deltas, k=len(cluster_deltas)))
                for _ in range(args.bootstrap)
            ]
            interval = (
                percentile(bootstrap, 0.025),
                percentile(bootstrap, 0.975),
            )
            print(
                f"{rate:g}\t{case_name(payload)}\t{payload['summary']['ers']:.9f}"
                f"\t{delta:+.9f}\t[{interval[0]:+.9f}, {interval[1]:+.9f}]"
            )


if __name__ == "__main__":
    main()
