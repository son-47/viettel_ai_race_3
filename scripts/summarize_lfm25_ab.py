"""Print a compact TSV summary for grading-spec A/B result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()

    columns = (
        "case",
        "ers",
        "ttft_mean",
        "ttft_p50",
        "ttft_p95",
        "tpot_mean",
        "tpot_p50",
        "tpot_p95",
        "ttft_zero",
        "tpot_zero",
        "successful",
    )
    print("\t".join(columns))
    for path in sorted(args.result_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        case = Path(summary["config"]["out"]).stem
        values = (
            case,
            summary["ers"],
            summary["ttft_ms"]["mean"],
            summary["ttft_ms"]["p50"],
            summary["ttft_ms"]["p95"],
            summary["tpot_ms"]["mean"],
            summary["tpot_ms"]["p50"],
            summary["tpot_ms"]["p95"],
            summary["ttft_zero_count"],
            summary["tpot_zero_count"],
            summary["requests_successful"],
        )
        print("\t".join(str(value) for value in values))


if __name__ == "__main__":
    main()
