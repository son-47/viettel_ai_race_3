"""Summarize vLLM speculative-decoding counters from a Prometheus dump."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path


SAMPLE = re.compile(
    r"^(?P<name>vllm:spec_decode_[a-z_]+_total)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[0-9.eE+-]+)$"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path, nargs="+")
    args = parser.parse_args()

    print(
        "file\tdrafts\tdraft_tokens\taccepted_tokens\tacceptance_rate"
        "\tmean_accepted_per_round\tmean_tokens_per_verifier_round"
    )
    for path in args.metrics:
        totals: dict[str, float] = defaultdict(float)
        for line in path.read_text(encoding="utf-8").splitlines():
            match = SAMPLE.match(line)
            if match is None or "per_pos" in match.group("name"):
                continue
            totals[match.group("name")] += float(match.group("value"))

        if not totals:
            raise RuntimeError(
                f"{path}: no speculative counters; rerun the server with "
                "COLLECT_SPEC_METRICS=1 (without --disable-log-stats)"
            )

        drafts = totals["vllm:spec_decode_num_drafts_total"]
        draft_tokens = totals["vllm:spec_decode_num_draft_tokens_total"]
        accepted = totals["vllm:spec_decode_num_accepted_tokens_total"]
        acceptance_rate = accepted / draft_tokens if draft_tokens else 0.0
        accepted_per_round = accepted / drafts if drafts else 0.0
        tokens_per_round = 1.0 + accepted_per_round if drafts else 0.0
        print(
            f"{path.name}\t{drafts:g}\t{draft_tokens:g}\t{accepted:g}"
            f"\t{acceptance_rate:.6f}\t{accepted_per_round:.6f}"
            f"\t{tokens_per_round:.6f}"
        )


if __name__ == "__main__":
    main()
