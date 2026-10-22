#!/usr/bin/env python3
"""Fail-closed source verifier for SGLang #45450 commit 20189751."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def verify(python_root: Path) -> None:
    target = (
        python_root
        / "sglang/srt/speculative/dspark_components/dspark_draft.py"
    )
    source = target.read_text()
    ast.parse(source, filename=str(target))
    required = (
        "def _stage_sampling_params(",
        "all(staged[i] is key[i] for i in (0, 2, 3))",
        "self.num_staged_params_skips += skip",
        "self._staged_params_key = key",
        "self._stage_sampling_params(\n                draft_sampler",
    )
    missing = [needle for needle in required if needle not in source]
    if missing:
        raise SystemExit(f"incomplete sampling-param restage port: {missing}")
    direct = "draft_sampler.stage_sampling_params(bs=bs, sampling_info=sampling_info)"
    if source.count(direct) != 1:
        raise SystemExit("graph forward still stages directly or helper lost its call")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    verify(Path(sys.argv[1]))
    print("SGLang #45450 sampling-param restage source: verified")


if __name__ == "__main__":
    main()
