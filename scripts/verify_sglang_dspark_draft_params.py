#!/usr/bin/env python3
"""Fail-closed source verification for the SGLang #45663 adaptation."""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path


HELPER_SHA256 = "993dc3d686f4673b774278110fdbf44a8b1d190af1841c597f9fd7e50d12fbce"


def verify(python_root: Path) -> None:
    helper = (
        python_root
        / "sglang/kernels/ops/speculative/dspark/draft_params.py"
    )
    sampler = (
        python_root
        / "sglang/srt/speculative/dspark_components/dspark_draft_sampler.py"
    )
    if hashlib.sha256(helper.read_bytes()).hexdigest() != HELPER_SHA256:
        raise SystemExit("draft_params.py does not match SGLang #45663 @ 833e5189")

    helper_source = helper.read_text()
    sampler_source = sampler.read_text()
    ast.parse(helper_source, filename=str(helper))
    ast.parse(sampler_source, filename=str(sampler))

    required = (
        "from sglang.kernels.ops.speculative.dspark.draft_params import",
        "self._step_temperatures = torch.zeros(",
        "fill_draft_params(",
        "self._step_temperatures[:bs]",
    )
    missing = [needle for needle in required if needle not in sampler_source]
    if missing:
        raise SystemExit(f"incomplete fused draft-parameter integration: {missing}")

    stage_start = sampler_source.index("    def stage_sampling_params(")
    stage_end = sampler_source.index("    def __call__(", stage_start)
    stage = sampler_source[stage_start:stage_end]
    if "torch.clamp(" in stage or "greedy_mask[:bs].copy_(" in stage:
        raise SystemExit("legacy two-launch parameter staging is still active")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    verify(Path(sys.argv[1]))
    print("SGLang #45663 fused draft-parameter adaptation: verified")


if __name__ == "__main__":
    main()
