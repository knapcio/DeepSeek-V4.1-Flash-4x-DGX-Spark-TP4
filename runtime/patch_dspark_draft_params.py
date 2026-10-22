#!/usr/bin/env python3
"""Adapt SGLang #45663's fused draft-parameter kernel to the pinned tree.

The pinned DSpark sampler consumes temperature and greedy-mask buffers but
predates upstream's step-temperature buffer.  Keep the exact upstream kernel
and provide its third output as a small static scratch buffer.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if new in source:
        return source
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, found {count}")
    return source.replace(old, new)


IMPORT_OLD = """from sglang.kernels.ops.speculative.dspark.dspark_draft_model import (
    SampleStepTokens,
)
"""
IMPORT_NEW = IMPORT_OLD + """from sglang.kernels.ops.speculative.dspark.draft_params import (
    fill_draft_params,
)
"""

BUFFERS_OLD = """        self.temperatures = None
        self.greedy_mask = None
        self.exp_noise = None
"""
BUFFERS_NEW = """        self.temperatures = None
        self.greedy_mask = None
        # SGLang #45663's exact kernel also emits per-step temperatures.  This
        # older sampler does not consume them, so retain a static scratch output.
        self._step_temperatures = None
        self.exp_noise = None
"""

ALLOC_OLD = """            self.greedy_mask = torch.ones((max_bs,), dtype=torch.bool, device=device)
            self.exp_noise = torch.empty(
"""
ALLOC_NEW = """            self.greedy_mask = torch.ones((max_bs,), dtype=torch.bool, device=device)
            self._step_temperatures = torch.zeros(
                (max_bs,), dtype=torch.float32, device=device
            )
            self.exp_noise = torch.empty(
"""

STAGE_OLD = """        torch.clamp(
            sampling_info.temperatures.view(-1)[:bs].to(torch.float32),
            min=1e-5,
            out=self.temperatures[:bs],
        )
        self.greedy_mask[:bs].copy_((sampling_info.top_ks <= 1).view(-1)[:bs])
"""
STAGE_NEW = """        fill_draft_params(
            sampling_info.temperatures.view(-1)[:bs],
            sampling_info.top_ks.view(-1)[:bs],
            self.temperatures[:bs],
            self.greedy_mask[:bs],
            self._step_temperatures[:bs],
        )
"""


def patch_tree(python_root: Path) -> None:
    target = (
        python_root
        / "sglang/srt/speculative/dspark_components/dspark_draft_sampler.py"
    )
    source = target.read_text()
    source = replace_once(source, IMPORT_OLD, IMPORT_NEW, "draft-params import")
    source = replace_once(source, BUFFERS_OLD, BUFFERS_NEW, "scratch declaration")
    source = replace_once(source, ALLOC_OLD, ALLOC_NEW, "scratch allocation")
    source = replace_once(source, STAGE_OLD, STAGE_NEW, "staging call")
    ast.parse(source, filename=str(target))
    target.write_text(source)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    patch_tree(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
