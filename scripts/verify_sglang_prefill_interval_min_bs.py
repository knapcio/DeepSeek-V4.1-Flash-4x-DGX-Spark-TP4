#!/usr/bin/env python3
"""Fail closed unless the pinned SGLang tree carries the #44001 backport."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


REQUIRED = {
    "sglang/srt/arg_groups/fields/schedule.py": (
        "prefill_decode_interval_min_bs: A[",
        "] = 0",
    ),
    "sglang/srt/arg_groups/validation_hook.py": (
        "cfg.prefill_decode_interval_min_bs < 0",
        "--prefill-decode-interval-min-bs must be non-negative.",
    ),
    "sglang/srt/managers/scheduler.py": (
        "self.prefill_decode_interval_min_bs = get_schedule().prefill_decode_interval_min_bs",
        "def _should_defer_prefill(self, running_batch: ScheduleBatch) -> bool:",
        "running_batch.batch_size() < self.prefill_decode_interval_min_bs",
        "self._should_defer_prefill(running_batch)",
    ),
}


def _function_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if not matches:
        raise ValueError(f"missing function {name}")
    result = ast.get_source_segment(source, matches[-1])
    if result is None:
        raise ValueError(f"cannot extract function {name}")
    return result


def verify(root: Path) -> None:
    failures: list[str] = []
    sources: dict[str, str] = {}
    for relative, needles in REQUIRED.items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
            continue
        source = path.read_text()
        sources[relative] = source
        try:
            ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            failures.append(f"syntax error in {relative}: {exc}")
        for needle in needles:
            if needle not in source:
                failures.append(f"{relative}: missing {needle!r}")

    scheduler_rel = "sglang/srt/managers/scheduler.py"
    scheduler = sources.get(scheduler_rel)
    if scheduler:
        try:
            decision = _function_source(scheduler, "_should_defer_prefill")
        except (SyntaxError, ValueError) as exc:
            failures.append(f"{scheduler_rel}: cannot inspect scheduler decision: {exc}")
        else:
            remaining = decision.find("self._prefill_decode_interval_remaining == 0")
            threshold = decision.find(
                "running_batch.batch_size() < self.prefill_decode_interval_min_bs"
            )
            decrement = decision.find("self._prefill_decode_interval_remaining -= 1")
            if min(remaining, threshold, decrement) < 0 or not remaining < threshold < decrement:
                failures.append(
                    f"{scheduler_rel}: interval checks must be remaining, batch threshold, decrement"
                )
        if "self._should_defer_prefill()" in scheduler:
            failures.append(f"{scheduler_rel}: zero-argument scheduler call remains")

    if failures:
        raise SystemExit("SGLang #44001 backport verification failed:\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} PATH_TO_SGLANG_PYTHON")
    verify(Path(sys.argv[1]).resolve())
    print("SGLang #44001 prefill-interval batch-size gate: verified")
