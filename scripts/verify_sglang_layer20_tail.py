#!/usr/bin/env python3
"""Fail closed unless the pinned SGLang tree carries the #42901 backport."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


REQUIRED = {
    "sglang/srt/layers/attention/deepseek_v4_backend.py": (
        "self.tail_source_kv_layer_id: Optional[int] = None",
        "def write_tail_source_kv(",
        "run_indexer=False",
        "if layer.layer_id == self.tail_source_kv_layer_id:",
        "self.tail_source_kv_layer_id = None",
        "if self.tail_source_kv_layer_id is None:",
    ),
    "sglang/srt/models/deepseek_v4.py": (
        "self.tail_writes_source_kv = not _is_hip",
        "last_source if self.tail_writes_source_kv else last_source + 1",
        "last_source in config.index_source_layer_ids",
        "def write_tail_source_kv(",
        "self.layers[i].write_tail_source_kv(",
    ),
}


def _function(source: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    if not matches:
        raise ValueError(f"missing function {name}")
    return matches[-1]


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

    model_rel = "sglang/srt/models/deepseek_v4.py"
    model_source = sources.get(model_rel)
    if model_source:
        try:
            forward = ast.get_source_segment(
                model_source, _function(model_source, "_forward_layers_hc_pre_from_prev")
            )
        except (SyntaxError, ValueError) as exc:
            failures.append(f"{model_rel}: cannot inspect forward loop: {exc}")
        else:
            write_at = forward.find("self.layers[i].write_tail_source_kv(")
            enter_at = forward.find("attn_backend.enter_late_layer_tail(")
            if write_at < 0 or enter_at < 0 or write_at >= enter_at:
                failures.append(
                    f"{model_rel}: full-row compressed KV must be written before entering the tail"
                )

    backend_rel = "sglang/srt/layers/attention/deepseek_v4_backend.py"
    backend_source = sources.get(backend_rel)
    if backend_source:
        try:
            writer = ast.get_source_segment(
                backend_source, _function(backend_source, "write_tail_source_kv")
            )
        except (SyntaxError, ValueError) as exc:
            failures.append(f"{backend_rel}: cannot inspect writer: {exc}")
        else:
            call_at = writer.find("self.forward_low_ratio_sources(")
            arm_at = writer.find("self.tail_source_kv_layer_id = layer.layer_id")
            if call_at < 0 or arm_at < 0 or call_at >= arm_at:
                failures.append(
                    f"{backend_rel}: tail-source marker must arm only after the full-row write"
                )

    if failures:
        raise SystemExit("SGLang #42901 backport verification failed:\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} PATH_TO_SGLANG_PYTHON")
    verify(Path(sys.argv[1]).resolve())
    print("SGLang #42901 layer-20 bounded-replay backport: verified")
