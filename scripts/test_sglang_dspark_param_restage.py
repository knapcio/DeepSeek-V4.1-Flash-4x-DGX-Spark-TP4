#!/usr/bin/env python3
"""Execute the staged #45450 restage method without importing SGLang/Torch."""

from __future__ import annotations

import ast
import logging
import sys
from pathlib import Path
from types import SimpleNamespace


def load_method(python_root: Path):
    source_path = (
        python_root
        / "sglang/srt/speculative/dspark_components/dspark_draft.py"
    )
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DraftBlockProposer"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_stage_sampling_params"
    )
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {
        "STAGED_PARAMS_LOG_EVERY": 500,
        "logger": logging.getLogger(__name__),
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["_stage_sampling_params"]


def test(python_root: Path) -> None:
    stage_method = load_method(python_root)
    proposer = SimpleNamespace(
        _staged_params_key=None,
        num_staged_params_skips=0,
        num_staged_params_calls=0,
    )
    staged = []
    sampler = SimpleNamespace(
        stage_sampling_params=lambda **kw: staged.append(
            (kw["bs"], kw["sampling_info"])
        )
    )
    info = SimpleNamespace(temperatures=object(), top_ks=object())

    def stage(active_sampler, bs, sampling_info):
        stage_method(
            proposer,
            active_sampler,
            bs=bs,
            sampling_info=sampling_info,
        )

    stage(sampler, 4, info)
    stage(sampler, 4, info)
    info.temperatures = object()
    stage(sampler, 3, info)
    stage(sampler, 2, info)
    other = SimpleNamespace(stage_sampling_params=sampler.stage_sampling_params)
    stage(other, 2, info)
    stage(other, 2, None)
    stage(other, 2, None)
    assert [bs for bs, _ in staged] == [4, 3, 2, 2, 2]
    assert proposer.num_staged_params_skips == 2
    assert proposer.num_staged_params_calls == 7


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    test(Path(sys.argv[1]))
    print("SGLang #45450 sampling-param restage test: PASS")


if __name__ == "__main__":
    main()
