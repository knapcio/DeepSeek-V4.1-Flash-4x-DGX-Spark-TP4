#!/usr/bin/env python3
"""Fail closed unless the pinned SGLang tree carries the #41303 safety backport."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


REQUIRED = {
    "sglang/kernels/ops/attention/dsv4_attn_metadata_kernels.py": (
        "max(0, s - tail_len)",
        "(seq_lens - tail_len).clamp_min(0)",
    ),
    "sglang/srt/mem_cache/cache_init_params.py": (
        "decoder_swa_bounded_replay: bool = False",
    ),
    "sglang/srt/mem_cache/base_prefix_cache.py": (
        "def swa_reprefill_tail_tokens(self, req=None) -> int:",
    ),
    "sglang/srt/mem_cache/swa_radix_cache.py": (
        "def swa_reprefill_tail_tokens(self, req=None) -> int:",
    ),
    "sglang/srt/mem_cache/kv_cache_builder.py": (
        "decoder_swa_bounded_replay=get_exec().features.enable_decoder_swa_bounded_replay",
    ),
    "sglang/srt/mem_cache/unified_radix_cache.py": (
        "def swa_reprefill_tail_tokens(self, req=None) -> int:",
        "self.decoder_swa_bounded_replay = params.decoder_swa_bounded_replay",
        "self.decoder_swa_bounded_replay and not self.session.will_lend_record(req)",
    ),
    "sglang/srt/session/streaming_session.py": (
        "def will_lend_record(self, req: Optional[Req]) -> bool:",
    ),
    "sglang/srt/managers/schedule_policy.py": (
        "swa_reprefill_tail_tokens(req)",
    ),
    "sglang/srt/managers/schedule_batch.py": (
        "swa_reprefill_tail_tokens(self)",
    ),
}


def verify(root: Path) -> None:
    failures: list[str] = []
    for relative, needles in REQUIRED.items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
            continue
        source = path.read_text()
        try:
            ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            failures.append(f"syntax error in {relative}: {exc}")
        for needle in needles:
            if needle not in source:
                failures.append(f"{relative}: missing {needle!r}")

    for relative in (
        "sglang/srt/managers/schedule_policy.py",
        "sglang/srt/managers/schedule_batch.py",
    ):
        path = root / relative
        if path.is_file() and "swa_reprefill_tail_tokens()" in path.read_text():
            failures.append(f"{relative}: request-unaware SWA holdback call remains")

    schedule_batch = root / "sglang/srt/managers/schedule_batch.py"
    if schedule_batch.is_file():
        count = schedule_batch.read_text().count("swa_reprefill_tail_tokens(self)")
        if count != 2:
            failures.append(
                "sglang/srt/managers/schedule_batch.py: expected two request-aware "
                f"SWA holdback calls, found {count}"
            )

    if failures:
        raise SystemExit("SGLang #41303 backport verification failed:\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} PATH_TO_SGLANG_PYTHON")
    verify(Path(sys.argv[1]).resolve())
    print("SGLang #41303 bounded-replay safety backport: verified")
