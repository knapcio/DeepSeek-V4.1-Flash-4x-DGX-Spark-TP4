#!/usr/bin/env python3
"""Backport DeepSeek V4.1 forced tool-call grammar from SGLang #39665.

The default dev-dsv41 image lacks the V4.1 structural tag. Required and named
tool choices therefore compile a legacy grammar that permits a bare invoke but
not the calls block consumed by the parser. Install the known upstream detector
and fail closed on any unexpected base-image source.
"""
import hashlib
import pathlib
import sys

DEFAULT_TARGET = (
    "/sgl-workspace/sglang/python/sglang/srt/function_call/"
    "deepseekv41_detector.py"
)
STOCK_SHA256 = "862168f54a45a336f99d6389840a07e1be55bb589a99a030ecf109c2b09027e4"
BACKPORT_SHA256 = "1db7eb88dfed0530a4c85231e525da83a5fd009e4feb85610bffd94d4c8c3672"
BACKPORT = (
    pathlib.Path(__file__).resolve().parent
    / "sglang"
    / "deepseekv41_detector.py"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    target = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TARGET)
    replacement = BACKPORT.read_bytes()
    if sha256(replacement) != BACKPORT_SHA256:
        print(
            f"[v41-tool-constraint] vendored backport hash mismatch: {BACKPORT}",
            file=sys.stderr,
        )
        return 1
    current = sha256(target.read_bytes())
    if current == BACKPORT_SHA256:
        print(f"[v41-tool-constraint] already applied: {target}")
        return 0
    if current != STOCK_SHA256:
        print(
            f"[v41-tool-constraint] {target} is neither the stock dev-dsv41 "
            f"file nor the backport (sha256 {current}); re-check against "
            "sglang#39665/#39026",
            file=sys.stderr,
        )
        return 1
    target.write_bytes(replacement)
    print(
        "[v41-tool-constraint] applied sglang#39665 V4.1 structural tag: "
        f"{target}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
