#!/usr/bin/env python3
"""CPU falsifier for #45794 alignment thresholds and logical score slicing."""

from __future__ import annotations


def score_width(lc: int, *, is_cuda: bool, is_bf16: bool) -> int:
    if is_cuda and is_bf16 and lc >= 1024:
        return (lc + 63) // 64 * 64
    return lc


def main() -> None:
    expected = {
        1: 1,
        1023: 1023,
        1024: 1024,
        1025: 1088,
        28799: 28800,
        28800: 28800,
        57343: 57344,
        57344: 57344,
    }
    for lc, width in expected.items():
        assert score_width(lc, is_cuda=True, is_bf16=True) == width
        assert score_width(lc, is_cuda=False, is_bf16=True) == lc
        assert score_width(lc, is_cuda=True, is_bf16=False) == lc

        logical = list(range(lc))
        padded = logical + [10**12] * (width - lc)
        sliced = padded[:lc]
        assert sliced == logical
        if lc:
            assert max(sliced) == lc - 1
    print("SGLang #45794 aligned-width CPU falsifier: PASS")


if __name__ == "__main__":
    main()
