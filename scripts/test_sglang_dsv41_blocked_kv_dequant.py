#!/usr/bin/env python3
"""CPU falsifier for #45793 block mapping, tails and FP4 output ordering."""

from __future__ import annotations

import random


BLOCK = 8


def scalar_bases(
    locs, page_size, bytes_per_page, data_bytes, scale_bytes, scale_off
):
    out = []
    for loc in locs:
        page_base = (loc // page_size) * bytes_per_page
        in_page = loc % page_size
        out.append(
            (
                page_base + in_page * data_bytes,
                page_base + scale_off + in_page * scale_bytes,
            )
        )
    return out


def blocked_bases(
    locs, page_size, bytes_per_page, data_bytes, scale_bytes, scale_off
):
    out = []
    for program in range((len(locs) + BLOCK - 1) // BLOCK):
        for lane in range(BLOCK):
            token_id = program * BLOCK + lane
            if token_id >= len(locs):
                continue
            loc = locs[token_id]
            page_base = (loc // page_size) * bytes_per_page
            in_page = loc % page_size
            out.append(
                (
                    page_base + in_page * data_bytes,
                    page_base + scale_off + in_page * scale_bytes,
                )
            )
    return out


def test_bases() -> None:
    rng = random.Random(45793)
    for layout in ((512, 16), (256, 32)):
        data_bytes, scale_bytes = layout
        for page_size in (16, 256):
            scale_off = page_size * data_bytes
            bytes_per_page = scale_off + page_size * scale_bytes + 64
            for count in (1, 7, 8, 9, 15, 16, 17, 63, 64, 65):
                locs = [rng.randrange(0, page_size * 7) for _ in range(count)]
                assert blocked_bases(
                    locs,
                    page_size,
                    bytes_per_page,
                    data_bytes,
                    scale_bytes,
                    scale_off,
                ) == scalar_bases(
                    locs,
                    page_size,
                    bytes_per_page,
                    data_bytes,
                    scale_bytes,
                    scale_off,
                )


def test_fp4_interleave_and_tail() -> None:
    rng = random.Random(4579301)
    for count in (1, 7, 8, 9, 15, 17):
        packed = [[rng.randrange(256) for _ in range(256)] for _ in range(count)]
        scalar = [
            [nibble for byte in row for nibble in (byte & 0xF, byte >> 4)]
            for row in packed
        ]
        output = [
            [-1] * 512
            for _ in range(((count + BLOCK - 1) // BLOCK) * BLOCK)
        ]
        for program in range((count + BLOCK - 1) // BLOCK):
            joined = []
            for lane in range(BLOCK):
                token_id = program * BLOCK + lane
                row = packed[token_id] if token_id < count else [0] * 256
                joined.append([[byte & 0xF, byte >> 4] for byte in row])
            reshaped = [
                [nibble for pair in lane for nibble in pair] for lane in joined
            ]
            for lane in range(BLOCK):
                token_id = program * BLOCK + lane
                if token_id < count:
                    output[token_id] = reshaped[lane]
        assert output[:count] == scalar
        assert all(row == [-1] * 512 for row in output[count:])


def main() -> None:
    test_bases()
    test_fp4_interleave_and_tail()
    print("SGLang #45793 blocked V4.1 KV CPU falsifier: PASS")


if __name__ == "__main__":
    main()
