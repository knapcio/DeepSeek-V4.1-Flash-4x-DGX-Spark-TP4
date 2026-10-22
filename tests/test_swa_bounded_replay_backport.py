#!/usr/bin/env python3
"""CPU regression tests for the SGLang #41303 short-tail floor backport."""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
    late_layer_tail_layout,
)


class ShortTailFloorTests(unittest.TestCase):
    def test_single_short_extend_keeps_the_full_prior_window(self):
        indices, tail_lens, floor = late_layer_tail_layout(
            extend_lens_cpu=[40],
            seq_lens_cpu=[300],
            tail_len=128,
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(indices, torch.arange(40))
        self.assertEqual(tail_lens, [40])
        torch.testing.assert_close(floor, torch.full((40,), 172, dtype=torch.int32))

    def test_mixed_batch_uses_one_full_window_per_request(self):
        _, tail_lens, floor = late_layer_tail_layout(
            extend_lens_cpu=[40, 200, 50],
            seq_lens_cpu=[300, 200, 50],
            tail_len=128,
            device=torch.device("cpu"),
        )
        self.assertEqual(tail_lens, [40, 128, 50])
        expected = torch.tensor([172] * 40 + [72] * 128 + [0] * 50, dtype=torch.int32)
        torch.testing.assert_close(floor, expected)


if __name__ == "__main__":
    unittest.main()
