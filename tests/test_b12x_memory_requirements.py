"""Regression tests for local b12x_next memory-composition fixes."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TYPES = ROOT / "runtime/b12x_next/b12x_next/preparation/types.py"


def _load_types():
    spec = importlib.util.spec_from_file_location("ds41_b12x_preparation_types", TYPES)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class _Scratch:
    nbytes: int


class MemoryRequirementsTests(unittest.TestCase):
    def test_sequential_scratch_keeps_larger_raw_variant_on_aligned_tie(self):
        """Both orderings must retain 321 B, not the first 257 B variant."""
        types = _load_types()
        for sizes in ((257, 321), (321, 257)):
            with self.subTest(sizes=sizes):
                requirements = tuple(
                    types.MemoryRequirements(scratch=(_Scratch(size),)) for size in sizes
                )
                combined = types.MemoryRequirements.sequential(requirements)
                self.assertEqual(combined.scratch_nbytes, 512)
                self.assertGreaterEqual(
                    sum(item.nbytes for item in combined.scratch), max(sizes)
                )


if __name__ == "__main__":
    unittest.main()
