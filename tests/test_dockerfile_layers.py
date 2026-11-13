#!/usr/bin/env python3
"""Layer budget of the serving recipes (issue #83).

A Docker daemon on the overlay2 graph driver (the classic image store, not the containerd one) refuses a build step
once the image would pass 129 layers: "failed to prepare ...: max depth exceeded". Every RUN, COPY, ADD and
WORKDIR step counts, also one that changes nothing. Measured with Docker 29.0.4 (dind, overlay2, BuildKit): a
1-layer base plus 128 RUN steps builds (129 layers); the 129th RUN fails at step [130/N] with that error.

The pinned base (lmsysorg/sglang:dev-dsv41@sha256:3dbc3130..., linux/arm64 manifest sha256:800cc9ad...) has 77
layers, so a recipe gets 52 steps. Dockerfile.canary-roce at v2.7.7 used 86 and failed at its 53rd step on such a
daemon; a containerd image store has no such limit, which is why it built elsewhere. The checks now share a few RUN
steps. This test keeps HEADROOM steps free for future levers.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_DIGEST = "sha256:3dbc313030a6ef2c5d7de8ecf48e9aece722694a82182cb618cc82b588816349"
BASE_LAYERS = 77      # rootfs.diff_ids of the pinned linux/arm64 image
MAX_LAYERS = 129      # overlay2 graph driver: the step that would make layer 130 fails
HEADROOM = 10
RECIPES = ("Dockerfile", "Dockerfile.canary", "Dockerfile.canary-roce")
LAYER_STEPS = {"RUN", "COPY", "ADD", "WORKDIR"}


def final_stage_steps(text):
    """Instruction keywords of the last stage (comments dropped, continuation lines joined)."""
    stages, joined = [], ""
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#") or (not line and not joined):
            continue
        if line.endswith("\\"):
            joined += line[:-1] + " "
            continue
        joined += line
        word = joined.split(None, 1)[0].upper()
        if word == "FROM":
            stages.append([])
        elif stages:
            stages[-1].append(word)
        joined = ""
    return stages[-1]


class LayerBudget(unittest.TestCase):
    def recipes(self):
        found = [(name, ROOT / name) for name in RECIPES if (ROOT / name).is_file()]
        if not found:
            self.skipTest("no recipes here")
        return found

    def test_base_is_the_counted_one(self):
        """BASE_LAYERS belongs to this digest; a new base needs a new count."""
        for name, path in self.recipes():
            with self.subTest(name=name):
                self.assertRegex(path.read_text(), r"ARG BASE_IMAGE=\S+@" + re.escape(BASE_DIGEST))

    def test_every_recipe_fits_with_headroom(self):
        for name, path in self.recipes():
            with self.subTest(name=name):
                steps = sum(1 for word in final_stage_steps(path.read_text()) if word in LAYER_STEPS)
                total = BASE_LAYERS + steps
                self.assertLessEqual(
                    total, MAX_LAYERS - HEADROOM,
                    f"{name}: {BASE_LAYERS} base + {steps} steps = {total} layers; the overlay2 limit is "
                    f"{MAX_LAYERS}. Chain new checks into an existing RUN (set -e; a && b) instead of adding steps.")

    def test_counter(self):
        text = "FROM a AS b\nRUN x\nFROM a\n# RUN y\nWORKDIR /w\nCOPY a \\\n  b /c/\nRUN p \\\n && q\nENV K=v\n"
        self.assertEqual(final_stage_steps(text), ["WORKDIR", "COPY", "RUN", "ENV"])


if __name__ == "__main__":
    unittest.main()
