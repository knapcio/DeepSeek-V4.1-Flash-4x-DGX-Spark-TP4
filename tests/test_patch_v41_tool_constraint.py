#!/usr/bin/env python3
"""Host-side idempotence and refusal tests for the V4.1 detector patcher."""
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / 'runtime' / 'patch_v41_tool_constraint.py'
BACKPORT = ROOT / 'runtime' / 'sglang' / 'deepseekv41_detector.py'


def run(path):
    return subprocess.run(
        [sys.executable, '-I', str(PATCHER), str(path)],
        capture_output=True,
        text=True,
    )


def main():
    source = PATCHER.read_text()
    digest = hashlib.sha256(BACKPORT.read_bytes()).hexdigest()
    assert f'BACKPORT_SHA256 = "{digest}"' in source
    with tempfile.TemporaryDirectory() as directory:
        applied = Path(directory) / 'applied.py'
        shutil.copyfile(BACKPORT, applied)
        result = run(applied)
        assert result.returncode == 0 and 'already applied' in result.stdout

        unknown = Path(directory) / 'unknown.py'
        unknown.write_text('class DeepSeekV41Detector: pass\n')
        result = run(unknown)
        assert result.returncode == 1 and 'neither the stock' in result.stderr
        assert unknown.read_text() == 'class DeepSeekV41Detector: pass\n'
    print('PASS test_patch_v41_tool_constraint')


if __name__ == '__main__':
    main()
