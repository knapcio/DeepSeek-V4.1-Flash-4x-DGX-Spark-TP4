"""Compile the v41c group-reader Triton kernels for sm_121a without a GPU (Triton AOT + ptxas -v).

Not a numerics test (tests/test_v41c_fast.py runs the same kernels in the Triton interpreter): this proves the kernels
lower to PTX / SASS for GB10 with the launch parameters the reader uses, and prints registers / spills per kernel.

usage (ds-b3-cutecheck:1, --network none):  python3 tests/v41c_triton_aot.py
"""
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.pop("TRITON_INTERPRET", None)
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("v41c_aot", ROOT / "runtime" / "flashinfer_v41c" / "v41c.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

import triton  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

TARGET = GPUTarget("cuda", 121, 32)
CASES = [
    # (name, kernel, signature, constexprs, num_warps)
    ("plan pbs256 len", v._v41c_plan_kernel,
     {"src": "*u8", "src_stride": "i32", "PBS": "constexpr", "eidx": "*i32", "tlen": "*i32", "HAS_LEN": "constexpr",
      "K": "i32", "new_idx": "*i32", "dst": "*u8", "R": "constexpr"},
     {"PBS": 256, "HAS_LEN": True, "R": v.TOK_R}, 4),
    ("plan pbs128 nolen", v._v41c_plan_kernel,
     {"src": "*u8", "src_stride": "i32", "PBS": "constexpr", "eidx": "*i32", "tlen": "*i32", "HAS_LEN": "constexpr",
      "K": "i32", "new_idx": "*i32", "dst": "*u8", "R": "constexpr"},
     {"PBS": 128, "HAS_LEN": False, "R": v.TOK_R}, 4),
    ("dedup N4096", v._v41c_dedup_kernel,
     {"eidx": "*i32", "tlen": "*i32", "HAS_LEN": "constexpr", "K": "i32", "G": "i32", "new_idx": "*i32",
      "toks": "*i32", "cnt": "*i32", "tmp": "*i64", "N": "constexpr"},
     {"HAS_LEN": True, "N": 4096}, 8),
    ("list pbs256", v._v41c_list_kernel,
     {"src": "*u8", "src_stride": "i32", "PBS": "constexpr", "toks": "*i32", "cnt": "*i32", "N": "i32", "n": "i32",
      "dst": "*u8", "R": "constexpr"},
     {"PBS": 256, "R": v.TOK_R}, 4),
]


def ptxas_bin():
    base = Path(triton.__file__).parent / "backends" / "nvidia" / "bin"
    for name in ("ptxas-blackwell", "ptxas"):
        if (base / name).exists():
            return str(base / name)
    return "ptxas"


def main() -> int:
    bad = 0
    for name, fn, sig, const, warps in CASES:
        try:
            src = ASTSource(fn=fn, signature=sig, constexprs=const)
            k = triton.compile(src, target=TARGET, options={"num_warps": warps})
            ptx = k.asm["ptx"]
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "k.ptx"
                p.write_text(ptx)
                r = subprocess.run([ptxas_bin(), "-arch=sm_121a", "-v", str(p), "-o", str(Path(d) / "k.cubin")],
                                   capture_output=True, text=True)
            info = " ".join(re.findall(r"Used \d+ registers|\d+ bytes spill stores|\d+ bytes spill loads|"
                                       r"\d+ bytes smem", r.stderr))
            ok = r.returncode == 0 and "cubin" in k.asm
            print(f"AOT {'OK ' if ok else 'BAD'} {name:20s} warps={warps} ptx={len(ptx)}B cubin={len(k.asm.get('cubin', b''))}B"
                  f" | ptxas: {info or r.stderr.strip()[:200]}")
            bad += not ok
        except Exception as e:  # noqa: BLE001
            print(f"AOT BAD {name}: {type(e).__name__}: {str(e)[:400]}")
            bad += 1
    print("V41C AOT", "PASS" if not bad else f"FAIL ({bad})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
