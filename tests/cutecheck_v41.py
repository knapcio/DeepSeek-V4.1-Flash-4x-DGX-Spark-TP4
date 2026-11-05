"""GPU-less sm_121a compile check of the v41c kernels (run in ds-b3-cutecheck:1, no GPU).

  python3 tests/cutecheck_v41.py OUT_DIR [JOB]
JOB: writer-indexed (the V4.1 ratio-1/2 record writer, the model's arithmetic) | writer-swa
(control: the unchanged 528-byte SWA writer). One JOB per process (the DSL reads its dump dir
once). Fake tensors stand in for every argument; CUTE_DSL_ARCH=sm_121a, PTX and CUBIN kept,
then ptxas -v reports registers / spills. A failure after the CUBIN exists (module load needs
a driver) still counts as compiled. Prints one JSON line per job and exits non-zero on a
compile error or any spill.
"""
import glob
import json
import os
import re
import subprocess
import sys
import traceback

OUT = sys.argv[1]
JOB = sys.argv[2] if len(sys.argv) > 2 else "writer-indexed"
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")
os.environ["CUTE_DSL_KEEP_PTX"] = "1"
os.environ["CUTE_DSL_KEEP_CUBIN"] = "1"
os.environ["CUTE_DSL_DUMP_DIR"] = OUT
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "runtime", "b12x_next"))

import cuda.bindings.driver as cuda  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass import Int32, Int64  # noqa: E402
from cutlass.cute.runtime import make_fake_tensor  # noqa: E402


def fake(dtype, ndim, leading, align):
    return make_fake_tensor(dtype, tuple(cute.sym_int(32) for _ in range(ndim)),
                            tuple(1 if i == leading else cute.sym_int(64) for i in range(ndim)),
                            assumed_align=align)


def compile_writer(kind):
    from b12x_next.attention._shared.mla import kv_cache as kc
    kernel = kc._build_compressed_cache_writer(256, kind)
    kv = fake(cutlass.BFloat16, 2, 1, 4)
    cache = fake(cutlass.Uint8, 2, 1, 16)
    slots = fake(cutlass.Int64, 1, 0, 8)
    tail = (Int64(256 * (528 if kind == "swa" else 288)), Int64(528 if kind == "swa" else 288),
            Int64(256 * 1024), Int32(64), cuda.CUstream(0))
    if kind == "swa":
        args = (kv, cache, slots, Int64(512), *tail)
    else:
        args = (kv, kv, cache, slots, Int64(512), Int64(512), *tail)
    return cute.compile(kernel, *args)


def ptxas_report():
    cubins = sorted(glob.glob(os.path.join(OUT, "**", "*.cubin"), recursive=True))
    ptxs = sorted(glob.glob(os.path.join(OUT, "**", "*.ptx"), recursive=True))
    row = {"ptx": len(ptxs), "cubin": len(cubins)}
    if ptxs:
        ptxas = os.environ.get("PTXAS", "ptxas")
        for cand in (ptxas, "/usr/local/cuda/bin/ptxas"):
            try:
                r = subprocess.run([cand, "-arch=sm_121a", "-v", "--warn-on-spills", ptxs[-1], "-o",
                                    os.path.join(OUT, "check.cubin")], capture_output=True, text=True)
            except FileNotFoundError:
                continue
            txt = r.stdout + r.stderr
            regs = re.findall(r"Used (\d+) registers", txt)
            spills = re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", txt)
            row.update(ptxas_rc=r.returncode, regs=max(map(int, regs)) if regs else None,
                       spills=sum(int(a) + int(b) for a, b in spills))
            break
    return row


def main():
    row = {"job": JOB}
    try:
        compile_writer("swa" if JOB == "writer-swa" else "indexed")
        row["compiled"] = True
    except Exception as exc:  # noqa: BLE001 - a module-load failure after codegen still counts
        row["compiled"] = False
        row["error"] = repr(exc)[:400]
        traceback.print_exc()
    row.update(ptxas_report())
    if not row["compiled"] and row.get("cubin"):
        row["compiled"] = True
        row["note"] = "cubin written; module load needs a driver"
    print(json.dumps(row))
    ok = row["compiled"] and row.get("ptxas_rc", 0) == 0 and not row.get("spills")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
