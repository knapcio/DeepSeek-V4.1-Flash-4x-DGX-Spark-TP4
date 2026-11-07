#!/usr/bin/env python3
"""The NCCL library receipt of the canary-roce image: NCCL 2.30.7 rebuilt with NVIDIA/nccl #2393 (v2.6.6).

  nccl_receipt.py write LIB OUT.json   in the nccl-builder stage: describe the library just built, refuse unless it is
                                       NCCL 2.30.7 (ncclGetVersion 23007, banner "NCCL version 2.30.7+cuda...") with
                                       the #2393 fence (exactly one `dmb ishld` in ncclIbIsend: stock 2.30.7 has none)
  nccl_receipt.py check LIB OUT.json   in the final image: the library torch loads is byte for byte the receipt's
                                       (sha256, size) and still passes the checks above

Why the fence: on AArch64 (GB10's Grace cores) ncclIbIsend could read a peer's CTS `nreqs` before the `idx` check that
guards it (a weakly ordered load); the patch adds one acquire fence there (NVIDIA/nccl #2393; backport to 2.30.7 by
Stanislav Bardyuk, patches/nccl-2393-aarch64-cts-fence.patch). Host code only: the device code (.nv_fatbin) of a
patched and a stock rebuild is the same, and two builds of the same tree differ only in a few bytes of host .text
(nvcc host stubs), so the library sha256 is recorded, not pinned, and the device code digest is recorded with it.

Stdlib + binutils (objdump / objcopy, present in the base image) only; no GPU needed (ncclGetVersion does not touch
a device).
"""
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys

VERSION_CODE = 23007
BANNER_RE = re.compile(rb"NCCL version 2\.30\.7\+cuda[0-9.]+")
REF = "73cf112295c33aee2b895f329f592f2a9b4b0f97"        # NVIDIA/nccl v2.30.7-1
PATCH_SHA256 = "8088c3a898028ef4c926cbad51a36000a6d7dfc7fd5bda566420b9701bc61e2c"
PIP_LIB = "/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def version_code(lib):
    v = ctypes.c_int()
    rc = ctypes.CDLL(lib).ncclGetVersion(ctypes.byref(v))
    return v.value if rc == 0 else -1


def banner(lib):
    with open(lib, "rb") as f:
        m = BANNER_RE.search(f.read())
    return m.group(0).decode() if m else ""


def isend_fences(lib, objdump="objdump"):
    """(ncclIbIsend symbols, `dmb ishld` count inside them) from the disassembly."""
    out = subprocess.run([objdump, "-d", "-C", "--no-show-raw-insn", lib], check=True, capture_output=True,
                         text=True).stdout
    return count_isend_fences(out)


def count_isend_fences(disassembly):
    symbols = fences = 0
    inside = False
    for line in disassembly.splitlines():
        if re.match(r"^[0-9a-f]+ <", line):
            inside = line.split("<", 1)[1].startswith("ncclIbIsend(")
            symbols += inside
            continue
        if inside and re.search(r"\bdmb\s+ishld\b", line):
            fences += 1
    return symbols, fences


def section_sha256(lib, section, objcopy="objcopy"):
    r = subprocess.run([objcopy, "-O", "binary", "--only-section=" + section, lib, "/dev/stdout"],
                       check=True, capture_output=True)
    return hashlib.sha256(r.stdout).hexdigest()


def describe(lib):
    symbols, fences = isend_fences(lib)
    return {"lib_sha256": sha256(lib), "lib_bytes": os.path.getsize(lib), "nccl_version_code": version_code(lib),
            "banner": banner(lib), "isend_symbols": symbols, "isend_dmb_ishld": fences,
            "fatbin_sha256": section_sha256(lib, ".nv_fatbin")}


def problems(d):
    bad = []
    if d["nccl_version_code"] != VERSION_CODE:
        bad.append("ncclGetVersion %s, not %d (the version is never bumped)" % (d["nccl_version_code"], VERSION_CODE))
    if not d["banner"]:
        bad.append("no 'NCCL version 2.30.7+cuda...' banner")
    if d["isend_symbols"] != 1 or d["isend_dmb_ishld"] != 1:
        bad.append("ncclIbIsend: %d symbol(s), %d `dmb ishld` (want 1 / 1: the #2393 fence)"
                   % (d["isend_symbols"], d["isend_dmb_ishld"]))
    return bad


def main(argv):
    if len(argv) != 4 or argv[1] not in ("write", "check"):
        print(__doc__.split("\n\n")[0], file=sys.stderr)
        return 2
    mode, lib, receipt = argv[1:]
    d = describe(lib)
    bad = problems(d)
    if mode == "write":
        d.update({"ref": REF, "patch": "patches/nccl-2393-aarch64-cts-fence.patch", "patch_sha256": PATCH_SHA256})
        if not bad:
            with open(receipt, "w") as f:
                json.dump(d, f, indent=1, sort_keys=True)
                f.write("\n")
    else:
        want = json.load(open(receipt))
        for k in ("lib_sha256", "lib_bytes", "fatbin_sha256"):
            if d[k] != want.get(k):
                bad.append("%s %s is not the receipt's %s" % (k, d[k], want.get(k)))
    if bad:
        print("nccl_receipt %s %s: FAIL: %s" % (mode, lib, "; ".join(bad)), file=sys.stderr)
        return 1
    print("nccl_receipt %s: OK %s sha256=%s %s, ncclIbIsend fence present, device code %s"
          % (mode, lib, d["lib_sha256"][:16], d["banner"], d["fatbin_sha256"][:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
