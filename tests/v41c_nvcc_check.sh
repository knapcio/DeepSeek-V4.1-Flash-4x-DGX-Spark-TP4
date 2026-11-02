#!/usr/bin/env bash
# GPU-less sm_121a compile of the v41c FlashInfer decode module next to the stock base, with ptxas -v.
#   bash tests/v41c_nvcc_check.sh OUT_DIR        (ds-b3-cutecheck:1: CUDA 13.0 nvcc)
# Fails on a compile error or any spill in a v41c (EXTRA_FP4) kernel; prints a per-kernel register / spill /
# shared-memory table for both builds (OUT_DIR/ptxas-{stock,v41c}.txt, OUT_DIR/summary.json).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; OUT="${1:?usage: v41c_nvcc_check.sh OUT_DIR}"; mkdir -p "$OUT"
F="$ROOT/runtime/flashinfer_v41c"
B="$ROOT/runtime/flashinfer_pce/base"   # the approved FlashInfer closure (K8)
T=$(mktemp -d)
FLAGS=(-std=c++17 -O3 -gencode arch=compute_121a,code=sm_121a --expt-relaxed-constexpr -Xptxas -v -c)
nvcc "${FLAGS[@]}" -I "$B/include" "$B/csrc/sparse_mla_sm120_decode_dsv4.cu" -o "$T/stock.o" > "$OUT/ptxas-stock.txt" 2>&1; s=$?
nvcc "${FLAGS[@]}" -I "$F/include" "$F/csrc/v41c_decode_dsv4.cu" -o "$T/v41c.o" > "$OUT/ptxas-v41c.txt" 2>&1; v=$?
python3 - "$OUT" "$s" "$v" <<'PY'
import json, re, sys
out, s, v = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
def parse(path):
    rows, cur = {}, None
    for ln in open(path, errors="replace"):
        m = re.search(r"Compiling entry function '(\S+)'", ln)
        if m:
            cur = m.group(1); rows[cur] = {}
        if cur is None:
            continue
        m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", ln)
        if m: rows[cur]["spill"] = int(m.group(1)) + int(m.group(2))
        m = re.search(r"Used (\d+) registers", ln)
        if m: rows[cur]["regs"] = int(m.group(1))
        m = re.search(r"(\d+) bytes smem", ln)
        if m: rows[cur]["smem"] = int(m.group(1))
    return rows
st, vc = parse(f"{out}/ptxas-stock.txt"), parse(f"{out}/ptxas-v41c.txt")
fp4 = {k: r for k, r in vc.items() if "dsv4_kernel" in k}
summ = {"stock_rc": s, "v41c_rc": v, "stock_kernels": len(st), "v41c_kernels": len(vc),
        "v41c_fp4_kernels": len(fp4),
        "v41c_max_regs": max([r.get("regs", 0) for r in fp4.values()] or [0]),
        "stock_max_regs": max([r.get("regs", 0) for k, r in st.items() if "dsv4_kernel" in k] or [0]),
        "v41c_spills": sum(r.get("spill", 0) for r in fp4.values()),
        "v41c_smem": sorted({r.get("smem") for r in fp4.values()}),
        "stock_smem": sorted({r.get("smem") for k, r in st.items() if "dsv4_kernel" in k})}
json.dump(summ, open(f"{out}/summary.json", "w"), indent=1)
print(json.dumps(summ))
ok = s == 0 and v == 0 and summ["v41c_fp4_kernels"] >= 25 and summ["v41c_spills"] == 0
sys.exit(0 if ok else 1)
PY
rc=$?
rm -rf "$T"
exit $rc
