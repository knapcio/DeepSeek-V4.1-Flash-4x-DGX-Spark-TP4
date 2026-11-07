#!/usr/bin/env python3
"""Compare the DS image's FlashInfer, as recorded by the v41w window receipt, with the approved base the v41c and K8
readers are written against (runtime/flashinfer_pce/base = APPROVED_CLOSURE.json).

    python3 tests/v41c_receipt_vs_base.py RECEIPT.json
The receipt's "flashinfer_reader_base" holds the image's FlashInfer version, git commit and the sha256 of six
sparse_mla_sm120 sources (plus mla/_sparse_mla_sm120.py, reported only). Exit 0 = every recorded source equals the base
(no rebase needed; the image build's full closure check still applies), 1 = a difference (rebase the reader patch and
re-approve the closure), 75 = the receipt lacks the hashes.
"""
import hashlib
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "runtime" / "flashinfer_pce" / "base"


def find(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = find(v, key)
            if r is not None:
                return r
    if isinstance(d, list):
        for v in d:
            r = find(v, key)
            if r is not None:
                return r
    return None


def main(argv):
    fi = find(json.load(open(argv[1])), "flashinfer_reader_base")
    if not isinstance(fi, dict):
        print("receipt has no flashinfer_reader_base")
        return 75
    print(f"image FlashInfer {fi.get('version')} git {fi.get('git_commit')}")
    diff, n = [], 0
    for rel, sha in sorted(fi.items()):
        if not rel.startswith("data/"):
            continue
        local = BASE / rel[len("data/"):]
        if sha is None:
            print(f"  {rel}: not recorded")
            continue
        n += 1
        mine = hashlib.sha256(local.read_bytes()).hexdigest() if local.is_file() else None
        print(f"  {rel}: {'same' if mine == sha else 'DIFFERENT'}")
        if mine != sha:
            diff.append(rel)
    if n == 0:
        return 75
    print("RECEIPT == BASE" if not diff else f"REBASE NEEDED: {diff}")
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
