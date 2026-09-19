#!/usr/bin/env python3
"""
verify_bundle.py -- OFFLINE verifier for an IBVAP evidence bundle.

Usage:
    unzip evidence_<alert-id>.zip -d bundle/
    pip install cryptography
    python verify_bundle.py bundle/

Needs no network and no IBVAP server. Checks, for every custody block:
    1. block hash recomputed from its contents        (detects edited entries)
    2. Ed25519 signature against ledger_public_key.pem (detects forged entries)
    3. block appears at the same idx in the chain skeleton, and the skeleton
       links contiguously (detects a block spliced in from elsewhere)
and for every evidence file:
    4. SHA-256 of the file == hash sealed in the ledger  (detects altered media)
    5. SHA-256 at each "viewed" moment == sealed hash    (detects late tampering)

Exit code 0 = everything verified, 1 = something failed.
NOTE: the block-hash preimage below must stay identical to core/ledger.block_hash().
"""
import hashlib
import json
import os
import sys

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def block_hash(b):
    return hashlib.sha256(canonical({
        "idx": b["idx"], "ts_us": b["ts_us"], "entry_type": b["entry_type"], "alert_id": b["alert_id"],
        "actor": b["actor"], "payload": b["payload"], "prev_hash": b["prev_hash"],
    }).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(folder):
    with open(os.path.join(folder, "custody.json")) as f:
        c = json.load(f)
    with open(os.path.join(folder, "ledger_public_key.pem"), "rb") as f:
        pub = serialization.load_pem_public_key(f.read())

    problems, sealed = [], {}
    skeleton = {s["idx"]: s for s in c["chain_skeleton"]}
    sk_idxs = sorted(skeleton)

    # skeleton must be contiguous and hash-linked
    for a, b in zip(sk_idxs, sk_idxs[1:]):
        if b != a + 1 or skeleton[b]["prev_hash"] != skeleton[a]["hash"]:
            problems.append(f"chain skeleton broken between blocks {a} and {b}")

    print(f"Alert {c['alert_id']}  |  {len(c['blocks'])} custody blocks  |  key fp {c['pubkey_fp']}\n")
    for b in c["blocks"]:
        ok_hash = block_hash(b) == b["hash"]
        try:
            pub.verify(bytes.fromhex(b["sig"]), bytes.fromhex(b["hash"]))
            ok_sig = True
        except (InvalidSignature, ValueError):
            ok_sig = False
        sk = skeleton.get(b["idx"])
        ok_chain = bool(sk) and sk["hash"] == b["hash"] and sk["prev_hash"] == b["prev_hash"]
        status = "OK " if (ok_hash and ok_sig and ok_chain) else "FAIL"
        print(f"  [{status}] #{b['idx']:<6} {b['entry_type']:<16} actor={b['actor']:<14} "
              f"hash={'ok' if ok_hash else 'BAD'} sig={'ok' if ok_sig else 'BAD'} chain={'ok' if ok_chain else 'BAD'}")
        if status == "FAIL":
            problems.append(f"block {b['idx']} ({b['entry_type']}) failed verification")
        p = b["payload"]
        if b["entry_type"] == "ALERT_RAISED" and p.get("snapshot_sha256"):
            sealed[p["snapshot_file"]] = p["snapshot_sha256"]
        if b["entry_type"] == "CLIP_SEALED":
            sealed[p["clip_file"]] = p["clip_sha256"]

    print()
    for fname, expected in sealed.items():
        path = os.path.join(folder, "evidence", fname)
        if not os.path.isfile(path):
            print(f"  [MISSING] {fname}")
            problems.append(f"evidence file {fname} not in bundle")
            continue
        actual = sha256_file(path)
        good = actual == expected
        print(f"  [{'OK ' if good else 'FAIL'}] {fname}  sha256={actual[:16]}...")
        if not good:
            problems.append(f"evidence file {fname} does not match its sealed SHA-256")

    for b in c["blocks"]:
        if b["entry_type"] == "VIEWED_EVIDENCE":
            p = b["payload"]
            if p["file"] in sealed and p["sha256_at_view"] != sealed[p["file"]]:
                problems.append(f"file {p['file']} had a different hash when viewed by {b['actor']} (block {b['idx']})")

    print()
    if problems:
        print("VERIFICATION FAILED:")
        for x in problems:
            print("  -", x)
        return 1
    print("VERIFICATION PASSED: custody trail and evidence files are intact.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python verify_bundle.py <unzipped-bundle-folder>")
    sys.exit(main(sys.argv[1]))
