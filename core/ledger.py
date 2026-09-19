"""
ledger.py
----------
Tamper-evident evidence ledger + chain-of-custody for IBVAP.

WHAT THIS IS
    A permissioned, append-only, hash-chained and digitally-signed log.
    Every block commits to:
        - the previous block's hash            (any edit/delete/reorder breaks the chain)
        - SHA-256 of the evidence file itself  (snapshot JPEG / video clip)
        - who did what, and when               (system / operator identity)
    and is signed with this node's Ed25519 key, so a block cannot be forged
    by someone who can only edit the database file.

WHAT IT RECORDS (entry types)
    ALERT_RAISED     a detection fired (weapon / intrusion / face ...)  actor=SYSTEM
    CLIP_SEALED      the video clip finished encoding, its SHA-256 is fixed
    ACKNOWLEDGED     operator X acknowledged the alert
    VIEWED_EVIDENCE  operator X opened the snapshot/clip (hash at view time)
    ACTION_TAKEN     operator X recorded a response (dispatched / false alarm ...)
    EXPORTED         operator X downloaded the court evidence bundle
    ANCHOR           optional checkpoint marker

HONEST SECURITY MODEL  (read this before claiming "immutable")
    * A single-node hash chain is TAMPER-EVIDENT, not tamper-PROOF. Someone
      with root on this machine can delete the DB and the key and start over.
      What they cannot do is *silently* change history that has already been
      anchored elsewhere or signed by a key they do not hold.
    * Therefore: (1) keep data/ledger_key.pem readable only by the service
      account, ideally in an HSM/TPM, and (2) push anchors (head hash + idx,
      emitted every ANCHOR_EVERY blocks) OFF this machine -- to the webhook
      / C2 server, a second BOP node, or a Hyperledger Fabric channel. That
      is what turns "evident" into "practically immutable".
    * The storage interface is small on purpose (append / verify_chain /
      custody / verify_alert_evidence), so this class can be swapped for a
      Hyperledger Fabric chaincode client without touching app.py.
    * PRIVACY: identities (e.g. a matched face name) are NEVER written to the
      chain in clear -- only a keyed HMAC commitment. An immutable ledger
      containing personal data cannot honour erasure requests; commitments
      let the mutable event DB be corrected/erased while the chain stays valid.
"""

import hashlib
import hmac
import io
import json
import os
import sqlite3
import threading
import time
import zipfile

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "ledger.db")
KEY_PATH = os.path.join(DATA_DIR, "ledger_key.pem")
ANCHOR_PATH = os.path.join(DATA_DIR, "ledger_anchors.jsonl")

GENESIS_PREV = "0" * 64
ANCHOR_EVERY = 25          # emit an external-anchor checkpoint every N blocks
VIEW_DEDUP_SECONDS = 600   # thumbnails reload constantly; log a "view" once per 10 min per user/artifact

ACTION_CODES = {"DISPATCHED", "ESCALATED", "MONITORING", "FALSE_ALARM", "RESOLVED", "OTHER"}


# ----------------------------------------------------------------- helpers
def canonical(obj) -> str:
    """Deterministic JSON (sorted keys, no whitespace) -- what gets hashed."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def block_hash(idx, ts_us, entry_type, alert_id, actor, payload, prev_hash) -> str:
    """The exact preimage of a block hash. verify_bundle.py re-implements THIS -- keep in sync."""
    return sha256_bytes(canonical({
        "idx": idx, "ts_us": ts_us, "entry_type": entry_type, "alert_id": alert_id,
        "actor": actor, "payload": payload, "prev_hash": prev_hash,
    }).encode())


class EvidenceLedger:
    def __init__(self, db_path=DB_PATH, key_path=KEY_PATH, anchor_path=ANCHOR_PATH,
                 snapshot_dir=None, clip_dir=None, anchor_every=ANCHOR_EVERY, anchor_hook=None):
        self.db_path = db_path
        self.anchor_path = anchor_path
        self.snapshot_dir = snapshot_dir
        self.clip_dir = clip_dir
        self.anchor_every = anchor_every
        self.anchor_hook = anchor_hook          # callable(anchor_dict) -> push OFF-box (webhook/Fabric/2nd node)
        self._lock = threading.Lock()
        self._view_seen = {}
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._key = self._load_or_create_key(key_path)
        self._pub = self._key.public_key()
        # Key for HMAC commitments (derived, so no second secret to manage).
        self._commit_key = hashlib.sha256(
            b"ibvap-ledger-commit|" + self._key.private_bytes(
                serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                serialization.NoEncryption())).digest()
        self._init_db()

    # --------------------------------------------------------------- keys
    @staticmethod
    def _load_or_create_key(path):
        # A zero-byte file (e.g. left by an interrupted first run) is treated as "no key yet".
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "rb") as f:
                return serialization.load_pem_private_key(f.read(), password=None)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        key = Ed25519PrivateKey.generate()
        pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())
        # NOTE: plain open(), not os.fdopen(): app.py calls eventlet.monkey_patch(), which replaces
        # os.fdopen with GreenPipe -- that raises NotImplementedError on Windows.
        with open(path, "wb") as f:
            f.write(pem)
        try:
            os.chmod(path, 0o600)      # effective on Linux/macOS; best-effort no-op on Windows
        except OSError:
            pass
        return key

    @property
    def public_key_pem(self) -> str:
        return self._pub.public_bytes(serialization.Encoding.PEM,
                                      serialization.PublicFormat.SubjectPublicKeyInfo).decode()

    @property
    def key_fingerprint(self) -> str:
        raw = self._pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        return sha256_bytes(raw)[:16]

    def commit(self, obj) -> str:
        """Keyed commitment to sensitive data (identities) -- see PRIVACY note above."""
        return hmac.new(self._commit_key, canonical(obj).encode(), hashlib.sha256).hexdigest()

    # ----------------------------------------------------------------- db
    def _connect(self):
        return sqlite3.connect(self.db_path, check_same_thread=False, timeout=15)

    def _init_db(self):
        with self._lock, self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS blocks (
                    idx INTEGER PRIMARY KEY,
                    ts_us INTEGER NOT NULL,
                    entry_type TEXT NOT NULL,
                    alert_id TEXT,
                    actor TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    hash TEXT NOT NULL,
                    sig TEXT NOT NULL
                )""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_blocks_alert ON blocks (alert_id)")
            # Defence-in-depth only: an attacker with file access can drop these,
            # but then the hash chain + signatures expose the tampering anyway.
            conn.execute("""CREATE TRIGGER IF NOT EXISTS blocks_no_update BEFORE UPDATE ON blocks
                            BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END""")
            conn.execute("""CREATE TRIGGER IF NOT EXISTS blocks_no_delete BEFORE DELETE ON blocks
                            BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END""")
            if conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0] == 0:
                self._insert(conn, "GENESIS", None, "SYSTEM",
                             {"note": "IBVAP evidence ledger genesis", "pubkey_fp": self.key_fingerprint})

    def _insert(self, conn, entry_type, alert_id, actor, payload):
        """Must be called with self._lock held and inside a transaction."""
        row = conn.execute("SELECT idx, hash FROM blocks ORDER BY idx DESC LIMIT 1").fetchone()
        idx, prev = (row[0] + 1, row[1]) if row else (0, GENESIS_PREV)
        ts_us = int(time.time() * 1_000_000)
        h = block_hash(idx, ts_us, entry_type, alert_id, actor, payload, prev)
        sig = self._key.sign(bytes.fromhex(h)).hex()
        conn.execute(
            "INSERT INTO blocks (idx, ts_us, entry_type, alert_id, actor, payload_json, prev_hash, hash, sig) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (idx, ts_us, entry_type, alert_id, actor, canonical(payload), prev, h, sig))
        return {"idx": idx, "ts_us": ts_us, "entry_type": entry_type, "alert_id": alert_id,
                "actor": actor, "payload": payload, "prev_hash": prev, "hash": h, "sig": sig}

    def append(self, entry_type, alert_id, actor, payload, anchor_now=False):
        """anchor_now=True pushes the new head off-box immediately. Blocks written after the
        latest anchor could be rolled back undetected, so the entries that matter most
        (CRITICAL alerts, every human action) are anchored on the spot."""
        with self._lock, self._connect() as conn:
            block = self._insert(conn, entry_type, alert_id, actor or "UNKNOWN", payload or {})
        if anchor_now or (self.anchor_every and block["idx"] % self.anchor_every == 0):
            self._emit_anchor(block)
        return block

    def _emit_anchor(self, block):
        anchor = {"type": "ledger_anchor", "idx": block["idx"], "hash": block["hash"],
                  "sig": block["sig"], "ts_us": block["ts_us"], "pubkey_fp": self.key_fingerprint}
        try:
            with open(self.anchor_path, "a") as f:
                f.write(canonical(anchor) + "\n")
        except Exception as e:
            print(f"[Ledger] could not write local anchor: {e}")
        if self.anchor_hook:
            try:
                self.anchor_hook(anchor)
            except Exception as e:
                print(f"[Ledger] anchor hook failed: {e}")
        return anchor

    def force_anchor(self):
        """Emit a checkpoint for the current head right now (e.g. on shutdown / nightly cron)."""
        head = self.head()
        return self._emit_anchor(head)

    # ------------------------------------------------------- domain entries
    def record_alert(self, alert, snapshot_path=None):
        """alert = Alert.to_dict(). Seals the alert + the SHA-256 of its snapshot."""
        details = alert.get("details") or {}
        payload = {
            "event_type": alert["event_type"],
            "severity": alert["severity"],
            "camera_id": alert["camera_id"],
            "threat_score": alert.get("threat_score", 0),
            "alert_ts_us": int(alert["timestamp"] * 1_000_000),
            "details_commit": self.commit(details),
            "snapshot_file": os.path.basename(snapshot_path) if snapshot_path else None,
            "snapshot_sha256": sha256_file(snapshot_path) if snapshot_path and os.path.isfile(snapshot_path) else None,
        }
        return self.append("ALERT_RAISED", alert["id"], "SYSTEM", payload,
                           anchor_now=(alert["severity"] == "CRITICAL"))

    def record_clip(self, alert_id, clip_path):
        if not os.path.isfile(clip_path):
            return None
        return self.append("CLIP_SEALED", alert_id, "SYSTEM", {
            "clip_file": os.path.basename(clip_path),
            "clip_sha256": sha256_file(clip_path),
            "clip_bytes": os.path.getsize(clip_path),
        })

    def record_ack(self, alert_id, actor, ip=None):
        return self.append("ACKNOWLEDGED", alert_id, actor, {"ip": ip}, anchor_now=True)

    def record_action(self, alert_id, actor, action, note="", ip=None):
        action = (action or "").upper()
        if action not in ACTION_CODES:
            raise ValueError(f"action must be one of {sorted(ACTION_CODES)}")
        return self.append("ACTION_TAKEN", alert_id, actor,
                           {"action": action, "note": (note or "")[:500], "ip": ip}, anchor_now=True)

    def record_view(self, alert_id, actor, artifact, path, ip=None):
        """Log that `actor` opened an evidence file, with the file's hash AT VIEW TIME."""
        if not self.is_sealed(alert_id):
            return None  # only alerts that were sealed as evidence carry a custody trail
        key = (alert_id, actor, artifact)
        now = time.time()
        if now - self._view_seen.get(key, 0) < VIEW_DEDUP_SECONDS:
            return None
        self._view_seen[key] = now
        return self.append("VIEWED_EVIDENCE", alert_id, actor, {
            "artifact": artifact, "file": os.path.basename(path),
            "sha256_at_view": sha256_file(path), "ip": ip})

    # -------------------------------------------------------------- queries
    @staticmethod
    def _row_to_block(r):
        return {"idx": r[0], "ts_us": r[1], "entry_type": r[2], "alert_id": r[3], "actor": r[4],
                "payload": json.loads(r[5]), "prev_hash": r[6], "hash": r[7], "sig": r[8]}

    _COLS = "idx, ts_us, entry_type, alert_id, actor, payload_json, prev_hash, hash, sig"

    def head(self):
        with self._lock, self._connect() as conn:
            r = conn.execute(f"SELECT {self._COLS} FROM blocks ORDER BY idx DESC LIMIT 1").fetchone()
        return self._row_to_block(r)

    def is_sealed(self, alert_id):
        with self._lock, self._connect() as conn:
            return conn.execute("SELECT 1 FROM blocks WHERE alert_id = ? AND entry_type = 'ALERT_RAISED' LIMIT 1",
                                (alert_id,)).fetchone() is not None

    def custody(self, alert_id):
        with self._lock, self._connect() as conn:
            rows = conn.execute(f"SELECT {self._COLS} FROM blocks WHERE alert_id = ? ORDER BY idx",
                                (alert_id,)).fetchall()
        return [self._row_to_block(r) for r in rows]

    def list_sealed_alerts(self, limit=100):
        """Most-recent-first summary of sealed alerts + custody status, for the UI."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"SELECT {self._COLS} FROM blocks WHERE entry_type='ALERT_RAISED' ORDER BY idx DESC LIMIT ?",
                (limit,)).fetchall()
            ids = [r[3] for r in rows]
            stats = {}
            if ids:
                q = ",".join("?" * len(ids))
                for aid, et, n in conn.execute(
                        f"SELECT alert_id, entry_type, COUNT(*) FROM blocks WHERE alert_id IN ({q}) "
                        f"GROUP BY alert_id, entry_type", ids):
                    stats.setdefault(aid, {})[et] = n
        out = []
        for r in rows:
            b = self._row_to_block(r)
            s = stats.get(b["alert_id"], {})
            out.append({
                "alert_id": b["alert_id"], "idx": b["idx"], "ts_us": b["ts_us"],
                "event_type": b["payload"]["event_type"], "severity": b["payload"]["severity"],
                "camera_id": b["payload"]["camera_id"], "threat_score": b["payload"]["threat_score"],
                "acknowledged": s.get("ACKNOWLEDGED", 0) > 0,
                "actions": s.get("ACTION_TAKEN", 0), "views": s.get("VIEWED_EVIDENCE", 0),
                "clip_sealed": s.get("CLIP_SEALED", 0) > 0,
            })
        return out

    # --------------------------------------------------------- verification
    def _verify_sig(self, block):
        try:
            self._pub.verify(bytes.fromhex(block["sig"]), bytes.fromhex(block["hash"]))
            return True
        except (InvalidSignature, ValueError):
            return False

    def _read_anchors(self):
        if not os.path.exists(self.anchor_path):
            return []
        out = []
        with open(self.anchor_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        out.append({"_corrupt": True})
        return out

    def verify_chain(self):
        """
        Re-derive every hash, check every link and signature, and cross-check the
        recorded anchors. Returns {"ok", "blocks", "head_idx", "head_hash",
        "first_bad_idx", "reason", "anchors_checked"}.
        """
        with self._lock, self._connect() as conn:
            rows = conn.execute(f"SELECT {self._COLS} FROM blocks ORDER BY idx").fetchall()
        result = {"ok": True, "blocks": len(rows), "head_idx": None, "head_hash": None,
                  "first_bad_idx": None, "reason": None, "anchors_checked": 0}

        def bad(idx, reason):
            result.update(ok=False, first_bad_idx=idx, reason=reason)
            return result

        prev, expect, by_idx = GENESIS_PREV, 0, {}
        for r in rows:
            b = self._row_to_block(r)
            if b["idx"] != expect:
                return bad(expect, f"block {expect} missing or reordered (found idx {b['idx']})")
            if b["prev_hash"] != prev:
                return bad(b["idx"], "prev_hash does not link to the previous block")
            if block_hash(b["idx"], b["ts_us"], b["entry_type"], b["alert_id"], b["actor"],
                          b["payload"], b["prev_hash"]) != b["hash"]:
                return bad(b["idx"], "block contents do not match its hash (edited)")
            if not self._verify_sig(b):
                return bad(b["idx"], "signature invalid (not signed by this node's key)")
            by_idx[b["idx"]] = b["hash"]
            prev, expect = b["hash"], expect + 1
        if rows:
            result["head_idx"], result["head_hash"] = expect - 1, prev

        for a in self._read_anchors():
            if a.get("_corrupt"):
                return bad(None, "anchor file contains a corrupt line")
            i = a.get("idx")
            if i not in by_idx:
                return bad(i, f"anchor references block {i} that no longer exists (chain truncated/rolled back)")
            if by_idx[i] != a.get("hash"):
                return bad(i, f"block {i} differs from the hash anchored earlier (history rewritten)")
            result["anchors_checked"] += 1
        return result

    def verify_alert_evidence(self, alert_id):
        """Recompute SHA-256 of the evidence files on disk and compare with what was sealed."""
        blocks = self.custody(alert_id)
        sealed = {"snapshot": None, "clip": None}
        for b in blocks:
            if b["entry_type"] == "ALERT_RAISED":
                sealed["snapshot"] = (b["payload"].get("snapshot_file"), b["payload"].get("snapshot_sha256"))
            elif b["entry_type"] == "CLIP_SEALED":
                sealed["clip"] = (b["payload"].get("clip_file"), b["payload"].get("clip_sha256"))
        dirs = {"snapshot": self.snapshot_dir, "clip": self.clip_dir}
        out = {}
        for kind, entry in sealed.items():
            if not entry or not entry[0] or not entry[1]:
                out[kind] = {"status": "NOT_SEALED", "file": None, "expected": None, "actual": None}
                continue
            fname, expected = entry
            path = os.path.join(dirs[kind] or "", fname)
            if not os.path.isfile(path):
                out[kind] = {"status": "MISSING", "file": fname, "expected": expected, "actual": None}
                continue
            actual = sha256_file(path)
            out[kind] = {"status": "OK" if actual == expected else "TAMPERED",
                         "file": fname, "expected": expected, "actual": actual}
        return out

    def stats(self):
        h = self.head()
        return {"blocks": h["idx"] + 1, "head_idx": h["idx"], "head_hash": h["hash"],
                "pubkey_fp": self.key_fingerprint, "anchor_every": self.anchor_every}

    # --------------------------------------------------------------- bundle
    def build_bundle(self, alert_id, actor, ip=None, verifier_script_path=None):
        """
        Court-ready ZIP: evidence files + signed custody blocks + chain skeleton +
        public key + an offline verifier. Logs an EXPORTED block *first* so the
        export itself is part of the custody trail that gets shipped.
        """
        if not self.custody(alert_id):
            raise KeyError(alert_id)
        self.append("EXPORTED", alert_id, actor, {"ip": ip, "what": "evidence_bundle"}, anchor_now=True)
        blocks = self.custody(alert_id)
        with self._lock, self._connect() as conn:
            skeleton = [{"idx": r[0], "prev_hash": r[1], "hash": r[2]} for r in conn.execute(
                "SELECT idx, prev_hash, hash FROM blocks ORDER BY idx")]
        evidence = self.verify_alert_evidence(alert_id)
        chain = self.verify_chain()

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("custody.json", json.dumps({
                "alert_id": alert_id, "exported_by": actor, "exported_ts_us": int(time.time() * 1e6),
                "pubkey_fp": self.key_fingerprint, "blocks": blocks,
                "chain_skeleton": skeleton,
                "server_side_check": {"chain": chain, "evidence": evidence},
            }, indent=2))
            z.writestr("ledger_public_key.pem", self.public_key_pem)
            for kind, d in (("snapshot", self.snapshot_dir), ("clip", self.clip_dir)):
                info = evidence.get(kind, {})
                if info.get("file") and info["status"] in ("OK", "TAMPERED"):
                    z.write(os.path.join(d, info["file"]), f"evidence/{info['file']}")
            if verifier_script_path and os.path.isfile(verifier_script_path):
                z.write(verifier_script_path, "verify_bundle.py")
            z.writestr("README.txt",
                       "Offline verification:\n  pip install cryptography\n  python verify_bundle.py <this-unzipped-folder>\n"
                       "The script recomputes every block hash + Ed25519 signature and the SHA-256 of each "
                       "evidence file, without contacting the IBVAP server.\n")
        return buf.getvalue()
