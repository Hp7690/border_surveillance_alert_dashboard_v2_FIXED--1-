"""
demo_tamper.py -- live demo for judges: "an insider edits the evidence log; watch it get caught."

Every attack runs on its OWN fresh COPY of your real data/ledger.db (the real ledger is never touched).
    python demo_tamper.py
"""
import os, shutil, sqlite3, sys, tempfile
from core.ledger import EvidenceLedger

SRC = "data"
if not os.path.exists(os.path.join(SRC, "ledger.db")):
    sys.exit("No data/ledger.db yet -- run the app and let it raise a few CRITICAL/HIGH alerts first.")


def fresh():
    tmp = tempfile.mkdtemp()
    for f in ("ledger.db", "ledger_key.pem", "ledger_anchors.jsonl"):
        if os.path.exists(os.path.join(SRC, f)):
            shutil.copy(os.path.join(SRC, f), tmp)
    led = EvidenceLedger(db_path=f"{tmp}/ledger.db", key_path=f"{tmp}/ledger_key.pem",
                         anchor_path=f"{tmp}/ledger_anchors.jsonl",
                         snapshot_dir=os.path.join(SRC, "snapshots"), clip_dir=os.path.join(SRC, "clips"))
    db = sqlite3.connect(f"{tmp}/ledger.db")
    for t in ("blocks_no_update", "blocks_no_delete"):
        db.execute(f"DROP TRIGGER IF EXISTS {t}")     # the insider first removes the DB safety triggers...
    return led, db


def verdict(title, led):
    r = led.verify_chain()
    print(f"{title:<52} -> " + ("CHAIN INTACT" if r["ok"] else f"TAMPERING DETECTED (block {r['first_bad_idx']}): {r['reason']}"))


led, db = fresh()
verdict("1. Untouched ledger", led)
sealed = [r[0] for r in db.execute("SELECT idx FROM blocks WHERE entry_type='ALERT_RAISED' ORDER BY idx")]
if len(sealed) < 2:
    sys.exit("Need at least 2 sealed alerts for the demo -- raise a couple more.")

led, db = fresh()
db.execute("UPDATE blocks SET actor='officer_X' WHERE idx=?", (sealed[-1],)); db.commit()
verdict(f"2. Insider rewrites block #{sealed[-1]} (who raised it)", led)

led, db = fresh()
db.execute("DELETE FROM blocks WHERE idx=?", (sealed[0],)); db.commit()
verdict(f"3. Insider deletes block #{sealed[0]} (hides an old alert)", led)

led, db = fresh()
db.execute("UPDATE blocks SET payload_json=REPLACE(payload_json, '\"severity\":\"CRITICAL\"', '\"severity\":\"LOW\"') "
           "WHERE entry_type='ALERT_RAISED'"); db.commit()
verdict("4. Insider downgrades CRITICAL alerts to LOW", led)

print("\nReal ledger untouched. (Editing evidence FILES is caught separately: the /ledger page shows TAMPERED.)")
