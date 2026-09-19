r"""
export_bundle.py -- build + offline-verify a court evidence bundle WITHOUT the browser.

    python export_bundle.py                 # latest sealed alert
    python export_bundle.py <alert_id>      # a specific alert (id shown on the /ledger page)
    python export_bundle.py --list          # show the newest sealed alerts

Creates  bundles\evidence_<id>.zip  and  bundles\evidence_<id>\  then runs verify_bundle.py on it.
The export itself is logged in the ledger (EXPORTED, actor "cli:<windows user>").
Safe to run while the app is running (SQLite serialises writes).
"""
import getpass, io, os, sys, time, zipfile

from core.ledger import EvidenceLedger
import verify_bundle

DATA = "data"
led = EvidenceLedger(db_path=os.path.join(DATA, "ledger.db"), key_path=os.path.join(DATA, "ledger_key.pem"),
                     anchor_path=os.path.join(DATA, "ledger_anchors.jsonl"),
                     snapshot_dir=os.path.join(DATA, "snapshots"), clip_dir=os.path.join(DATA, "clips"))

sealed = led.list_sealed_alerts(limit=20)
if "--list" in sys.argv:
    for a in sealed:
        print(f"{a['alert_id']}  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(a['ts_us'] / 1e6))}  "
              f"{a['severity']:<8} {a['event_type']}")
    sys.exit(0)

args = [a for a in sys.argv[1:] if not a.startswith("--")]
if args:
    alert_id = args[0]
elif sealed:
    alert_id = sealed[0]["alert_id"]
else:
    sys.exit("No sealed alerts in data/ledger.db yet. Trigger a CRITICAL/HIGH alert (or a face event) first.")

here = os.path.dirname(os.path.abspath(__file__))
try:
    data = led.build_bundle(alert_id, "cli:" + getpass.getuser(),
                            verifier_script_path=os.path.join(here, "verify_bundle.py"))
except KeyError:
    sys.exit(f"Alert {alert_id} is not in the ledger. Use --list to see valid ids.")

os.makedirs("bundles", exist_ok=True)
zpath = os.path.join("bundles", f"evidence_{alert_id}.zip")
with open(zpath, "wb") as f:
    f.write(data)
folder = os.path.join("bundles", f"evidence_{alert_id}")
zipfile.ZipFile(io.BytesIO(data)).extractall(folder)
print(f"Bundle : {os.path.abspath(zpath)}\nFolder : {os.path.abspath(folder)}\n")
sys.exit(verify_bundle.main(folder))
