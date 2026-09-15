"""Repair pass: re-query unparseable multi pilot rows with strict letter-only prompt."""
import csv
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from pilot_vlm import (load_key, resolve_video, contact_sheet_b64, call_model,
                       parse_final, ROOT, CATS)
import re

STRICT = (
    "You are an expert activity-recognition judge. The attached image is a 16-frame "
    "contact sheet (chronological, top-left to bottom-right) from a DEPTH video "
    "(false-color; brighter = closer). No RGB is available.\n\n"
    "Question: {q}\nOptions:\n{opts}\n\n"
    "Multiple options may apply. Mentally judge each option PRESENT/ABSENT, then "
    "output ONLY the concatenated present letters in A-D order and NOTHING else. "
    "Example output: ABD"
)

key = load_key()
rows = list(csv.DictReader(open(ROOT / "training_qa.csv", encoding="utf-8-sig")))
by_id = {r["\ufeffqa_id" if "\ufeffqa_id" in r else "qa_id"]: r for r in rows}
res = list(csv.DictReader(open("research/t2_vlm/pilot_results.csv")))
fixed = 0
for r in res:
    if r["category"] == "multi" and r["pred"] == "" and r["err"] == "":
        q = by_id[r["qa_id"]]
        opts = "\n".join(f"{L}: {q[L]}" for L in "ABCD" if q[L].strip())
        b64 = contact_sheet_b64(resolve_video(q["path"]))
        txt, dt, pt, ct = call_model(key, STRICT.format(q=q["question"], opts=opts), b64)
        pred = parse_final(txt) or re.findall(r"\b([A-D]{1,4})\b", txt.upper())
        pred = pred if isinstance(pred, str) else (pred[0] if pred else "")
        ok = 1 if pred == q["answer"].strip().upper() else 0
        r.update(pred=pred, correct=str(ok), raw=txt.replace("\n", " | ")[:600],
                 dt=str(round(dt, 1)), pt=str(pt), ct=str(ct))
        fixed += 1
        print(f"{r['qa_id']} ans={r['answer']} pred={pred} ok={ok} raw={txt[:120]!r}", flush=True)
with open("research/t2_vlm/pilot_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=res[0].keys())
    w.writeheader(); w.writerows(res)
print(f"repaired {fixed} rows")
