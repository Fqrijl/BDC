import json
import csv
from pathlib import Path

JSON_PATH = Path("stack_out/qwen7b_lane/qwen7b_test_predictions.json")
CSV_PATH = Path("stack_out/qwen7b_lane/qwen7b_predictions.csv")

results = json.loads(JSON_PATH.read_text(encoding="utf-8"))

n_failed = 0
rows = []
for img_id, v in results.items():
    idx = v["label_idx"]
    if idx is None:
        n_failed += 1
        idx = -1  # gagal parse -> ditandai salah, bukan diam-diam di-skip
    rows.append((img_id, idx))

rows.sort(key=lambda r: int(r[0]) if str(r[0]).isdigit() else r[0])

CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "predicted"])
    w.writerows(rows)

print(f"[SAVED] {CSV_PATH} -- {len(rows)} baris ({n_failed} gagal parse -> ditandai -1)")
