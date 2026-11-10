"""Consistency check of annotations/parts.csv against annotations/<stem>_labels.png.   python check_annotations.py"""
from pathlib import Path
import cv2, numpy as np, pandas as pd
df = pd.read_csv("annotations/parts.csv", dtype=str); problems = []
def bad(msg): problems.append(msg); print("PROBLEM:", msg)
for col in ["image", "part_id", "type", "finish", "head_x", "head_y", "notes"]:
    if col not in df.columns: bad(f"column '{col}' missing")
df["stem"] = df.image.map(lambda s: Path(s).stem)
for stem, g in df.groupby("stem"):
    f = Path(f"annotations/{stem}_labels.png")
    if not f.exists(): bad(f"{f} missing"); continue
    lab = cv2.imread(str(f), cv2.IMREAD_UNCHANGED); lab = lab[..., 0] if lab.ndim == 3 else lab
    in_lab, in_csv = set(int(i) for i in np.unique(lab[lab > 0])), set(int(i) for i in g.part_id)
    for i in sorted(in_lab - in_csv): bad(f"{stem}: label image has part {i} but parts.csv has no row -> python edit_labels.py delete {stem} {i}  (or add a row)")
    for i in sorted(in_csv - in_lab): bad(f"{stem}: parts.csv has part {i} but it is not in the label image")
    for r in g.itertuples():
        if r.type not in ("screw", "nut"): bad(f"{stem} part {r.part_id}: type '{r.type}' must be screw or nut")
        if not isinstance(r.finish, str) or not r.finish.strip(): bad(f"{stem} part {r.part_id}: finish empty")
        if r.type == "screw":
            try:
                x, y = float(r.head_x), float(r.head_y)
                if not (0 <= x < lab.shape[1] and 0 <= y < lab.shape[0]): bad(f"{stem} part {r.part_id}: head outside the image")
                elif int(r.part_id) in in_lab and not (cv2.dilate((lab == int(r.part_id)).astype(np.uint8), np.ones((81, 81), np.uint8))[int(y), int(x)]):
                    bad(f"{stem} part {r.part_id}: head position is far from the part")
            except (TypeError, ValueError): bad(f"{stem} part {r.part_id}: screw without head_x/head_y")
        elif isinstance(r.head_x, str) and r.head_x.strip(): bad(f"{stem} part {r.part_id}: nut should have no head position")
print("\nparts per type  :", df.type.value_counts().to_dict())
print("parts per finish:", df.finish.value_counts().to_dict())
small = [k for k, v in df.finish.value_counts().items() if v < 6]
if small: print("WARNING: very few examples for", small, "- classifiers will struggle; merge the class or discuss it")
print("\nOK - no problems found" if not problems else f"\n{len(problems)} problem(s) found")
