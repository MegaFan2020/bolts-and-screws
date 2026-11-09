"""Pre-fill the `finish` column of annotations/parts.csv for the BLACK parts only (the one finish that colour identifies
reliably: median lightness of the part, edges eroded, is far below every other finish). Every other row is left empty for
you to fill in by looking at the photo. Existing finish values are never overwritten. A backup parts.csv.bak is written.

    python prefill_finish.py
"""
from pathlib import Path
import shutil, cv2, numpy as np, pandas as pd
EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
L_BLACK = 48          # cv2 Lab lightness (0-255). Measured: black parts 12-32, darkest non-black part 63.

df = pd.read_csv("annotations/parts.csv", dtype={"finish": str}); shutil.copy("annotations/parts.csv", "annotations/parts.csv.bak")
df["finish"] = df["finish"].fillna(""); n = 0
for img_name, g in df.groupby("image"):
    stem = Path(str(img_name)).stem
    f = next((p for p in Path("inputs").glob(stem + ".*") if p.suffix.lower() in EXT), None)
    if f is None or not Path(f"annotations/{stem}_labels.png").exists(): continue
    img = cv2.imread(str(f)); lab = cv2.imread(f"annotations/{stem}_labels.png", cv2.IMREAD_UNCHANGED); lab = lab[..., 0] if lab.ndim == 3 else lab
    if lab.shape != img.shape[:2]: lab = cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
    for i in g.index:
        m = (lab == int(df.at[i, "part_id"])).astype(np.uint8)
        if m.sum() == 0: continue
        e = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
        e = e if e.sum() >= 30 else m > 0
        if df.at[i, "finish"] == "" and np.median(L[e]) < L_BLACK: df.at[i, "finish"] = "black-oxidized"; n += 1
df.to_csv("annotations/parts.csv", index=False); print(f"filled {n} of {len(df)} rows with black-oxidized; {int((df.finish == '').sum())} left for you")
