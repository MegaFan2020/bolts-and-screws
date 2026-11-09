"""Small helper to correct the draft annotations without any image editor.

All x, y, r values are PIXELS AS YOU SEE THEM IN outputs/annotation_preview/<image>.png (the 1200-px-wide preview;
most image viewers show the cursor position). The script converts them to the full-resolution label image itself.

  python edit_labels.py delete  screws_061  1 2 11          # remove parts (chips) from the label image AND from parts.csv
  python edit_labels.py carve   screws_028  5  640 410 9    # make a hole (nut bore) in part 5: circle centre x y, radius r
  python edit_labels.py head    screws_026  3  455 300      # set the head position of screw 3
After each command the preview image is redrawn. Part ids are never renumbered.
"""
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd

ANN, PREV = Path("annotations"), Path("outputs/annotation_preview")
EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def load_lab(stem):
    f = ANN / f"{stem}_labels.png"
    if not f.exists(): sys.exit(f"{f} not found")
    lab = cv2.imread(str(f), cv2.IMREAD_UNCHANGED); return lab[..., 0] if lab.ndim == 3 else lab
def load_csv():
    df = pd.read_csv(ANN / "parts.csv"); df["_stem"] = df["image"].astype(str).map(lambda s: Path(s).stem); return df
def save_csv(df): df.drop(columns="_stem").to_csv(ANN / "parts.csv", index=False)
def prev_scale(lab): return 1200 / max(lab.shape)

def redraw(stem, lab, df):
    f = next((p for p in Path("inputs").glob(stem + ".*") if p.suffix.lower() in EXT), None)
    vis = cv2.imread(str(f)) if f else np.full(lab.shape + (3,), 255, np.uint8)
    if vis.shape[:2] != lab.shape: vis = cv2.resize(vis, (lab.shape[1], lab.shape[0]))
    rows = {int(r.part_id): r for r in df[df._stem == stem].itertuples()}
    for k in np.unique(lab[lab > 0]):
        r = rows.get(int(k)); typ = r.type if r is not None else "screw"
        col = (255, 80, 0) if typ == "nut" else (0, 0, 255)
        m = (lab == k).astype(np.uint8); cs, _ = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, col, 3); y, x = np.argwhere(m).min(0)
        cv2.putText(vis, str(k), (int(x), int(max(y - 8, 20))), cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3)
        if r is not None and pd.notna(r.head_x) and pd.notna(r.head_y) and str(r.head_x) != "":
            cv2.circle(vis, (int(float(r.head_x)), int(float(r.head_y))), 12, (0, 200, 0), -1)
    PREV.mkdir(parents=True, exist_ok=True); s = prev_scale(lab)
    cv2.imwrite(str(PREV / f"{stem}.png"), cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))

if __name__ == "__main__":
    if len(sys.argv) < 4: sys.exit(__doc__)
    cmd, stem, a = sys.argv[1], sys.argv[2], sys.argv[3:]
    lab, df = load_lab(stem), load_csv(); s = prev_scale(lab)
    if cmd == "delete":
        ids = [int(i) for i in a]; lab[np.isin(lab, ids)] = 0
        n = len(df); df = df[~((df._stem == stem) & df.part_id.isin(ids))]
        print(f"removed ids {ids} from the label image; removed {n - len(df)} row(s) from parts.csv")
    elif cmd == "carve":
        pid, x, y, r = int(a[0]), *[float(v) / s for v in a[1:4]]
        hole = np.zeros(lab.shape, np.uint8); cv2.circle(hole, (int(x), int(y)), int(r), 1, -1)
        before = int((lab == pid).sum()); lab[(hole > 0) & (lab == pid)] = 0
        print(f"part {pid}: {before - int((lab == pid).sum())} pixels turned into a hole")
    elif cmd == "head":
        pid, x, y = int(a[0]), float(a[1]) / s, float(a[2]) / s
        i = df.index[(df._stem == stem) & (df.part_id == pid)]
        if len(i) == 0: sys.exit("part not in parts.csv")
        df.loc[i, "head_x"], df.loc[i, "head_y"] = round(x, 1), round(y, 1); print(f"head of part {pid} set")
    else: sys.exit(__doc__)
    cv2.imwrite(str(ANN / f"{stem}_labels.png"), lab.astype(np.uint16)); save_csv(df); redraw(stem, lab, df); print("done; preview updated")
