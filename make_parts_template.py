"""Draft annotations/parts.csv from annotations/<stem>_labels.png (0 = background, k = part_id).

    python make_parts_template.py            # writes annotations/parts.csv (refuses to overwrite) + outputs/annotation_preview/
    python make_parts_template.py --out annotations/parts_draft.csv

The columns type / head_x / head_y are GUESSES (nut = has a large bore and is round; screw head = the thicker end of the
long axis). `finish` is left empty. Look at the previews (blue = nut, red = screw, green dot = head), CORRECT the csv,
and fill in `finish`. Coordinates are in the pixel frame of the label image. Skips augmented images (__aug).
"""
import argparse
from pathlib import Path
import cv2, numpy as np, pandas as pd
from scipy import ndimage as ndi
from skimage import measure

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SMALL_FRAC = 8e-4          # filled area below this fraction of the image -> flag "check_small"

def guess(m):
    filled = ndi.binary_fill_holes(m); farea = filled.sum()
    holes = [r for r in measure.regionprops(measure.label(filled & ~m, connectivity=1)) if r.area >= 0.03 * farea]
    ys, xs = np.nonzero(m); pts = np.c_[xs, ys].astype(float); c = pts.mean(0)
    _, s, vt = np.linalg.svd(pts - c, full_matrices=False); aspect = s[0] / max(s[1], 1e-6)
    typ = "nut" if (holes and aspect < 1.5) else "screw"
    hx = hy = np.nan
    if typ == "screw":
        # head = the end whose WIDEST point (largest inscribed radius in the outer 25%) is bigger: heads are wider than
        # the shank/thread, tips taper. (Mean thickness of the extreme pixels fails on thin bolt flanges.)
        proj = (pts - c) @ vt[0]; dt = ndi.distance_transform_edt(filled)[ys, xs]
        span = proj.max() - proj.min()
        lo, hi = proj <= proj.min() + 0.25 * span, proj >= proj.max() - 0.25 * span
        end = hi if dt[hi].max() > dt[lo].max() else lo
        sub = proj >= proj.max() - 0.12 * span if end is hi else proj <= proj.min() + 0.12 * span
        hx, hy = pts[sub].mean(0)
    return typ, hx, hy, int(farea)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="annotations/parts.csv"); a = ap.parse_args()
    if Path(a.out).exists(): raise SystemExit(f"{a.out} exists - not overwriting (use --out to write elsewhere)")
    prev = Path("outputs/annotation_preview"); prev.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(Path("annotations").glob("*_labels.png")):
        stem = f.name[:-len("_labels.png")]
        if "__aug" in stem: continue
        lab = cv2.imread(str(f), cv2.IMREAD_UNCHANGED); lab = lab[..., 0] if lab.ndim == 3 else lab
        img_f = next((p for p in Path("inputs").glob(stem + ".*") if p.suffix.lower() in EXT), None)
        vis = cv2.imread(str(img_f)) if img_f else np.full(lab.shape + (3,), 255, np.uint8)
        if vis.shape[:2] != lab.shape: vis = cv2.resize(vis, (lab.shape[1], lab.shape[0]))
        for k in np.unique(lab[lab > 0]):
            m = lab == k; typ, hx, hy, fa = guess(m)
            note = "check_small(maybe chip)" if fa < SMALL_FRAC * lab.size else ""
            rows.append(dict(image=img_f.name if img_f else stem, part_id=int(k), type=typ, finish="",
                             head_x=round(hx, 1) if hx == hx else "", head_y=round(hy, 1) if hy == hy else "", notes=note))
            col = (255, 80, 0) if typ == "nut" else (0, 0, 255)
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cs, -1, col, 3)
            y, x = np.argwhere(m).min(0)
            cv2.putText(vis, f"{k}{'?' if note else ''}", (int(x), int(max(y - 8, 20))), cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3)
            if hx == hx: cv2.circle(vis, (int(hx), int(hy)), 12, (0, 200, 0), -1)
        s = 1200 / max(vis.shape[:2]); cv2.imwrite(str(prev / f"{stem}.png"), cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
    pd.DataFrame(rows).to_csv(a.out, index=False); print(len(rows), "parts ->", a.out, "| previews in", prev)
