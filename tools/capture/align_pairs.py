"""Turns photos of the capture slideshow into aligned training pairs.

Usage:
  python3 tools/capture/align_pairs.py KIT_DIR PHOTOS_DIR OUT_DIR

For each photo: identify the slide (ArUco marker ids, or feature matching for close-ups that
cut the markers off), estimate the slide->photo homography, refine it with ECC, warp the photo
into slide coordinates and save (input, target, valid-mask). Rejects blurry frames, frames
caught mid-transition and poor alignments. Writes report.csv with the reason for every photo.
"""
import csv, glob, json, os, sys
import cv2, numpy as np

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
W, H = 1920, 1080
MARKER, PAD = 110, 14
BOX = MARKER + 2 * PAD
ORIGINS = [(0, 0), (W - BOX, 0), (W - BOX, H - BOX), (0, H - BOX)]


def marker_corners_in_slide(idx):
    x, y = ORIGINS[idx]
    x += PAD; y += PAD
    return np.float32([[x, y], [x + MARKER, y], [x + MARKER, y + MARKER], [x, y + MARKER]])


class Kit:
    def __init__(self, kit_dir):
        self.dir = kit_dir
        self.manifest = json.load(open(os.path.join(kit_dir, "manifest.json")))
        self._gt = {}
        self._feat = None
        self.sift = cv2.SIFT_create(4000)

    def gt(self, k):
        if k not in self._gt:
            self._gt[k] = cv2.imread(os.path.join(self.dir, "slides", self.manifest[k]["file"]))
        return self._gt[k]

    def features(self):
        if self._feat is None:
            self._feat = {}
            for m in self.manifest:
                g = cv2.cvtColor(self.gt(m["id"]), cv2.COLOR_BGR2GRAY)
                kp, d = self.sift.detectAndCompute(g, None)
                if d is not None and len(kp) >= 50:
                    self._feat[m["id"]] = (np.float32([p.pt for p in kp]), d)
        return self._feat


def detect_markers(photo):
    """Returns {slide_id: [(marker_index, 4x2 corners in photo px)]}."""
    det = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())
    s = min(1.0, 2400 / max(photo.shape[:2]))
    g = cv2.cvtColor(cv2.resize(photo, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    corners, ids, _ = det.detectMarkers(g)
    found = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            found.setdefault(int(i) // 4, []).append((int(i) % 4, c.reshape(4, 2) / s))
    return found


def homography_from_markers(markers):
    src = np.concatenate([marker_corners_in_slide(i) for i, _ in markers])
    dst = np.concatenate([c for _, c in markers]).astype(np.float32)
    if len(markers) >= 2:
        H_, _ = cv2.findHomography(src, dst, 0)
    else:
        H_ = cv2.getPerspectiveTransform(src, dst)  # single marker: rough, ECC/feature refine follows
    return H_


def homography_from_features(kit, photo, only=None):
    """Matches photo features against slides; returns (slide_id, H slide->photo, inliers)."""
    s = min(1.0, 1600 / max(photo.shape[:2]))
    g = cv2.cvtColor(cv2.resize(photo, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), 1.2)  # soften moire before feature detection
    kp, d = kit.sift.detectAndCompute(g, None)
    if d is None or len(kp) < 30:
        return None
    pts = np.float32([p.pt for p in kp]) / s
    bf = cv2.BFMatcher()
    cands = []
    for k, (spts, sd) in kit.features().items():
        if only is not None and k != only:
            continue
        m = bf.knnMatch(sd, d, k=2)
        good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.8 * b.distance]
        if len(good) < 25:
            continue
        H_, inl = cv2.findHomography(spts[[g_.queryIdx for g_ in good]], pts[[g_.trainIdx for g_ in good]], cv2.RANSAC, 4.0 / s)
        if H_ is None:
            continue
        n = int(inl.sum())
        if n >= 40:
            cands.append((k, H_, n))
    return sorted(cands, key=lambda c: -c[2])[:3]


def edge_score(warped, gt, valid, tol=2):
    """(recall, precision) of slide edges vs photo edges within tol px, inside the valid area.
    Robust to moire, glare and colour shifts; drops sharply for misalignment or mixed slides."""
    def edges(img):
        return cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 1.5), 30, 90) > 0
    m = cv2.erode(valid, np.ones((7, 7), np.uint8)) > 0
    ew, eg = edges(warped) & m, edges(gt) & m
    if eg.sum() < 200 or ew.sum() < 200:
        return 0.0, 0.0
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    rec = float((cv2.dilate(ew.astype(np.uint8), k) > 0)[eg].mean())
    prec = float((cv2.dilate(eg.astype(np.uint8), k) > 0)[ew].mean())
    return rec, prec


def warp_to_slide(photo, H_):
    warped = cv2.warpPerspective(photo, H_, (W, H), flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP)
    valid = cv2.warpPerspective(np.full(photo.shape[:2], 255, np.uint8), H_, (W, H),
                                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    return warped, valid


def refine(kit, photo, gt, H_):
    """Residual homography from features between the slide and the warped photo; kept only if
    the edge score improves."""
    warped, valid = warp_to_slide(photo, H_)
    base = edge_score(warped, gt, valid)
    a = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    b = cv2.GaussianBlur(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
    ka, da = kit.sift.detectAndCompute(a, None)
    kb, db = kit.sift.detectAndCompute(b, (valid > 0).astype(np.uint8))
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return H_, base
    m = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [x for x, y in (t for t in m if len(t) == 2) if x.distance < 0.75 * y.distance]
    if len(good) < 30:
        return H_, base
    R, _ = cv2.findHomography(np.float32([ka[g.queryIdx].pt for g in good]),
                              np.float32([kb[g.trainIdx].pt for g in good]), cv2.RANSAC, 2.0)
    if R is None:
        return H_, base
    H2 = H_ @ R
    w2, v2 = warp_to_slide(photo, H2)
    s2 = edge_score(w2, gt, v2)
    return (H2, s2) if sum(s2) > sum(base) else (H_, base)


def sharpness(photo):
    g = cv2.cvtColor(cv2.resize(photo, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def process(kit, path, out_dir, min_rec=0.8, min_prec=0.9, min_sharp=25.0):
    photo = cv2.imread(path)
    if photo is None:
        return dict(status="unreadable")
    sh = sharpness(photo)
    if sh < min_sharp:
        return dict(status="blurry", sharp=round(sh, 1))
    found = detect_markers(photo)
    k, H_, how = None, None, ""
    if found:
        k = max(found, key=lambda s: len(found[s]))
        if k >= len(kit.manifest):
            k = None
    if k is not None and len(found[k]) >= 2:
        H_, how = homography_from_markers(found[k]), f"markers:{len(found[k])}"
    else:
        cands = homography_from_features(kit, photo, only=k)  # one marker narrows the search to its slide
        if not cands:
            return dict(status="no match", sharp=round(sh, 1))
        # Several slides can share features (similar UI chrome): keep the best edge agreement.
        best = None
        for ck, cH, n in cands:
            rH, sc = refine(kit, photo, kit.gt(ck), cH)
            if best is None or sum(sc) > sum(best[2]):
                best = (ck, rH, sc, n)
        k, H_, (rec, prec), n = best
        how = f"features:{n}"
    gt = kit.gt(k)
    if how.startswith("markers"):
        H_, (rec, prec) = refine(kit, photo, gt, H_)
    warped, valid = warp_to_slide(photo, H_)
    valid = cv2.erode(valid, np.ones((9, 9), np.uint8))  # drop resampling edge
    # Markers are not real content: exclude them from the training mask.
    for x, y in ORIGINS:
        valid[y:y + BOX, x:x + BOX] = 0
    cover = float((valid > 0).mean())
    row = dict(slide=k, kind=kit.manifest[k]["kind"], how=how, rec=round(rec, 3), prec=round(prec, 3),
               cover=round(cover, 3), sharp=round(sh, 1))
    if prec < min_prec:
        return dict(row, status="mid-transition or misaligned")
    if rec < min_rec:
        return dict(row, status="misaligned or too soft")
    if cover < 0.05:
        return dict(row, status="too little of the slide visible")
    name = os.path.splitext(os.path.basename(path))[0]
    ys, xs = np.where(valid > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    cv2.imwrite(os.path.join(out_dir, f"{name}_in.png"), warped[y0:y1, x0:x1])
    cv2.imwrite(os.path.join(out_dir, f"{name}_gt.png"), gt[y0:y1, x0:x1])
    cv2.imwrite(os.path.join(out_dir, f"{name}_mask.png"), valid[y0:y1, x0:x1])
    return dict(row, status="ok")


def main():
    kit_dir, photos_dir, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    kit = Kit(kit_dir)
    files = sorted(p for p in glob.glob(os.path.join(photos_dir, "**", "*"), recursive=True)
                   if p.lower().endswith((".jpg", ".jpeg", ".png", ".heic", ".webp")))
    rows = []
    for p in files:
        r = process(kit, p, out_dir)
        r["file"] = os.path.relpath(p, photos_dir)
        rows.append(r)
        print(f"{r['file']}: {r['status']} {r.get('how', '')} rec={r.get('rec', '-')} prec={r.get('prec', '-')}", flush=True)
    keys = ["file", "status", "slide", "kind", "how", "rec", "prec", "cover", "sharp"]
    with open(os.path.join(out_dir, "report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    ok = sum(r["status"] == "ok" for r in rows)
    print(f"\n{ok} of {len(rows)} photos became training pairs.")


if __name__ == "__main__":
    main()
