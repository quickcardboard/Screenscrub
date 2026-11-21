"""Builds the capture kit: ~120 ground-truth slides with alignment markers, plus the slideshow page.

Usage:
  python3 tools/capture/make_slides.py OUT_DIR [--photos DIR] [--custom DIR]

Every slide is 1920x1080 with four ArUco markers (DICT_6X6_1000) in the corners. Slide k uses
marker ids 4k..4k+3 (TL, TR, BR, BL), so a photo identifies its slide and its alignment from the
markers alone. The slide PNG is the exact ground truth ("native screenshot") for that photo.
"""
import argparse, glob, json, os, random, shutil, subprocess, sys, tempfile
import cv2, numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ui_templates import TEMPLATES, words  # noqa: E402

W, H = 1920, 1080
MARKER, PAD = 110, 14  # marker size and white quiet zone, px
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
HERE = os.path.dirname(os.path.abspath(__file__))


def fit(img, w=W, h=H, fill=False):
    """Scale to cover (fill=True, cropping) or contain (letterbox on black)."""
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih) if fill else min(w / iw, h / ih)
    r = cv2.resize(img, (round(iw * s), round(ih * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    out = np.zeros((h, w, 3), np.uint8)
    rh, rw = r.shape[:2]
    if fill:
        y, x = (rh - h) // 2, (rw - w) // 2
        return r[y:y + h, x:x + w].copy()
    y, x = (h - rh) // 2, (w - rw) // 2
    out[y:y + rh, x:x + rw] = r
    return out


def add_markers(img, k):
    out = img.copy()
    box = MARKER + 2 * PAD
    for i, (x, y) in enumerate([(0, 0), (W - box, 0), (W - box, H - box), (0, H - box)]):
        out[y:y + box, x:x + box] = 255
        m = cv2.aruco.generateImageMarker(DICT, 4 * k + i, MARKER)
        out[y + PAD:y + PAD + MARKER, x + PAD:x + PAD + MARKER] = m[..., None]
    return out


# ---------- content generators ----------

def calibration():
    s = []
    for v in (255, 0, 128, 200, 60):
        s.append(("calib_flat_%d" % v, np.full((H, W, 3), v, np.uint8)))
    bars = np.zeros((H, W, 3), np.uint8)
    cols = [(255, 255, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 255), (0, 0, 255), (255, 0, 0), (0, 0, 0)]
    for i, c in enumerate(cols):
        bars[:, i * W // 8:(i + 1) * W // 8] = c
    s.append(("calib_bars", bars))
    g = np.tile(np.linspace(0, 255, W, dtype=np.float32), (H, 1))
    s.append(("calib_gradient", cv2.merge([g, g, g]).astype(np.uint8)))
    hsv = np.zeros((H, W, 3), np.uint8)
    hsv[..., 0] = np.tile(np.linspace(0, 179, W), (H, 1)).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.tile(np.linspace(255, 40, H)[:, None], (1, W)).astype(np.uint8)
    s.append(("calib_hue", cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)))
    steps = np.zeros((H, W, 3), np.uint8)
    for i in range(16):
        steps[:, i * W // 16:(i + 1) * W // 16] = i * 17
    s.append(("calib_steps", steps))
    # Fine detail: 1-px lines and checkerboards provoke moire, the main thing to learn to remove.
    fine = np.full((H, W, 3), 255, np.uint8)
    fine[:, ::2] = 0
    fine[H // 2:, :] = 255
    fine[H // 2::2, :] = 0
    s.append(("calib_fine_lines", fine))
    return s


def ui_slides(tmp):
    jobs = []
    for name, fn in TEMPLATES.items():
        for v in range(6):
            p = os.path.join(tmp, f"ui_{name}_{v}.html")
            with open(p, "w") as f:
                f.write(fn(v))
            jobs.append(p)
    script = os.path.join(tmp, "render.js")
    with open(script, "w") as f:
        f.write("""const { chromium } = require('playwright');
(async () => { const b = await chromium.launch(); const p = await b.newPage({viewport:{width:1920,height:1080}});
for (const f of process.argv.slice(2)) { await p.goto('file://' + f); await p.waitForTimeout(80);
  await p.screenshot({ path: f.replace(/\\.html$/, '.png') }); }
await b.close(); })();""")
    subprocess.run(["node", script, *jobs], check=True)
    return [(os.path.basename(j)[:-5], cv2.imread(j[:-5] + ".png")) for j in jobs]


def text(img, s, org, scale, color, thick=2, font=cv2.FONT_HERSHEY_DUPLEX):
    cv2.putText(img, s, org, font, scale, color, thick, cv2.LINE_AA)


def tv_slides(photos, rnd):
    out = []
    for i in range(24):
        bg = fit(cv2.imread(rnd.choice(photos)), fill=True)
        kind = i % 6
        if kind == 0:  # sports scoreboard
            cv2.rectangle(bg, (140, 120), (860, 200), (30, 30, 30), -1)
            cv2.rectangle(bg, (140, 120), (260, 200), (40, 40, 200), -1)
            cv2.rectangle(bg, (480, 120), (600, 200), (200, 80, 20), -1)
            text(bg, f"HOM {rnd.randint(0, 120)}   AWY {rnd.randint(0, 120)}   Q{rnd.randint(1, 4)} {rnd.randint(0, 11)}:{rnd.randint(10, 59)}",
                 (160, 175), 1.3, (255, 255, 255))
        elif kind == 1:  # news ticker
            cv2.rectangle(bg, (0, 800), (W, 900), (160, 30, 20), -1)
            cv2.rectangle(bg, (0, 800), (320, 900), (255, 255, 255), -1)
            text(bg, "BREAKING", (40, 865), 1.4, (160, 30, 20), 3)
            text(bg, words(rnd, 8).upper(), (350, 865), 1.3, (255, 255, 255))
        elif kind == 2:  # subtitles
            s = words(rnd, 7)
            (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX, 1.6, 3)
            text(bg, s, ((W - tw) // 2 + 3, 963), 1.6, (0, 0, 0), 6)
            text(bg, s, ((W - tw) // 2, 960), 1.6, (255, 255, 255), 3)
        elif kind == 3:  # streaming menu: dark grid of thumbnails
            bg[:] = 15
            text(bg, words(rnd, 2), (140, 230), 1.6, (240, 240, 240), 3)
            for r in range(2):
                for c in range(5):
                    th = fit(cv2.imread(rnd.choice(photos)), 300, 400, fill=True)
                    y, x = 280 + r * 400, 140 + c * 340
                    th = th[:380]
                    bg[y:y + 380, x:x + 300] = th
            cv2.rectangle(bg, (136, 276), (444, 664), (255, 255, 255), 5)
        elif kind == 4:  # game HUD
            cv2.rectangle(bg, (60, 60), (560, 100), (40, 40, 40), -1)
            cv2.rectangle(bg, (60, 60), (60 + rnd.randint(80, 500), 100), (60, 200, 60), -1)
            cv2.circle(bg, (W - 220, 220), 150, (30, 30, 30), -1)
            cv2.circle(bg, (W - 220, 220), 150, (220, 220, 220), 3)
            text(bg, f"{rnd.randint(1, 30)} / {rnd.randint(30, 90)}", (W - 420, H - 80), 2.0, (255, 255, 255), 4)
            text(bg, f"SCORE {rnd.randint(1000, 99999)}", (80, 170), 1.3, (0, 220, 255), 3)
        else:  # photo with caption overlay (like a smart display or TV screensaver)
            text(bg, f"{rnd.choice(['Mar', 'Jun', 'Aug', 'Dec'])}. {rnd.randint(1, 28)}, {rnd.randint(2010, 2025)}", (140, 900), 2.2, (255, 255, 255), 4)
            text(bg, words(rnd, 3), (140, 970), 1.4, (255, 255, 255), 2)
        out.append((f"tv_{['score', 'ticker', 'subs', 'menu', 'game', 'caption'][kind]}_{i}", bg))
    return out


def photo_slides(photos, rnd, n):
    return [(f"photo_{i}", fit(cv2.imread(p), fill=True)) for i, p in enumerate(rnd.sample(photos, n))]


def custom_slides(folder):
    out = []
    for i, p in enumerate(sorted(glob.glob(os.path.join(folder, "*")))):
        img = cv2.imread(p)
        if img is not None:
            out.append((f"custom_{i}_{os.path.splitext(os.path.basename(p))[0][:30]}", fit(img)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--photos", default=os.path.join(HERE, "../../samples/other"))
    ap.add_argument("--custom", default=None)
    a = ap.parse_args()
    rnd = random.Random(7)
    photos = [p for p in sorted(glob.glob(os.path.join(a.photos, "*"))) if cv2.imread(p) is not None]
    slides_dir = os.path.join(a.out, "slides")
    os.makedirs(slides_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        content = calibration() + ui_slides(tmp) + tv_slides(photos, rnd) + photo_slides(photos, rnd, 30)
    if a.custom:
        content += custom_slides(a.custom)
    # Interleave kinds so any short pass still covers every kind.
    rnd.shuffle(content)
    assert len(content) <= 250, "DICT_6X6_1000 supports 250 slides"
    manifest = []
    for k, (name, img) in enumerate(content):
        # Photographic slides as high-quality JPEG (the decoded file is the ground truth either way);
        # UI, text and calibration slides stay lossless PNG.
        photo_like = name.startswith(("photo_", "tv_"))
        fn = f"{k:03d}.jpg" if photo_like else f"{k:03d}.png"
        params = [cv2.IMWRITE_JPEG_QUALITY, 95] if photo_like else []
        cv2.imwrite(os.path.join(slides_dir, fn), add_markers(img, k), params)
        manifest.append({"id": k, "file": fn, "kind": name})
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    with open(os.path.join(a.out, "slides.js"), "w") as f:
        f.write("window.SLIDES = " + json.dumps([m["file"] for m in manifest]) + ";\n")
    shutil.copy(os.path.join(HERE, "index.html"), os.path.join(a.out, "index.html"))
    shutil.copy(os.path.join(HERE, "SHOOTING_GUIDE.md"), os.path.join(a.out, "SHOOTING_GUIDE.md"))
    credits = os.path.join(HERE, "../../samples/CREDITS.md")
    if os.path.exists(credits):
        shutil.copy(credits, os.path.join(a.out, "PHOTO_CREDITS.md"))
    print(len(manifest), "slides written to", a.out)


if __name__ == "__main__":
    main()
