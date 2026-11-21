# ScreenScrub — Handoff

## What this is
An Android app for the person **taking** the photo. It watches the phone's camera roll, detects photos of **another device's screen** (monitor, laptop, phone, TV) or of a **paper document**, and automatically saves a cleaned-up copy: flattened, de-moiréd, high-contrast and small, ready to send.

Received photos (WhatsApp, Messages, email) are out of scope.

**Hard constraints**
- **Fully offline.** No `INTERNET` permission.
- **No AI/ML models.** Classic computer vision (OpenCV) and heuristics only.
- **Never modify originals.** Outputs are new files in `Pictures/ScreenScrub/`.
- "Screen photo" means a *camera photo of a screen*, not an OS screenshot.

## Why it's worth building
Scanner apps and Google Photos clean up documents, but only when you ask them to, one photo at a time. Moiré removers are mostly online AI services or only work while shooting. Nobody does **automatic, offline, non-AI cleanup** of screen and document photos. The automatic part is the product, so both screens and documents are in v1.

## Decisions
| Topic | Decision |
|---|---|
| Inputs | The user's own camera photos (full-res, EXIF present). Plus a share-sheet target for anything else. |
| Classes | SCREEN, DOCUMENT, OTHER. OTHER → do nothing. |
| Output | Saved automatically to `Pictures/ScreenScrub/` (shows up as an album in gallery apps). |
| Access | In-app list with one-tap Share, a notification with a Share action, and a "Clean & share" share target. |
| Distribution | Sideload first. Play needs a Photo & Video permissions declaration for `READ_MEDIA_IMAGES` (risk of rejection). |
| Speed | The clean copy must exist a few seconds after capture, before the user shares. Measure trigger latency early. |

## Tech stack
- Kotlin, minSdk 26, compile/targetSdk 36. Jetpack Compose. Gradle Kotlin DSL + version catalog.
- OpenCV **4.9.0** on both sides of the same API:
  - Android: `org.opencv:opencv:4.9.0` (AAR).
  - Desktop/JVM tests and tuning: `org.openpnp:opencv:4.9.0-0`.
- WorkManager with a content-URI trigger on `MediaStore.Images.Media.EXTERNAL_CONTENT_URI`, chained to re-arm after each run. A `ContentObserver` while the app is in the foreground. A catch-up scan whenever the app opens.
- Room DB records **every** decision (including OTHER) so nothing is analysed twice.
- DataStore for settings.

## Modules
```
vision/   Pure Kotlin/JVM. Mat in, result out. No Android imports.
          Unit tests + the tuning harness run on a laptop.
app/      Android: UI, workers, MediaStore, Room, settings.
samples/  screen/ document/ other/ — test images (not committed if personal; use LFS or local only).
```

## Permissions
- `READ_MEDIA_IMAGES` (33+), `READ_EXTERNAL_STORAGE` (≤32), `WRITE_EXTERNAL_STORAGE` (≤28 only).
- `POST_NOTIFICATIONS` (33+).
- Android 14 "selected photos only": auto mode **cannot work**. Say so plainly and offer the full-access prompt. Manual and share modes still work.

## Pipeline

### 1. Candidates (metadata only)
- New, non-pending (`IS_PENDING=0`) images whose path is under `DCIM/` and not `Screenshots` or our own folder.
- Cursor: `GENERATION_MODIFIED` (API 30+), else `DATE_ADDED`. Only images added after auto mode was turned on.
- Skip any ID already in the DB.

### 2. Decode
- `ImageDecoder` (28+, handles HEIC and orientation), else `BitmapFactory` + EXIF rotation.
- Cap decode at ~16 MP so 50/200 MP modes don't run out of memory.
- Analysis copy: ~1024 px long edge. Moiré analysis uses **native-resolution tiles** from the full image (the downscale destroys the moiré).

### 3. Classify
Weighted scores with configurable thresholds (`ClassifierConfig`), all features logged.

**Quad:** Canny + contours + `approxPolyDP`, and an Otsu bright-region pass. Largest convex 4-gon ≥ 15% of the frame. Prefer the inner panel over the bezel.

**Screen signals:** moiré peaks (native tiles + analysis image), bright quad over darker surround, blue-shifted white point, clipped glare inside the quad.

**Screen without a quad (close-ups):** allowed only when moiré and emissive cues are both strong.

**Document signals:** large matte quad, low saturation, bimodal luminance (Otsu separation), text-like connected components.

**Conservative:** when in doubt, OTHER. Precision on normal photos matters more than recall.

### 4. Clean up
- **Warp** the quad to a rectangle. Aspect ratio from the Zhang–He single-view method using the EXIF 35 mm focal length (or a focal length estimated from the quad), then snap to 16:9, 16:10, 4:3, 3:2, 19.5:9, 20:9, A4, Letter when within 4%.
- **Screen:**
  1. Tile-based FFT notch filter on luminance (moiré frequency varies across the frame), low-pass on chroma.
  2. Resample down to the content's effective resolution (also removes moiré and shrinks the file).
  3. White balance from the brightest low-saturation pixels (white-patch), not gray-world.
  4. Flatten illumination **by polarity**: light UIs divide by a max-filtered background; dark-mode UIs subtract a min-filtered background.
  5. Inpaint glare only if the clipped area is small.
- **Document:** flatten (divide by max-filtered background), CLAHE.
- **Output style:** Auto, Color (palette of ≤32 colours, indexed PNG), B&W (adaptive threshold, 1-bit PNG), Gray, Photo (JPEG 80). Auto picks Photo for photographic content, and B&W for documents with almost no colour.

### 5. Save
- `Pictures/ScreenScrub/<name>_clean.png|jpg` via MediaStore. Copy DATE_TAKEN. Never copy GPS.
- Record source ID, output URI, class, scores, features, style, sizes and timing.

## UI (v1)
- **Home:** auto toggle, permission card, "Clean a photo" (photo picker), list of results with before/after thumbnails, class and a Share button.
- **Detail:** hold to see the original, Share, reprocess with another style, delete output, debug panel (scores, quad overlay, FFT spectrum).
- **Settings:** default style per class, sensitivity, only while charging, notifications.
- **Share target:** "Clean & share" — process, then hand the result to the share sheet. If the photo looks like OTHER, ask: send original / clean as screen / clean as document.

## Acceptance criteria
- No network permission; works in airplane mode. Originals byte-for-byte untouched.
- Sample set from **several phone models**: ≥100 screens, ≥100 documents, ≥300 normal photos including hard cases (night windows, framed art, lit signs, TVs in the background, e-ink, whiteboards, posters, fabric, blinds).
- ≥85% recall on screens and documents; **≤1% of normal photos** classified as either.
- 12 MP photo processed in < 3 s on a mid-range phone; clean copy available < 10 s after capture with the app in the background.
- Color-mode PNG of a UI screen is noticeably smaller than the original JPEG.

## Milestones
1. ✅ Skeleton: modules, OpenCV on Android and JVM, Compose home, permissions.
2. ✅ Vision v0: quad, moiré, classifier, enhancer, indexed-PNG encoder, synthetic-image tests.
3. ✅ Manual mode, share target, detail/debug screen, auto mode, notifications, settings.
4. **In progress:** tuning. A first pass on 231 web photos gives 40% / 77% recall at 1.7% false positives (see Tuning log). Needs real phone photos to finish.
5. Measure trigger latency on real devices (Pixel, Samsung, one budget phone).
6. Manual corner adjustment, quads with corners outside the frame, line-segment quad detector.
7. Later: batch-process the existing library, video frames, iOS share-extension port.

## Tuning log
**2026-11-21: first real-photo run** (231 Wikimedia Commons photos: 20 screens, 30 documents, 181 normal incl. framed paintings, posters, night windows, brick, snow). Commons served them resized to 1280–2560 px, which erases most moiré.

| | Screen | Document | Other | Recall |
|---|---|---|---|---|
| Screen | 8 | 1 | 11 | **40%** |
| Document | 0 | 23 | 7 | **77%** |
| Other | 0 | 3 | 178 | 1.7% false positives |

What changed to get here (from 0% / 63% / 18% FP): quad edge support, text-*line* feature, paper relative to the image's own white and required to be smooth, bezel uniformity, "glow" (much brighter than surround), blue white point as the screen-vs-paper tiebreak.

**Below target.** Most remaining screen misses are photos where the screen shows a picture and there is no moiré left (looks like a framed print), close-ups with no visible edges, and dark screens in bright rooms. The test set is also not representative: product shots, not phone photos of screen content. **Next step: tune on real, unresized phone photos** — the case moiré detection was designed for.

## Known risks
- Moiré is weakened by modern phone image processing and mimicked by fabric, blinds and brick. It is one vote, not the decision.
- Background trigger latency varies; some manufacturers kill background work. The foreground observer and catch-up scan cover part of this.
- PNG outputs may sort by file date instead of capture date in some gallery apps. Verify on devices.
