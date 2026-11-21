# ScreenScrub

An offline Android app that spots the photos you take of **screens** and **paper documents** and saves a cleaned-up copy: straightened, de-moiréd, high-contrast and small, ready to send.

- No internet permission, no AI models: classic computer vision (OpenCV) only.
- Originals are never modified. Cleaned copies go to `Pictures/ScreenScrub/`.

See [HANDOFF.md](HANDOFF.md) for the design, decisions and roadmap.

## Layout
| Path | What |
|---|---|
| `vision/` | Pure Kotlin/JVM image pipeline (Mat in, result out). Unit tests and the tuning harness run on a laptop. |
| `app/` | Android app: Compose UI, WorkManager auto mode, MediaStore, Room, share target. |
| `samples/` | Local test photos in `screen/`, `document/`, `other/` (images are git-ignored). |

## Build
Requires JDK 17+ and the Android SDK (platform 36).

```sh
./gradlew :vision:test          # vision unit tests (desktop OpenCV)
./gradlew :app:assembleDebug    # app/build/outputs/apk/debug/app-debug.apk
```

Install with `adb install app/build/outputs/apk/debug/app-debug.apk`.

## Tuning
Put photos in `samples/screen`, `samples/document` and `samples/other`, then:

```sh
./gradlew :vision:tune
```

It prints every decision, a confusion matrix and the false-positive rate on normal photos, and writes all features to `samples/features.csv`. Thresholds and weights live in `ClassifierConfig` and `Classifier.screenScore` / `documentScore`.

## Using the app
- **Automatic:** turn on "Clean new photos automatically" and allow full photo access. New camera photos of screens or documents are cleaned in the background; a notification offers **Share**.
- **By hand:** "Clean a photo" opens the photo picker.
- **From any app:** share an image to **Clean & share**; the cleaned copy goes straight back to the share sheet.
- Each result: hold to compare with the original, share, open, change the output style, treat as screen/document, or delete the cleaned copy.
