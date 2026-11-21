package com.screenscrub.vision

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Per-class cleanup. Input is the full-resolution BGR photo; output is an encoded image.
 */
class Enhancer(private val config: ClassifierConfig = ClassifierConfig()) {

    private val moire = MoireAnalyzer(config.moireTileSize, config.moireTiles, config.moirePeakLogRatio)

    fun clean(
        full: Mat,
        cls: ImageClass,
        quad: Quad?,
        features: Features,
        style: OutputStyle,
        meta: PhotoMeta = PhotoMeta(),
    ): EncodedImage {
        val img = cleanScreen(full, quad, features, meta)
        val darkMode = medianLuma(img) < 0.45
        val resolved = if (style == OutputStyle.AUTO) autoStyle(img) else style
        val out = encode(img, resolved, darkMode)
        img.release()
        return out
    }

    // ---- SCREEN ----

    private fun cleanScreen(full: Mat, quad: Quad?, f: Features, meta: PhotoMeta): Mat {
        var img = rectify(full, quad, meta, maxLong = 2600)

        // 1. Moiré: tile-wise notch on luminance, low-pass on chroma.
        val hasMoire = f.moirePeak >= config.moirePeakLogRatio || f.moirePeakCount > 0
        val ycc = Mat()
        Imgproc.cvtColor(img, ycc, Imgproc.COLOR_BGR2YCrCb)
        img.release()
        val ch = mutableListOf<Mat>()
        Core.split(ycc, ch)
        ycc.release()
        if (hasMoire) {
            val yf = ImageOps.toFloat(ch[0])
            val cleaned = moire.removeMoire(yf, size = 256, logRatio = config.moirePeakLogRatio - 0.5)
            cleaned.convertTo(ch[0], CvType.CV_8U)
            yf.release(); cleaned.release()
            for (i in 1..2) Imgproc.GaussianBlur(ch[i], ch[i], Size(0.0, 0.0), 2.0)
        }

        // 2. Resample towards the content's effective resolution (also suppresses residual moiré).
        val long = max(ch[0].cols(), ch[0].rows())
        val target = (long * 0.8).roundToInt().coerceAtMost(2048).coerceAtLeast(minOf(long, 1024))
        val f2 = target.toDouble() / long
        val w = (ch[0].cols() * f2).roundToInt(); val h = (ch[0].rows() * f2).roundToInt()
        for (i in 0..2) {
            val r = ImageOps.resizeTo(ch[i], w, h)
            ch[i].release(); ch[i] = r
        }

        val merged = Mat()
        Core.merge(ch, merged)
        ch.forEach { it.release() }
        img = Mat()
        Imgproc.cvtColor(merged, img, Imgproc.COLOR_YCrCb2BGR)
        merged.release()

        // 3. Make it look like a native screenshot.
        val out = screenshotLook(img)
        img.release()
        return out
    }

    /**
     * Turns a photographed screen into something that looks captured, not photographed:
     *  - edge-preserving smoothing removes fine moiré, noise and dust while keeping text edges;
     *  - the UI's flat background is estimated with a morphological closing (light UIs) or
     *    opening (dark UIs) at a scale larger than text and icons, so it follows lighting falloff,
     *    glare and reflections; normalising against it removes them;
     *  - near-white snaps to white and near-black to black, as in a real screenshot;
     *  - weak chroma (colour casts, moiré fringes) is neutralised, strong UI colours are kept.
     * Photographic content (a photo or video on screen) gets a gentler version.
     */
    internal fun screenshotLook(bgr: Mat): Mat {
        val photo = isPhotographic(bgr)
        val lab = Mat()
        Imgproc.cvtColor(bgr, lab, Imgproc.COLOR_BGR2Lab)
        val ch = mutableListOf<Mat>()
        Core.split(lab, ch)
        lab.release()
        val l8 = ch[0]
        val sm = Mat()
        Imgproc.bilateralFilter(l8, sm, 9, 18.0, 5.0)
        if (!photo) { val t = Mat(); Imgproc.bilateralFilter(sm, t, 9, 18.0, 5.0); sm.release(); t.copyTo(sm); t.release() }
        val l = ImageOps.toFloat(sm)
        sm.release()

        val short = min(bgr.cols(), bgr.rows())
        val dark = ImageOps.percentile8u(l8, 0.5) < 0.45 * 255
        val k = max(15, short / (if (photo) 3 else 6))
        // Morphology at reduced scale for speed.
        val f = min(1.0, 400.0 / short)
        val small = Mat()
        Imgproc.resize(l, small, Size(0.0, 0.0), f, f, Imgproc.INTER_AREA)
        val ks = ImageOps.ellipse(max(3, ImageOps.odd((k * f).roundToInt())))
        Imgproc.morphologyEx(small, small, if (dark) Imgproc.MORPH_OPEN else Imgproc.MORPH_CLOSE, ks)
        Imgproc.GaussianBlur(small, small, Size(0.0, 0.0), max(1.0, k * f / 3))
        val bg = Mat()
        Imgproc.resize(small, bg, l.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
        small.release(); ks.release()

        val n = Mat()
        if (!dark) {
            Core.max(bg, Scalar(20.0), bg)
            Core.divide(l, bg, n) // 1.0 = background
            // Snap near-white to white (UI) or just rescale (photo).
            val (lo, hi) = if (photo) 0.0 to 1.0 else 0.03 to 0.95
            Core.subtract(n, Scalar(lo), n)
            Core.multiply(n, Scalar(255.0 / (hi - lo)), n)
        } else {
            Core.subtract(l, bg, n)
            val n8 = Mat(); n.convertTo(n8, CvType.CV_8U)
            val top = max(1.0, ImageOps.percentile8u(n8, 0.995).toDouble())
            n8.release()
            val (lo, hi) = if (photo) 0.0 to 1.0 else 0.06 to 0.97
            Core.subtract(n, Scalar(lo * top), n)
            Core.multiply(n, Scalar(255.0 / ((hi - lo) * top)), n)
        }
        n.convertTo(ch[0], CvType.CV_8U) // saturates to 0..255
        listOf(l, bg, n).forEach { it.release() }

        // Chroma: smooth fringes; neutralise weak chroma and the white/black extremes.
        val a = ImageOps.toFloat(ch[1]); val b = ImageOps.toFloat(ch[2])
        Core.subtract(a, Scalar(128.0), a); Core.subtract(b, Scalar(128.0), b)
        Imgproc.GaussianBlur(a, a, Size(0.0, 0.0), 3.0)
        Imgproc.GaussianBlur(b, b, Size(0.0, 0.0), 3.0)
        val sat = Mat(); Core.magnitude(a, b, sat)
        val keep = Mat()
        if (photo) {
            keep.create(sat.size(), CvType.CV_32F); keep.setTo(Scalar(1.0))
        } else {
            Core.subtract(sat, Scalar(6.0), keep); Core.multiply(keep, Scalar(0.1), keep)
            Imgproc.threshold(keep, keep, 1.0, 1.0, Imgproc.THRESH_TRUNC)
            Imgproc.threshold(keep, keep, 0.0, 0.0, Imgproc.THRESH_TOZERO)
        }
        // Fade chroma where L is pure white or black.
        val lf = ImageOps.toFloat(ch[0])
        val ext = Mat()
        Core.subtract(lf, Scalar(127.5), ext)
        Core.absdiff(ext, Scalar(0.0), ext)
        Core.subtract(ext, Scalar(110.0), ext)
        Core.multiply(ext, Scalar(1 / 17.5), ext)
        Imgproc.threshold(ext, ext, 1.0, 1.0, Imgproc.THRESH_TRUNC)
        Imgproc.threshold(ext, ext, 0.0, 0.0, Imgproc.THRESH_TOZERO)
        Core.subtract(Mat.ones(ext.size(), CvType.CV_32F), ext, ext)
        Core.multiply(keep, ext, keep)
        Core.multiply(a, keep, a); Core.multiply(b, keep, b)
        Core.add(a, Scalar(128.0), a); Core.add(b, Scalar(128.0), b)
        a.convertTo(ch[1], CvType.CV_8U); b.convertTo(ch[2], CvType.CV_8U)
        listOf(a, b, sat, keep, lf, ext).forEach { it.release() }

        val merged = Mat()
        Core.merge(ch, merged)
        ch.forEach { it.release() }
        val out = Mat()
        Imgproc.cvtColor(merged, out, Imgproc.COLOR_Lab2BGR)
        merged.release()
        return out
    }

    /** Many distinct colours spread out = a photo or video frame on the screen, not a UI. */
    private fun isPhotographic(bgr: Mat): Boolean {
        val (small, _) = ImageOps.resizeLongEdge(bgr, 256)
        val px = ImageOps.bytes(small)
        small.release()
        val hist = IntArray(4096)
        val n = px.size / 3
        for (i in 0 until n) {
            val b = (px[i * 3].toInt() and 0xFF) shr 4
            val g = (px[i * 3 + 1].toInt() and 0xFF) shr 4
            val r = (px[i * 3 + 2].toInt() and 0xFF) shr 4
            hist[(b shl 8) or (g shl 4) or r]++
        }
        return hist.sortedDescending().take(32).sum().toDouble() / n < 0.6
    }

    // ---- shared ----

    private val rectifier = ScreenRectifier()

    /**
     * Whole screen in view (a quad): warp it to a rectangle, which also drops the bezel.
     * Close-up (no quad): straighten from the UI's own lines, then crop bezel/case bands.
     */
    private fun rectify(full: Mat, quad: Quad?, meta: PhotoMeta, maxLong: Int): Mat {
        if (quad == null) return rectifier.rectify(full, maxLong).image
        val aspect = Geometry.snap(Geometry.aspectRatio(quad, full.cols(), full.rows(), meta.focalLength35mm))
        val (w, h) = Geometry.warpSize(quad, aspect, maxLong)
        return ImageOps.warp(full, quad, w, h)
    }





    private fun medianLuma(bgr: Mat): Double {
        val g = ImageOps.gray(bgr)
        val v = ImageOps.percentile8u(g, 0.5) / 255.0
        g.release()
        return v
    }

    /** Photo for photographic content (many colours spread out), Color otherwise. */
    internal fun autoStyle(bgr: Mat): OutputStyle {
        val (small, _) = ImageOps.resizeLongEdge(bgr, 512)
        val px = ImageOps.bytes(small)
        val hist = IntArray(4096)
        val n = px.size / 3
        for (i in 0 until n) {
            val b = (px[i * 3].toInt() and 0xFF) shr 4
            val g = (px[i * 3 + 1].toInt() and 0xFF) shr 4
            val r = (px[i * 3 + 2].toInt() and 0xFF) shr 4
            hist[(b shl 8) or (g shl 4) or r]++
        }
        val top = hist.sortedDescending().take(config.paletteColors).sum().toDouble() / n
        val hsv = Mat()
        Imgproc.cvtColor(small, hsv, Imgproc.COLOR_BGR2HSV)
        val mask = Mat()
        Core.inRange(hsv, Scalar(0.0, 80.0, 60.0), Scalar(180.0, 255.0, 255.0), mask)
        val colorful = Core.countNonZero(mask).toDouble() / n
        hsv.release(); mask.release(); small.release()
        return when {
            top < 0.75 -> OutputStyle.PHOTO
            else -> OutputStyle.COLOR
        }
    }

    private fun encode(img: Mat, style: OutputStyle, darkMode: Boolean): EncodedImage = when (style) {
        OutputStyle.PHOTO, OutputStyle.AUTO -> OutputEncoder.jpeg(img, 80)
        OutputStyle.COLOR -> OutputEncoder.palettePng(img, config.paletteColors)
        OutputStyle.GRAY -> {
            val g = ImageOps.gray(img)
            OutputEncoder.grayPng(g).also { g.release() }
        }
        OutputStyle.BW -> {
            val g = ImageOps.gray(img)
            if (darkMode) Core.bitwise_not(g, g) // keep dark text on white
            Imgproc.medianBlur(g, g, 3) // sensor noise would otherwise become speckles
            val bin = Mat()
            val block = ImageOps.odd(max(15, max(g.cols(), g.rows()) / 50))
            Imgproc.adaptiveThreshold(g, bin, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, block, 18.0)
            g.release()
            OutputEncoder.bwPng(bin).also { bin.release() }
        }
    }
}
