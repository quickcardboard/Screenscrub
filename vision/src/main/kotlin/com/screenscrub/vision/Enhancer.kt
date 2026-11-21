package com.screenscrub.vision

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max
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
        val img = when (cls) {
            ImageClass.SCREEN -> cleanScreen(full, quad, features, meta)
            else -> cleanDocument(full, quad, meta)
        }
        val darkMode = cls == ImageClass.SCREEN && medianLuma(img) < 0.45
        val resolved = if (style == OutputStyle.AUTO) autoStyle(img, cls) else style
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

        // 3. Flatten illumination by polarity.
        flattenLuma(ch[0], dark = ImageOps.percentile8u(ch[0], 0.5) < 0.45 * 255)
        stretch(ch[0], 0.005, 0.995)

        val merged = Mat()
        Core.merge(ch, merged)
        ch.forEach { it.release() }
        img = Mat()
        Imgproc.cvtColor(merged, img, Imgproc.COLOR_YCrCb2BGR)
        merged.release()

        // 4. White balance from bright, unsaturated pixels.
        whitePatch(img)
        // Mild unsharp mask to restore text edges after the low-pass steps.
        unsharp(img, 1.2, 0.5)
        return img
    }

    // ---- DOCUMENT ----

    private fun cleanDocument(full: Mat, quad: Quad?, meta: PhotoMeta): Mat {
        val img = rectify(full, quad, meta, maxLong = 2600)
        // Divide each channel by a max-filtered background (paper white), so shadows vanish.
        val k = max(15, max(img.cols(), img.rows()) / 30)
        val ch = mutableListOf<Mat>()
        Core.split(img, ch)
        for (c in ch) {
            val bg = ImageOps.fastMorph(c, k, dilate = true)
            val bgBlur = ImageOps.fastBlur(bg, k.toDouble())
            bg.release()
            Core.divide(c, bgBlur, c, 245.0)
            bgBlur.release()
        }
        Core.merge(ch, img)
        ch.forEach { it.release() }

        // CLAHE on lightness.
        val lab = Mat()
        Imgproc.cvtColor(img, lab, Imgproc.COLOR_BGR2Lab)
        val l = mutableListOf<Mat>()
        Core.split(lab, l)
        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        clahe.apply(l[0], l[0])
        Core.merge(l, lab)
        Imgproc.cvtColor(lab, img, Imgproc.COLOR_Lab2BGR)
        l.forEach { it.release() }; lab.release()
        return img
    }

    // ---- shared ----

    private fun rectify(full: Mat, quad: Quad?, meta: PhotoMeta, maxLong: Int): Mat {
        if (quad == null) {
            val (scaled, _) = ImageOps.resizeLongEdge(full, maxLong)
            return scaled
        }
        val aspect = Geometry.snap(Geometry.aspectRatio(quad, full.cols(), full.rows(), meta.focalLength35mm))
        val (w, h) = Geometry.warpSize(quad, aspect, maxLong)
        return ImageOps.warp(full, quad, w, h)
    }

    /** Light content: divide by a max-filtered background. Dark content: subtract a min-filtered one. */
    private fun flattenLuma(y: Mat, dark: Boolean) {
        val k = max(31, max(y.cols(), y.rows()) / 8)
        val bg = ImageOps.fastMorph(y, k, dilate = !dark)
        val smooth = ImageOps.fastBlur(bg, k / 2.0)
        bg.release()
        val yf = ImageOps.toFloat(y)
        val bf = ImageOps.toFloat(smooth)
        smooth.release()
        val out = Mat()
        if (!dark) {
            Core.max(bf, Scalar(16.0), bf)
            Core.divide(yf, bf, out, 240.0)
            // Blend 80/20 with the original so gentle gradients in the content survive.
            Core.addWeighted(out, 0.8, yf, 0.2, 0.0, out)
        } else {
            val level = Core.mean(bf).`val`[0]
            Core.subtract(bf, Scalar(level), bf)
            Core.subtract(yf, bf, out)
        }
        out.convertTo(y, CvType.CV_8U)
        yf.release(); bf.release(); out.release()
    }

    private fun stretch(y: Mat, lo: Double, hi: Double) {
        val a = ImageOps.percentile8u(y, lo).toDouble()
        val b = ImageOps.percentile8u(y, hi).toDouble()
        if (b - a < 32) return
        val scale = 255.0 / (b - a)
        y.convertTo(y, CvType.CV_8U, scale, -a * scale)
    }

    private fun whitePatch(bgr: Mat) {
        val hsv = Mat()
        Imgproc.cvtColor(bgr, hsv, Imgproc.COLOR_BGR2HSV)
        val ch = mutableListOf<Mat>()
        Core.split(hsv, ch)
        val p90 = ImageOps.percentile8u(ch[2], 0.9)
        val mask = Mat()
        Core.inRange(hsv, Scalar(0.0, 0.0, max(p90, 60).toDouble()), Scalar(180.0, 70.0, 254.0), mask)
        val n = Core.countNonZero(mask)
        if (n > bgr.total() * 0.005) {
            val m = Core.mean(bgr, mask).`val`
            val ref = maxOf(m[0], m[1], m[2])
            val gains = DoubleArray(3) { (ref / max(1.0, m[it])).coerceIn(0.7, 1.5) }
            val planes = mutableListOf<Mat>()
            Core.split(bgr, planes)
            for (i in 0..2) planes[i].convertTo(planes[i], CvType.CV_8U, gains[i], 0.0)
            Core.merge(planes, bgr)
            planes.forEach { it.release() }
        }
        hsv.release(); mask.release(); ch.forEach { it.release() }
    }

    private fun unsharp(img: Mat, sigma: Double, amount: Double) {
        val blur = Mat()
        Imgproc.GaussianBlur(img, blur, Size(0.0, 0.0), sigma)
        Core.addWeighted(img, 1 + amount, blur, -amount, 0.0, img)
        blur.release()
    }

    private fun medianLuma(bgr: Mat): Double {
        val g = ImageOps.gray(bgr)
        val v = ImageOps.percentile8u(g, 0.5) / 255.0
        g.release()
        return v
    }

    /**
     * Photo for photographic content (many colours spread out), B&W for colourless documents,
     * Color otherwise.
     */
    internal fun autoStyle(bgr: Mat, cls: ImageClass): OutputStyle {
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
            cls == ImageClass.DOCUMENT && colorful < 0.01 -> OutputStyle.BW
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
