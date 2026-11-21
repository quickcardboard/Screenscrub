package com.screenscrub.vision

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Feature extraction and weighted scoring. Conservative: when unsure, OTHER.
 */
class Classifier(val config: ClassifierConfig = ClassifierConfig()) {

    private val quadDetector = QuadDetector(config.minQuadAreaFraction)
    private val moire = MoireAnalyzer(config.moireTileSize, config.moireTiles, config.moirePeakLogRatio)

    /** [full] is the full-resolution BGR image. */
    fun classify(full: Mat, meta: PhotoMeta = PhotoMeta()): Classification {
        val (small, scale) = ImageOps.resizeLongEdge(full, config.analysisLongEdge)
        val quadResult = quadDetector.detect(small)

        // Region to measure: the quad, rectified at analysis scale, or the central 80%.
        val region: Mat = if (quadResult != null) {
            val aspect = Geometry.aspectRatio(quadResult.quad, small.cols(), small.rows(), meta.focalLength35mm)
            val (w, h) = Geometry.warpSize(quadResult.quad, aspect, config.analysisLongEdge)
            ImageOps.warp(small, quadResult.quad, w, h)
        } else {
            small.submat(ImageOps.centralRect(small.cols(), small.rows(), 0.8)).clone()
        }

        // Moiré on native-resolution tiles inside the quad (or the centre).
        val fullGray = ImageOps.gray(full)
        val fullQuad = quadResult?.quad?.scaled(1.0 / scale)
        val tileRegion = if (fullQuad != null) innerRect(fullQuad, fullGray.cols(), fullGray.rows())
        else ImageOps.centralRect(fullGray.cols(), fullGray.rows(), 0.7)
        val moireResult = moire.analyze(fullGray, tileRegion)
        fullGray.release()

        val features = measure(region, quadResult, moireResult)
        region.release(); small.release()

        val screen = screenScore(features, quadResult != null)
        val doc = documentScore(features, quadResult != null)
        val cls = when {
            screen >= config.effectiveScreenThreshold && screen >= doc -> ImageClass.SCREEN
            doc >= config.effectiveDocumentThreshold -> ImageClass.DOCUMENT
            screen >= config.effectiveScreenThreshold -> ImageClass.SCREEN
            else -> ImageClass.OTHER
        }
        return Classification(cls, screen, doc, features, fullQuad)
    }

    internal fun measure(region: Mat, quad: QuadResult?, moire: MoireResult): Features {
        val gray = ImageOps.gray(region)
        // Saturation-based features are measured after neutralising the white point, so paper
        // under warm indoor light still counts as paper. The blue shift uses the raw colours.
        val balanced = neutralize(region, gray)
        val hsv = Mat()
        Imgproc.cvtColor(balanced, hsv, Imgproc.COLOR_BGR2HSV)
        balanced.release()
        val rawHsv = Mat()
        Imgproc.cvtColor(region, rawHsv, Imgproc.COLOR_BGR2HSV)
        val ch = mutableListOf<Mat>()
        Core.split(hsv, ch)
        val sat = ch[1]; val value = ch[2]
        val total = gray.total().toDouble()

        val meanSat = Core.mean(sat).`val`[0] / 255.0

        val colorfulMask = Mat()
        Core.inRange(hsv, Scalar(0.0, 90.0, 60.0), Scalar(180.0, 255.0, 255.0), colorfulMask)
        val colorful = Core.countNonZero(colorfulMask) / total

        val clipped = Mat()
        Core.inRange(region, Scalar(250.0, 250.0, 250.0), Scalar(255.0, 255.0, 255.0), clipped)
        val glare = Core.countNonZero(clipped) / total

        val paperMask = Mat()
        Core.inRange(hsv, Scalar(0.0, 0.0, 150.0), Scalar(180.0, 45.0, 255.0), paperMask)
        val paper = Core.countNonZero(paperMask) / total

        // White point: blue minus red of the bright, nearly neutral pixels (whites, not blue sky).
        val p75 = ImageOps.percentile8u(value, 0.75)
        val bright = Mat()
        Core.inRange(rawHsv, Scalar(0.0, 0.0, max(p75 - 1, 60).toDouble()), Scalar(180.0, 70.0, 255.0), bright)
        val blueShift = if (Core.countNonZero(bright) > total * 0.01) {
            val bgr = Core.mean(region, bright).`val`
            (bgr[0] - bgr[2]) / 255.0
        } else 0.0

        val hist = ImageOps.histogram8u(gray)
        val bimodality = otsuSeparation(hist)
        val medianLuma = ImageOps.percentile8u(gray, 0.5) / 255.0
        val text = textScore(gray)

        listOf(hsv, rawHsv, colorfulMask, clipped, paperMask, bright, gray).forEach { it.release() }
        ch.forEach { it.release() }

        return Features(
            quadAreaFraction = quad?.areaFraction ?: 0.0,
            insideOutsideContrast = quad?.contrast ?: 0.0,
            moirePeak = moire.maxProminence,
            moirePeakCount = moire.peakCount,
            blueShift = blueShift,
            meanSaturation = meanSat,
            colorfulFraction = colorful,
            glareFraction = glare,
            bimodality = bimodality,
            textScore = text,
            paperFraction = paper,
            medianLuma = medianLuma,
        )
    }

    fun screenScore(f: Features, hasQuad: Boolean): Double {
        val moire = ramp(f.moirePeak, config.moirePeakLogRatio, config.moirePeakLogRatio + 3)
        val blue = ramp(f.blueShift, 0.0, 0.08)
        val glare = ramp(f.glareFraction, 0.002, 0.03)
        // Saturated UI accents; a frame that is saturated almost everywhere is a scene, not a UI.
        val emissive = ramp(f.colorfulFraction, 0.02, 0.15) * (1 - ramp(f.colorfulFraction, 0.5, 0.8))
        return if (hasQuad) {
            // Either polarity: a bright screen in a dim room, or a dark-mode screen against a bright wall.
            val contrast = ramp(kotlin.math.abs(f.insideOutsideContrast), 0.05, 0.3)
            // A bright quad alone is not a screen (windows, posters): it needs one emissive cue.
            val cue = max(moire, max(blue, 0.6 * emissive))
            val s = 0.40 * moire + 0.25 * contrast + 0.20 * blue + 0.05 * glare + 0.10 * emissive
            if (cue < 0.3) s * 0.5 else s
        } else {
            // Close-up with no visible edges: only strong moiré plus an emissive cue counts.
            if (moire < 0.6 || max(blue, emissive) < 0.3) 0.0
            else 0.9 * (0.6 * moire + 0.25 * max(blue, emissive) + 0.15 * glare)
        }
    }

    fun documentScore(f: Features, hasQuad: Boolean): Double {
        val paper = ramp(f.paperFraction, 0.35, 0.75)
        val text = ramp(f.textScore, 0.15, 0.6)
        val bimodal = ramp(f.bimodality, 0.55, 0.85)
        val matte = 1.0 - ramp(f.meanSaturation, 0.12, 0.35)
        var s = 0.30 * paper + 0.30 * text + 0.20 * bimodal + 0.20 * matte
        if (!hasQuad) {
            // Page filling the frame: allowed, but needs strong paper and text evidence.
            if (paper < 0.6 || text < 0.5) return 0.0
            s *= 0.9
        }
        // Emissive evidence argues against paper.
        val moire = ramp(f.moirePeak, config.moirePeakLogRatio, config.moirePeakLogRatio + 3)
        if (moire > 0.5) s *= 0.5
        if (f.blueShift > 0.08) s *= 0.7
        return s
    }

    /** Scales channels so the brightest tenth of pixels averages neutral grey. */
    private fun neutralize(bgr: Mat, gray: Mat): Mat {
        val p90 = ImageOps.percentile8u(gray, 0.9)
        val mask = Mat()
        Imgproc.threshold(gray, mask, (p90 - 1).toDouble(), 255.0, Imgproc.THRESH_BINARY)
        val out = bgr.clone()
        if (Core.countNonZero(mask) > 50) {
            val m = Core.mean(bgr, mask).`val`
            val ref = (m[0] + m[1] + m[2]) / 3
            val planes = mutableListOf<Mat>()
            Core.split(out, planes)
            for (i in 0..2) planes[i].convertTo(planes[i], CvType.CV_8U, (ref / max(1.0, m[i])).coerceIn(0.6, 1.6), 0.0)
            Core.merge(planes, out)
            planes.forEach { it.release() }
        }
        mask.release()
        return out
    }

    /** Text-like connected components per area, normalised so ~1 means a dense page. */
    private fun textScore(gray: Mat): Double {
        val bin = Mat()
        val block = ImageOps.odd(max(15, min(gray.cols(), gray.rows()) / 30))
        Imgproc.adaptiveThreshold(gray, bin, 255.0, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, block, 12.0)
        val labels = Mat(); val stats = Mat(); val centroids = Mat()
        val n = Imgproc.connectedComponentsWithStats(bin, labels, stats, centroids, 8, CvType.CV_32S)
        val s = IntArray((stats.total() * stats.channels()).toInt())
        stats.get(0, 0, s)
        val cols = stats.cols()
        val short = min(gray.cols(), gray.rows()).toDouble()
        val minH = max(4.0, short * 0.006); val maxH = short * 0.06
        var count = 0
        for (i in 1 until n) {
            val w = s[i * cols + Imgproc.CC_STAT_WIDTH]
            val h = s[i * cols + Imgproc.CC_STAT_HEIGHT]
            val a = s[i * cols + Imgproc.CC_STAT_AREA]
            if (h < minH || h > maxH || w > 3 * maxH || w < 1) continue
            val fill = a.toDouble() / (w * h)
            if (fill < 0.12 || fill > 0.9) continue
            count++
        }
        listOf(bin, labels, stats, centroids).forEach { it.release() }
        val per10k = count / (gray.total() / 10_000.0)
        return (per10k / 20.0).coerceIn(0.0, 1.0)
    }

    private fun otsuSeparation(hist: LongArray): Double {
        val total = hist.sum().toDouble()
        if (total == 0.0) return 0.0
        var sum = 0.0
        for (i in 0 until 256) sum += i * hist[i].toDouble()
        val mean = sum / total
        var variance = 0.0
        for (i in 0 until 256) variance += hist[i] * (i - mean) * (i - mean)
        variance /= total
        if (variance < 1e-6) return 0.0
        var wB = 0.0; var sumB = 0.0; var best = 0.0
        for (t in 0 until 256) {
            wB += hist[t]; if (wB == 0.0) continue
            val wF = total - wB; if (wF == 0.0) break
            sumB += t * hist[t].toDouble()
            val mB = sumB / wB; val mF = (sum - sumB) / wF
            val between = wB / total * (wF / total) * (mB - mF) * (mB - mF)
            if (between > best) best = between
        }
        return (best / variance).coerceIn(0.0, 1.0)
    }

    private fun innerRect(q: Quad, w: Int, h: Int): Rect {
        // Axis-aligned rectangle inside the quad (shrunk bounding box of the inner corners).
        val x0 = max(q.tl.x, q.bl.x); val x1 = min(q.tr.x, q.br.x)
        val y0 = max(q.tl.y, q.tr.y); val y1 = min(q.bl.y, q.br.y)
        val mx = (x1 - x0) * 0.05; val my = (y1 - y0) * 0.05
        val r = Rect(
            (x0 + mx).roundToInt().coerceIn(0, w), (y0 + my).roundToInt().coerceIn(0, h),
            max(0.0, x1 - x0 - 2 * mx).roundToInt(), max(0.0, y1 - y0 - 2 * my).roundToInt(),
        )
        return if (r.width < 64 || r.height < 64) ImageOps.centralRect(w, h, 0.7) else r
    }

    companion object {
        fun ramp(v: Double, lo: Double, hi: Double): Double = ((v - lo) / (hi - lo)).coerceIn(0.0, 1.0)
    }
}
