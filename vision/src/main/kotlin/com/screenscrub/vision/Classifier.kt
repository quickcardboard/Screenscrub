package com.screenscrub.vision

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
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
        val (gridLuma, gridChroma) = moire.gridPeriodicity(full, tileRegion)

        val features = measure(region, quadResult, moireResult).copy(gridLuma = gridLuma, gridChroma = gridChroma)
        region.release(); small.release()

        val screen = screenScore(features, quadResult != null)
        val cls = if (screen >= config.effectiveScreenThreshold) ImageClass.SCREEN else ImageClass.OTHER
        return Classification(cls, screen, features, fullQuad)
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

        // Paper: bright relative to the region's own white level, unsaturated, and smooth
        // (low local variance), so snow, rock and foliage texture don't count.
        val white = ImageOps.percentile8u(value, 0.95)
        val paperMask = Mat()
        Core.inRange(hsv, Scalar(0.0, 0.0, max(60.0, 0.72 * white)), Scalar(180.0, 45.0, 255.0), paperMask)
        val smooth = smoothMask(gray)
        Core.bitwise_and(paperMask, smooth, paperMask)
        smooth.release()
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
            quadEdgeSupport = quad?.edgeSupport ?: 0.0,
            ringStd = quad?.ringStd ?: 0.0,
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

    /**
     * Screen evidence. With a quad (whole screen in view): contrast against the surround, a blue
     * white point or glow, a uniform bezel, moiré or pixel grid, saturated UI accents; at least
     * one screen-specific cue is required. Without a quad (close-ups): the pixel grid or moiré
     * must be clearly present, helped by a blue white point or UI colours.
     * Weights were set against the sample set (see HANDOFF.md, "Tuning log").
     */
    fun screenScore(f: Features, hasQuad: Boolean): Double {
        val moire = ramp(f.moirePeak, config.moirePeakLogRatio + 0.5, config.moirePeakLogRatio + 2.5)
        // The pixel grid is the main periodic cue; off-axis moiré alone also fires on blinds,
        // slats and rugs, so it only adds a little on top of the grid.
        val grid = ramp(f.gridLuma + f.gridChroma, 6.0, 7.0)
        val periodic = min(1.0, grid + 0.15 * moire)
        val blue = ramp(f.blueShift, 0.02, 0.08)
        // Saturated UI accents; a frame that is saturated almost everywhere is a scene, not a UI.
        val emissive = ramp(f.colorfulFraction, 0.02, 0.15) * (1 - ramp(f.colorfulFraction, 0.5, 0.8))
        return if (hasQuad) {
            val edges = ramp(f.quadEdgeSupport, 0.6, 0.9)
            // Either polarity: a bright screen in a dim room, or a dark-mode screen against a bright wall.
            val contrast = ramp(abs(f.insideOutsideContrast), 0.05, 0.3)
            val bezel = bezel(f)
            val glow = glow(f)
            var s = 0.30 * contrast + 0.30 * max(blue, glow) + 0.15 * periodic + 0.10 * emissive + 0.25 * bezel
            s *= 0.7 + 0.3 * edges
            // A bright quad alone is not a screen (windows, posters): it needs a screen-specific cue.
            if (maxOf(blue, periodic, bezel, glow) < 0.3) s *= 0.6
            // A clearly warm white point (paper under room light) argues against a screen.
            s *= 1 - 0.6 * ramp(-f.blueShift, 0.06, 0.12)
            s
        } else {
            if (periodic < 0.3) 0.0
            else 0.7 * periodic + 0.3 * max(blue, emissive)
        }
    }

    /**
     * Much brighter than its surroundings. Paper only reflects room light, so it rarely
     * outshines its surround by this much; a lit screen does.
     */
    private fun glow(f: Features) = ramp(f.insideOutsideContrast, 0.45, 0.6)

    /** Strong contrast with a uniform band right outside the quad: a monitor or laptop bezel. */
    private fun bezel(f: Features) =
        ramp(f.insideOutsideContrast, 0.3, 0.5) * (1 - ramp(f.ringStd, 0.08, 0.15))

    /** Pixels whose 7×7 neighbourhood has a standard deviation below ~6 grey levels. */
    private fun smoothMask(gray: Mat): Mat {
        val f = ImageOps.toFloat(gray)
        val mean = Mat(); val sq = Mat(); val meanSq = Mat()
        Imgproc.blur(f, mean, org.opencv.core.Size(7.0, 7.0))
        Core.multiply(f, f, sq)
        Imgproc.blur(sq, meanSq, org.opencv.core.Size(7.0, 7.0))
        Core.multiply(mean, mean, mean)
        Core.subtract(meanSq, mean, meanSq) // variance
        val mask = Mat()
        Imgproc.threshold(meanSq, mask, 36.0, 255.0, Imgproc.THRESH_BINARY_INV)
        mask.convertTo(mask, CvType.CV_8U)
        listOf(f, mean, sq, meanSq).forEach { it.release() }
        return mask
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

    /**
     * Text lines: character-sized dark components (after adaptive threshold) that have a
     * right-hand neighbour of similar height on the same baseline. Normalised so ~1 means a
     * dense page of text.
     */
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
        val boxes = ArrayList<IntArray>()
        for (i in 1 until n) {
            val x = s[i * cols + Imgproc.CC_STAT_LEFT]; val y = s[i * cols + Imgproc.CC_STAT_TOP]
            val w = s[i * cols + Imgproc.CC_STAT_WIDTH]; val h = s[i * cols + Imgproc.CC_STAT_HEIGHT]
            val a = s[i * cols + Imgproc.CC_STAT_AREA]
            if (h < minH || h > maxH || w > 3 * maxH || w < 1) continue
            val fill = a.toDouble() / (w * h)
            if (fill < 0.12 || fill > 0.9) continue
            boxes += intArrayOf(x, y, w, h)
        }
        listOf(bin, labels, stats, centroids).forEach { it.release() }
        // Sort by x so each box only scans a short window of candidates to its right.
        boxes.sortBy { it[0] }
        var chained = 0
        for (i in boxes.indices) {
            val b = boxes[i]
            val right = b[0] + b[2]; val bottom = b[1] + b[3]
            var j = i + 1
            while (j < boxes.size && boxes[j][0] <= right + 1.5 * b[3]) {
                val c = boxes[j]
                val hr = c[3].toDouble() / b[3]
                if (c[0] >= right - 0.2 * b[3] && hr in 0.6..1.6 &&
                    abs((c[1] + c[3]) - bottom) < 0.25 * b[3]
                ) { chained++; break }
                j++
            }
        }
        val per10k = chained / (gray.total() / 10_000.0)
        return (per10k / 15.0).coerceIn(0.0, 1.0)
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
