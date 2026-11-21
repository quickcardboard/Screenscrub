package com.screenscrub.vision

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

data class QuadResult(
    val quad: Quad,
    val areaFraction: Double,
    /** Mean luminance inside minus the surrounding ring, 0..1 scale. */
    val contrast: Double,
    val insideMean: Double,
    /**
     * Fraction of points along the sides that sit on a strong edge parallel to the side
     * (mean of the best three sides, so one occluded side is tolerated). Real screens and pages
     * have long straight edges; the hull of a random blob does not.
     */
    val edgeSupport: Double = 0.0,
    /** Grey-level standard deviation of a thin band just outside the quad (0..1). A bezel is uniform. */
    val ringStd: Double = 0.0,
)

/**
 * Finds the dominant convex quadrilateral (screen panel or page) in a small BGR image.
 *
 * Two candidate sources: Canny edges (all contours, so inner panels are found inside bezels)
 * and an Otsu bright-region mask. Rounded corners are handled by falling back to the
 * minimum-area rectangle when the contour fills it well.
 */
class QuadDetector(
    private val minAreaFraction: Double = 0.15,
    private val minEdgeSupport: Double = 0.55,
) {

    private class Gradients(val gx: FloatArray, val gy: FloatArray, val w: Int, val h: Int)

    private fun edgeSupport(g: Gradients, q: Quad): Double {
        val p = q.points
        val perSide = DoubleArray(4) { i ->
            val a = p[i]; val b = p[(i + 1) % 4]
            val len = kotlin.math.hypot(b.x - a.x, b.y - a.y)
            if (len < 10) return@DoubleArray 0.0
            val nx = -(b.y - a.y) / len; val ny = (b.x - a.x) / len
            val samples = 40
            var hit = 0
            for (k in 0 until samples) {
                val t = 0.05 + 0.9 * k / (samples - 1)
                val cx = a.x + (b.x - a.x) * t; val cy = a.y + (b.y - a.y) * t
                var ok = false
                for (d in -3..3) {
                    val x = (cx + nx * d).toInt(); val y = (cy + ny * d).toInt()
                    if (x < 1 || y < 1 || x >= g.w - 1 || y >= g.h - 1) continue
                    val i2 = y * g.w + x
                    val gxv = g.gx[i2]; val gyv = g.gy[i2]
                    val mag = kotlin.math.sqrt(gxv * gxv + gyv * gyv)
                    // Strong, and pointing across the side (an edge parallel to it).
                    if (mag > 60f && abs(gxv * nx + gyv * ny) > 0.8 * mag) { ok = true; break }
                }
                if (ok) hit++
            }
            hit.toDouble() / samples
        }
        perSide.sortDescending()
        return (perSide[0] + perSide[1] + perSide[2]) / 3
    }

    fun detect(bgr: Mat): QuadResult? {
        val scored = candidates(bgr)
        if (scored.isEmpty()) return null

        fun score(r: QuadResult) = r.areaFraction + 1.5 * abs(r.contrast)
        var best = scored.maxBy { score(it) }

        // Prefer an inner panel that stands out from its own ring (the bezel) over the bezel quad.
        val inner = scored
            .filter { r ->
                r !== best && r.areaFraction >= 0.5 * best.areaFraction &&
                    r.areaFraction < best.areaFraction * 0.97 &&
                    insetAllAround(best.quad, r.quad, 0.004 * max(bgr.cols(), bgr.rows()))
            }
            .sortedByDescending { it.areaFraction }
        if (inner.isNotEmpty()) {
            val gray = ImageOps.gray(bgr)
            val outer = best
            // The device-vs-surround contrast belongs to the bezel quad; keep it for scoring.
            inner.firstOrNull { bandContrast(gray, outer.quad, it.quad) > 0.04 }
                ?.let { best = it.copy(contrast = outer.contrast, ringStd = minOf(it.ringStd, outer.ringStd)) }
            gray.release()
        }
        return best
    }

    /** Every plausible quad with its measurements (exposed for the debug screen and tests). */
    internal fun candidates(bgr: Mat): List<QuadResult> {
        val gray = ImageOps.gray(bgr)
        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)
        val imgArea = gray.total().toDouble()
        val candidates = mutableListOf<Quad>()

        // 1. Edges.
        // Median-based thresholds, plus a low pass for faint edges (dark-mode panel inside a black bezel).
        val median = ImageOps.percentile8u(gray, 0.5).toDouble()
        val edges = Mat()
        for ((lo, hi) in listOf(max(10.0, 0.5 * median) to max(30.0, 1.2 * median), 12.0 to 36.0)) {
            Imgproc.Canny(gray, edges, lo, hi)
            Imgproc.dilate(edges, edges, ImageOps.rect(3))
            collect(edges, Imgproc.RETR_LIST, imgArea, candidates)
        }
        edges.release()

        // 2. Bright regions (Otsu), plus the dark complement for bright surrounds.
        val bin = Mat()
        Imgproc.threshold(gray, bin, 0.0, 255.0, Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU)
        Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_CLOSE, ImageOps.rect(7))
        collect(bin, Imgproc.RETR_EXTERNAL, imgArea, candidates)
        Core.bitwise_not(bin, bin)
        collect(bin, Imgproc.RETR_EXTERNAL, imgArea, candidates)
        bin.release()

        val gx = Mat(); val gy = Mat()
        Imgproc.Sobel(gray, gx, CvType.CV_32F, 1, 0)
        Imgproc.Sobel(gray, gy, CvType.CV_32F, 0, 1)
        val grad = Gradients(ImageOps.floats(gx), ImageOps.floats(gy), gray.cols(), gray.rows())
        gx.release(); gy.release()
        val scored = candidates.map { q -> measure(gray, q, imgArea).let { r -> r.copy(edgeSupport = edgeSupport(grad, q)) } }
            .filter { it.edgeSupport >= minEdgeSupport }
        gray.release()
        return scored
    }

    private fun collect(bin: Mat, mode: Int, imgArea: Double, out: MutableList<Quad>) {
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(bin, contours, hierarchy, mode, Imgproc.CHAIN_APPROX_SIMPLE)
        hierarchy.release()
        for (c in contours) {
            val area = Imgproc.contourArea(c)
            if (area < minAreaFraction * imgArea || area > 0.98 * imgArea) {
                c.release(); continue
            }
            toQuad(c, imgArea, bin.cols(), bin.rows())?.let { out += it }
            c.release()
        }
    }

    private fun toQuad(contour: MatOfPoint, imgArea: Double, w: Int, h: Int): Quad? {
        val hullIdx = MatOfInt()
        Imgproc.convexHull(contour, hullIdx)
        val pts = contour.toArray()
        val hull = MatOfPoint2f(*hullIdx.toArray().map { pts[it] }.toTypedArray())
        hullIdx.release()
        val peri = Imgproc.arcLength(hull, true)
        val hullArea = Imgproc.contourArea(hull)
        var quad: Quad? = null
        for (eps in doubleArrayOf(0.02, 0.03, 0.045, 0.06)) {
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(hull, approx, eps * peri, true)
            val a = approx.toArray()
            approx.release()
            if (a.size == 4) {
                val q = Quad.fromPoints(a.toList())
                if (isReasonable(q, imgArea, w, h)) quad = q
                break
            }
            if (a.size < 4) break
        }
        if (quad == null) {
            // Rounded corners (phones): use the min-area rectangle if the hull fills it.
            val rr = Imgproc.minAreaRect(hull)
            val rectArea = rr.size.area()
            if (rectArea > 0 && hullArea / rectArea > 0.9) {
                val corners = arrayOfNulls<Point>(4)
                rr.points(corners)
                val q = Quad.fromPoints(corners.filterNotNull())
                if (isReasonable(q, imgArea, w, h)) quad = q
            }
        }
        hull.release()
        return quad
    }

    /**
     * Rejects degenerate shapes (tiny angles, extreme aspect ratios, self-intersection) and
     * regions bounded mostly by the frame itself (sky above a horizon is not an object).
     */
    private fun isReasonable(q: Quad, imgArea: Double, w: Int, h: Int): Boolean {
        if (q.area() < minAreaFraction * imgArea) return false
        val p = q.points
        val m = 0.015 * max(w, h)
        fun onBorder(a: Point, b: Point) =
            (a.x < m && b.x < m) || (a.y < m && b.y < m) || (a.x > w - 1 - m && b.x > w - 1 - m) || (a.y > h - 1 - m && b.y > h - 1 - m)
        if ((0 until 4).count { onBorder(p[it], p[(it + 1) % 4]) } >= 2) return false
        var sign = 0
        for (i in 0 until 4) {
            val a = p[i]; val b = p[(i + 1) % 4]; val c = p[(i + 2) % 4]
            val cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
            val s = if (cross > 0) 1 else -1
            if (sign == 0) sign = s else if (s != sign) return false
            // Interior angle between 35 and 145 degrees.
            val v1x = a.x - b.x; val v1y = a.y - b.y; val v2x = c.x - b.x; val v2y = c.y - b.y
            val cos = (v1x * v2x + v1y * v2y) / (kotlin.math.hypot(v1x, v1y) * kotlin.math.hypot(v2x, v2y) + 1e-9)
            if (abs(cos) > 0.82) return false
        }
        val w = (dist(p[0], p[1]) + dist(p[3], p[2])) / 2
        val h = (dist(p[0], p[3]) + dist(p[1], p[2])) / 2
        val ar = max(w, h) / max(1.0, min(w, h))
        return ar < 6.0 // long receipts
    }

    private fun measure(gray: Mat, q: Quad, imgArea: Double): QuadResult {
        val mask = Mat.zeros(gray.size(), CvType.CV_8U)
        Imgproc.fillPoly(mask, listOf(MatOfPoint(*q.points.toTypedArray())), Scalar(255.0))
        val ringK = max(5, ImageOps.odd((max(gray.cols(), gray.rows()) * 0.04).toInt()))
        val dil = Mat()
        Imgproc.dilate(mask, dil, ImageOps.rect(ringK))
        val ring = Mat()
        Core.subtract(dil, mask, ring)
        val inside = Core.mean(gray, mask).`val`[0] / 255.0
        val ringPixels = Core.countNonZero(ring)
        val outside = if (ringPixels > 50) Core.mean(gray, ring).`val`[0] / 255.0 else inside
        // Thin band (≈1.5% of the frame) for the bezel uniformity measure.
        val thinK = max(3, ImageOps.odd((max(gray.cols(), gray.rows()) * 0.015).toInt()))
        val thin = Mat()
        Imgproc.dilate(mask, thin, ImageOps.rect(thinK))
        Core.subtract(thin, mask, thin)
        val mean = org.opencv.core.MatOfDouble(); val std = org.opencv.core.MatOfDouble()
        Core.meanStdDev(gray, mean, std, thin)
        val ringStd = if (Core.countNonZero(thin) > 50) std.toArray()[0] / 255.0 else 1.0
        listOf(mask, dil, ring, thin, mean, std).forEach { it.release() }
        return QuadResult(q, q.area() / imgArea, inside - outside, inside, ringStd = ringStd)
    }

    /** |mean inside [inner] − mean of the band between [outer] and [inner]| (panel vs bezel), 0..1. */
    private fun bandContrast(gray: Mat, outer: Quad, inner: Quad): Double {
        val outerMask = Mat.zeros(gray.size(), CvType.CV_8U)
        Imgproc.fillPoly(outerMask, listOf(MatOfPoint(*outer.points.toTypedArray())), Scalar(255.0))
        val innerMask = Mat.zeros(gray.size(), CvType.CV_8U)
        Imgproc.fillPoly(innerMask, listOf(MatOfPoint(*inner.points.toTypedArray())), Scalar(255.0))
        Core.subtract(outerMask, innerMask, outerMask)
        val band = Core.mean(gray, outerMask).`val`[0]
        val inside = Core.mean(gray, innerMask).`val`[0]
        outerMask.release(); innerMask.release()
        return abs(inside - band) / 255.0
    }

    /** True when every corner of [inner] lies inside [outer] by at least [margin] px (a bezel all round). */
    private fun insetAllAround(outer: Quad, inner: Quad, margin: Double): Boolean {
        val poly = MatOfPoint2f(*outer.points.toTypedArray())
        val ok = inner.points.all { Imgproc.pointPolygonTest(poly, it, true) >= margin }
        poly.release()
        return ok
    }

    private fun dist(a: Point, b: Point) = kotlin.math.hypot(a.x - b.x, a.y - b.y)
}
