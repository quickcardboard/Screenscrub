package com.screenscrub.vision

import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Recovers the true aspect ratio of a photographed rectangle.
 *
 * Uses the single-view method of Zhang & He ("Whiteboard scanning and image enhancement"):
 * with the focal length known (EXIF) or estimated from the quad itself, the vanishing geometry
 * gives the width/height ratio. Falls back to mean edge lengths when the estimate is unstable.
 */
object Geometry {

    data class Ratio(val name: String, val value: Double)

    val COMMON_RATIOS = listOf(
        Ratio("16:9", 16.0 / 9), Ratio("16:10", 16.0 / 10), Ratio("4:3", 4.0 / 3),
        Ratio("3:2", 3.0 / 2), Ratio("19.5:9", 19.5 / 9), Ratio("20:9", 20.0 / 9),
        Ratio("A4", 297.0 / 210), Ratio("Letter", 11.0 / 8.5),
    )

    /**
     * Width/height ratio of the rectangle seen as [q] in an image of [imgW]×[imgH].
     * [focal35mm] is the EXIF 35 mm-equivalent focal length, if known.
     */
    fun aspectRatio(q: Quad, imgW: Int, imgH: Int, focal35mm: Double?): Double {
        val edge = edgeAspect(q)
        val u0 = imgW / 2.0; val v0 = imgH / 2.0
        val m1 = doubleArrayOf(q.tl.x - u0, q.tl.y - v0, 1.0)
        val m2 = doubleArrayOf(q.tr.x - u0, q.tr.y - v0, 1.0)
        val m3 = doubleArrayOf(q.bl.x - u0, q.bl.y - v0, 1.0)
        val m4 = doubleArrayOf(q.br.x - u0, q.br.y - v0, 1.0)
        val c14 = cross(m1, m4)
        val d2 = dot(cross(m2, m4), m3)
        val d3 = dot(cross(m3, m4), m2)
        if (abs(d2) < 1e-9 || abs(d3) < 1e-9) return edge
        val k2 = dot(c14, m3) / d2
        val k3 = dot(c14, m2) / d3
        val n2 = DoubleArray(3) { k2 * m2[it] - m1[it] }
        val n3 = DoubleArray(3) { k3 * m3[it] - m1[it] }

        val diag = hypot(imgW.toDouble(), imgH.toDouble())
        var f2: Double? = focal35mm?.takeIf { it > 5 }?.let { val f = it / 43.27 * diag; f * f }
        if (f2 == null) {
            // Nearly fronto-parallel views make this unstable; only trust plausible values.
            if (abs(n2[2]) > 1e-6 && abs(n3[2]) > 1e-6) {
                val est = -(n2[0] * n3[0] + n2[1] * n3[1]) / (n2[2] * n3[2])
                if (est > (0.4 * diag) * (0.4 * diag) && est < (3.0 * diag) * (3.0 * diag)) f2 = est
            }
        }
        if (f2 == null) return edge
        val num = (n2[0] * n2[0] + n2[1] * n2[1]) / f2 + n2[2] * n2[2]
        val den = (n3[0] * n3[0] + n3[1] * n3[1]) / f2 + n3[2] * n3[2]
        if (num <= 0 || den <= 0) return edge
        val a = sqrt(num / den)
        // Guard against wild results from bad corners.
        return if (a.isFinite() && a / edge in 0.5..2.0) a else edge
    }

    fun edgeAspect(q: Quad): Double {
        val w = (dist(q.tl, q.tr) + dist(q.bl, q.br)) / 2
        val h = (dist(q.tl, q.bl) + dist(q.tr, q.br)) / 2
        return w / max(1.0, h)
    }

    /** Snaps [aspect] (w/h) to a common ratio (either orientation) when within [tolerance]. */
    fun snap(aspect: Double, tolerance: Double = 0.04): Double {
        var best = aspect
        var bestErr = tolerance
        for (r in COMMON_RATIOS) {
            for (v in doubleArrayOf(r.value, 1 / r.value)) {
                val err = abs(aspect / v - 1)
                if (err < bestErr) { bestErr = err; best = v }
            }
        }
        return best
    }

    /** Output size for the warp: keeps the longer measured edge (capped), applies the aspect. */
    fun warpSize(q: Quad, aspect: Double, maxLongEdge: Int): Pair<Int, Int> {
        val w = max(dist(q.tl, q.tr), dist(q.bl, q.br))
        val h = max(dist(q.tl, q.bl), dist(q.tr, q.br))
        var long = max(w, h).coerceAtMost(maxLongEdge.toDouble())
        long = max(long, 64.0)
        return if (aspect >= 1) {
            long.roundToInt() to (long / aspect).roundToInt()
        } else {
            (long * aspect).roundToInt() to long.roundToInt()
        }
    }

    private fun cross(a: DoubleArray, b: DoubleArray) = doubleArrayOf(
        a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0],
    )

    private fun dot(a: DoubleArray, b: DoubleArray) = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    private fun dist(a: org.opencv.core.Point, b: org.opencv.core.Point) = hypot(a.x - b.x, a.y - b.y)
}
