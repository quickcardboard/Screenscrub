package com.screenscrub.vision

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Straightens a close-up of a screen when the screen's outline is not visible, then crops away
 * whatever is not screen (bezel, case, table) along the borders.
 *
 * Straightening: line segments (LSD) are split into a horizontal and a vertical family; each
 * family's vanishing point is found with RANSAC. A homography sends both vanishing points to
 * infinity (removing perspective) and maps their directions to the image axes (removing
 * rotation and skew). UI text and layout supply plenty of both kinds of line.
 */
class ScreenRectifier(private val analysisLongEdge: Int = 1600) {

    /** Result in full-resolution pixels. */
    class Rectified(val image: Mat, val straightened: Boolean, val cropped: IntArray)

    fun rectify(full: Mat, maxLongEdge: Int): Rectified {
        val h = homography(full)
        var img: Mat
        var straightened = false
        if (h != null) {
            val warped = warpWhole(full, h, maxLongEdge)
            if (warped != null) {
                img = warped; straightened = true
            } else img = ImageOps.resizeLongEdge(full, maxLongEdge).first
            h.release()
        } else {
            img = ImageOps.resizeLongEdge(full, maxLongEdge).first
        }
        val cut = cropBorders(img)
        val cropped = img.submat(Rect(cut[2], cut[0], img.cols() - cut[2] - cut[3], img.rows() - cut[0] - cut[1])).clone()
        img.release()
        return Rectified(cropped, straightened, cut)
    }

    // ---- straightening ----

    private class Seg(val x1: Double, val y1: Double, val x2: Double, val y2: Double) {
        val len = hypot(x2 - x1, y2 - y1)
        val line: DoubleArray = cross(doubleArrayOf(x1, y1, 1.0), doubleArrayOf(x2, y2, 1.0))
        val mx = (x1 + x2) / 2; val my = (y1 + y2) / 2
        val dx = (x2 - x1) / len; val dy = (y2 - y1) / len
    }

    /** Homography in full-resolution coordinates, or null if the lines don't support one. */
    internal fun homography(full: Mat): Mat? {
        val (small, s) = ImageOps.resizeLongEdge(full, analysisLongEdge)
        val gray = ImageOps.gray(small)
        small.release()
        val lsd = Imgproc.createLineSegmentDetector(Imgproc.LSD_REFINE_STD)
        val lines = Mat()
        lsd.detect(gray, lines)
        val cx = gray.cols() / 2.0; val cy = gray.rows() / 2.0
        val longEdge = max(gray.cols(), gray.rows()).toDouble()
        gray.release()
        val data = FloatArray((lines.total() * lines.channels()).toInt())
        if (data.isNotEmpty()) lines.get(0, 0, data)
        lines.release()

        val hor = ArrayList<Seg>(); val ver = ArrayList<Seg>()
        for (i in 0 until data.size / 4) {
            val sg = Seg(data[i * 4] - cx, data[i * 4 + 1] - cy, data[i * 4 + 2] - cx, data[i * 4 + 3] - cy)
            val ang = (Math.toDegrees(atan2(sg.y2 - sg.y1, sg.x2 - sg.x1)) + 180) % 180
            // Short segments count too: text strokes and baselines are plentiful in UIs.
            if ((ang < 40 || ang > 140) && sg.len > longEdge * 0.015) hor += sg
            else if (ang in 50.0..130.0 && sg.len > longEdge * 0.012) ver += sg
        }
        if (DEBUG) println("rectifier: segs=${data.size / 4} hor=${hor.size} ver=${ver.size} first=${data.take(8)}")
        if (hor.size < 6) return null
        val vh = vanishingPoint(hor)
        val vv = if (ver.size >= 6) vanishingPoint(ver) else null
        if (DEBUG) println("rectifier: hor=${hor.size} ver=${ver.size} vh=${vh?.toList()} vv=${vv?.toList()}")
        if (vh == null) return null

        val hm: Array<DoubleArray> = if (vv != null) {
            // Projective part: vanishing line to infinity.
            var l = cross(vh, vv)
            if (abs(l[2]) < 1e-12) return null
            l = DoubleArray(3) { l[it] / l[2] }
            val h1 = arrayOf(doubleArrayOf(1.0, 0.0, 0.0), doubleArrayOf(0.0, 1.0, 0.0), l)
            val dh = normalize2(mul(h1, vh)); val dv = normalize2(mul(h1, vv))
            if (dh[0] < 0) { dh[0] = -dh[0]; dh[1] = -dh[1] }
            if (dv[1] < 0) { dv[0] = -dv[0]; dv[1] = -dv[1] }
            val det = dh[0] * dv[1] - dv[0] * dh[1]
            if (DEBUG) println("  dh=${dh.toList()} dv=${dv.toList()} det=$det l=${l.toList()}")
            if (abs(det) < 0.3) return null // the two families are nearly parallel: unreliable
            // Affine part: map the two directions to the x and y axes.
            val a = arrayOf(
                doubleArrayOf(dv[1] / det, -dv[0] / det, 0.0),
                doubleArrayOf(-dh[1] / det, dh[0] / det, 0.0),
                doubleArrayOf(0.0, 0.0, 1.0),
            )
            matmul(a, h1)
        } else {
            // Only horizontals: undo the rotation, leave perspective alone.
            val d = if (abs(vh[2]) < 1e-9) normalize2(vh) else {
                normalize2(doubleArrayOf(vh[0] / vh[2], vh[1] / vh[2]))
            }
            val ang = atan2(d[1], d[0]).let { if (it > Math.PI / 2) it - Math.PI else if (it < -Math.PI / 2) it + Math.PI else it }
            if (abs(Math.toDegrees(ang)) < 0.3) return null
            val c = cos(-ang); val sn = kotlin.math.sin(-ang)
            arrayOf(doubleArrayOf(c, -sn, 0.0), doubleArrayOf(sn, c, 0.0), doubleArrayOf(0.0, 0.0, 1.0))
        }
        // Full-res pixel → centred small coords → rectified (still in small-image scale).
        val toSmall = arrayOf(doubleArrayOf(s, 0.0, -cx), doubleArrayOf(0.0, s, -cy), doubleArrayOf(0.0, 0.0, 1.0))
        val back = arrayOf(doubleArrayOf(1 / s, 0.0, 0.0), doubleArrayOf(0.0, 1 / s, 0.0), doubleArrayOf(0.0, 0.0, 1.0))
        val total = matmul(back, matmul(hm, toSmall))
        val m = Mat(3, 3, CvType.CV_64F)
        for (r in 0..2) for (k in 0..2) m.put(r, k, total[r][k])
        return m
    }

    /** RANSAC over segment pairs, then a weighted least-squares refinement on the inliers. */
    private fun vanishingPoint(segs: List<Seg>): DoubleArray? {
        val rnd = java.util.Random(7)
        val totalLen = segs.sumOf { it.len }
        fun pick(): Seg {
            var r = rnd.nextDouble() * totalLen
            for (sg in segs) { r -= sg.len; if (r <= 0) return sg }
            return segs.last()
        }
        val cosTol = cos(Math.toRadians(2.0))
        var bestScore = -1.0
        var bestInliers: List<Seg> = emptyList()
        repeat(300) {
            val a = pick(); val b = pick()
            if (a === b) return@repeat
            val v = cross(a.line, b.line)
            if (abs(v[0]) + abs(v[1]) + abs(v[2]) < 1e-12) return@repeat
            val inl = segs.filter { consistent(it, v, cosTol) }
            val score = inl.sumOf { it.len }
            if (score > bestScore) { bestScore = score; bestInliers = inl }
        }
        if (DEBUG) println("  vp: n=${segs.size} inliers=${bestInliers.size} frac=${bestScore / totalLen}")
        if (bestInliers.size < 4 || bestScore < 0.25 * totalLen) return null
        // Smallest eigenvector of Σ w·l·lᵀ (lines normalised so the distance metric is pixels).
        val m = Array(3) { DoubleArray(3) }
        for (sg in bestInliers) {
            val n = hypot(sg.line[0], sg.line[1])
            val l = DoubleArray(3) { sg.line[it] / n }
            for (i in 0..2) for (j in 0..2) m[i][j] += sg.len * l[i] * l[j]
        }
        return smallestEigenvector(m)
    }

    private fun consistent(sg: Seg, v: DoubleArray, cosTol: Double): Boolean {
        val (tx, ty) = if (abs(v[2]) > 1e-9) {
            val x = v[0] / v[2] - sg.mx; val y = v[1] / v[2] - sg.my
            val n = hypot(x, y); if (n < 1e-9) return true
            x / n to y / n
        } else {
            val n = hypot(v[0], v[1]); v[0] / n to v[1] / n
        }
        return abs(tx * sg.dx + ty * sg.dy) > cosTol
    }

    /** Warps the whole frame, translating it into view, then cuts to the fully valid area. */
    private fun warpWhole(full: Mat, h: Mat, maxLongEdge: Int): Mat? {
        val w = full.cols().toDouble(); val hh = full.rows().toDouble()
        val hm = Array(3) { r -> DoubleArray(3) { c -> h.get(r, c)[0] } }
        val corners = listOf(doubleArrayOf(0.0, 0.0, 1.0), doubleArrayOf(w, 0.0, 1.0), doubleArrayOf(w, hh, 1.0), doubleArrayOf(0.0, hh, 1.0))
            .map { mul(hm, it) }
        if (corners.any { it[2] <= 0 }) return null
        val xs = corners.map { it[0] / it[2] }; val ys = corners.map { it[1] / it[2] }
        val x0 = xs.min(); val y0 = ys.min(); val bw = xs.max() - x0; val bh = ys.max() - y0
        if (bw * bh > 6 * w * hh || bw < 32 || bh < 32) return null
        val scale = min(1.0, maxLongEdge / max(bw, bh))
        val adj = arrayOf(doubleArrayOf(scale, 0.0, -x0 * scale), doubleArrayOf(0.0, scale, -y0 * scale), doubleArrayOf(0.0, 0.0, 1.0))
        val tot = matmul(adj, hm)
        val m = Mat(3, 3, CvType.CV_64F)
        for (r in 0..2) for (c in 0..2) m.put(r, c, tot[r][c])
        val size = Size((bw * scale).roundToInt().toDouble(), (bh * scale).roundToInt().toDouble())
        val out = Mat()
        Imgproc.warpPerspective(full, out, m, size, Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, Scalar(0.0, 0.0, 0.0))
        val valid = Mat()
        Imgproc.warpPerspective(Mat(full.size(), CvType.CV_8U, Scalar(255.0)), valid, m, size, Imgproc.INTER_NEAREST)
        m.release()
        val r = innerValidRect(valid)
        valid.release()
        if (r.width < 32 || r.height < 32) { out.release(); return null }
        val cropped = out.submat(r).clone()
        out.release()
        return cropped
    }

    /** Shrinks the frame side by side until it contains only valid (warped-in) pixels. */
    private fun innerValidRect(valid: Mat): Rect {
        val w = valid.cols(); val h = valid.rows()
        val v = ImageOps.bytes(valid)
        var t = 0; var b = h; var l = 0; var r = w
        fun rowFrac(y: Int): Double { var c = 0; for (x in l until r) if (v[y * w + x].toInt() != 0) c++; return c.toDouble() / (r - l) }
        fun colFrac(x: Int): Double { var c = 0; for (y in t until b) if (v[y * w + x].toInt() != 0) c++; return c.toDouble() / (b - t) }
        val step = max(1, (0.004 * max(w, h)).toInt())
        repeat(400) {
            if (b - t < 32 || r - l < 32) return Rect(l, t, max(0, r - l), max(0, b - t))
            val fr = doubleArrayOf(rowFrac(t), rowFrac(b - 1), colFrac(l), colFrac(r - 1))
            val worst = fr.indices.minBy { fr[it] }
            if (fr[worst] >= 0.999) return Rect(l, t, r - l, b - t)
            when (worst) { 0 -> t += step; 1 -> b -= step; 2 -> l += step; else -> r -= step }
        }
        return Rect(l, t, r - l, b - t)
    }

    // ---- border cropping ----

    /**
     * Crops bezel, case or table along each side: a band from the border to a strong straight
     * edge, where the band is uniform and differs in colour from what lies just inside.
     * Returns rows/cols to cut as [top, bottom, left, right].
     */
    internal fun cropBorders(bgr: Mat): IntArray {
        val lab = Mat()
        Imgproc.cvtColor(bgr, lab, Imgproc.COLOR_BGR2Lab)
        val w = bgr.cols(); val h = bgr.rows()
        val px = ImageOps.bytes(lab)
        lab.release()
        // Accessors in each orientation: (i = distance from the border, j = position along it).
        fun get(side: Int, i: Int, j: Int, ch: Int): Int = when (side) {
            0 -> px[(i * w + j) * 3 + ch]
            1 -> px[((h - 1 - i) * w + j) * 3 + ch]
            2 -> px[(j * w + i) * 3 + ch]
            else -> px[(j * w + (w - 1 - i)) * 3 + ch]
        }.toInt() and 0xFF
        val out = IntArray(4)
        for (side in 0..3) {
            val depth = if (side < 2) h else w
            val along = if (side < 2) w else h
            out[side] = sideCrop(depth, along) { i, j, ch -> get(side, i, j, ch) }
        }
        // Never crop away more than 70% in either direction.
        if (out[0] + out[1] > 0.7 * h) { out[0] = 0; out[1] = 0 }
        if (out[2] + out[3] > 0.7 * w) { out[2] = 0; out[3] = 0 }
        return out
    }

    private fun sideCrop(depth: Int, along: Int, at: (Int, Int, Int) -> Int): Int {
        val lim = (depth * 0.4).toInt()
        if (lim < 20) return 0
        val stride = max(1, along / 400)
        val cols = (0 until along step stride).toList()
        // Per-line luminance mean and std.
        val rowMean = DoubleArray(lim); val rowStd = DoubleArray(lim)
        for (i in 0 until lim) {
            var s = 0.0; var s2 = 0.0
            for (j in cols) { val v = at(i, j, 0).toDouble(); s += v; s2 += v * v }
            val m = s / cols.size
            rowMean[i] = m; rowStd[i] = sqrt(max(0.0, s2 / cols.size - m * m))
        }
        val k = 6
        val inside = max(20, depth / 20)
        var best = 0
        for (y in k until lim - k) {
            // Edge coverage: fraction of positions with a clear step between the 6 lines above and below.
            var hit = 0
            for (j in cols) {
                var a = 0.0; var b = 0.0
                for (d in 1..k) { a += at(y - d, j, 0); b += at(y + d - 1, j, 0) }
                if (abs(a - b) / k > 18) hit++
            }
            if (hit < 0.5 * cols.size) continue
            // The band must be uniform throughout (80th percentile of line std).
            val stds = rowStd.copyOfRange(0, y).sorted()
            if (stds[(stds.size * 0.8).toInt().coerceAtMost(stds.size - 1)] > 24) continue
            // ...and differ in colour from what lies just inside.
            val bandLab = medianLab(at, 0, y, cols)
            val inLab = medianLab(at, y, min(depth, y + inside), cols)
            val dist = sqrt((0..2).sumOf { (bandLab[it] - inLab[it]).let { d -> d * d } })
            if (dist < 20) continue
            best = y
        }
        return best
    }

    private fun medianLab(at: (Int, Int, Int) -> Int, from: Int, to: Int, cols: List<Int>): DoubleArray {
        val out = DoubleArray(3)
        val rowStep = max(1, (to - from) / 40)
        for (ch in 0..2) {
            val hist = IntArray(256); var n = 0
            var i = from
            while (i < to) { for (j in cols) { hist[at(i, j, ch)]++; n++ }; i += rowStep }
            var acc = 0
            for (v in 0..255) { acc += hist[v]; if (acc * 2 >= n) { out[ch] = v.toDouble(); break } }
        }
        return out
    }

    companion object {
        var DEBUG = false

        fun cross(a: DoubleArray, b: DoubleArray) = doubleArrayOf(
            a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0],
        )

        private fun mul(m: Array<DoubleArray>, v: DoubleArray) = DoubleArray(3) { r -> m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2] }

        private fun matmul(a: Array<DoubleArray>, b: Array<DoubleArray>) =
            Array(3) { r -> DoubleArray(3) { c -> a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c] } }

        private fun normalize2(v: DoubleArray): DoubleArray {
            val n = hypot(v[0], v[1]); return doubleArrayOf(v[0] / n, v[1] / n)
        }

        /** Smallest eigenvector of a symmetric 3×3 matrix (via OpenCV). */
        private fun smallestEigenvector(m: Array<DoubleArray>): DoubleArray {
            val mat = Mat(3, 3, CvType.CV_64F)
            for (r in 0..2) for (c in 0..2) mat.put(r, c, m[r][c])
            val vals = Mat(); val vecs = Mat()
            Core.eigen(mat, vals, vecs)
            val out = DoubleArray(3) { vecs.get(2, it)[0] } // rows sorted by descending eigenvalue
            listOf(mat, vals, vecs).forEach { it.release() }
            return out
        }
    }
}
