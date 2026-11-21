package com.screenscrub.vision

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * A periodic peak in an N×N spectrum. fx, fy are signed frequencies in cycles per tile.
 * [prominence] is the natural log of (peak magnitude / median magnitude at that radius).
 */
data class SpectralPeak(val fx: Int, val fy: Int, val prominence: Double)

data class MoireResult(val maxProminence: Double, val peakCount: Int, val tileRects: List<Rect>)

/**
 * Detects and removes moiré: sharp, isolated off-centre peaks in the 2D spectrum.
 *
 * Detection runs on native-resolution tiles (a downscale would erase or alias the pattern).
 * Removal is tile-based with overlap-add, because perspective makes the moiré frequency
 * vary across the frame.
 */
class MoireAnalyzer(
    private val tileSize: Int = 512,
    private val maxTiles: Int = 4,
    private val minLogRatio: Double = 3.0,
) {

    /** [gray] is the full-resolution 8-bit image; [region] limits where tiles are taken. */
    fun analyze(gray: Mat, region: Rect): MoireResult {
        val rects = tileRects(gray, region, tileSize, maxTiles)
        var best = 0.0
        var count = 0
        for (r in rects) {
            val tile = gray.submat(r)
            val peaks = detectPeaks(tile, minLogRatio, minRadius = tileSize / 16)
            tile.release()
            count += peaks.size
            peaks.maxOfOrNull { it.prominence }?.let { best = max(best, it) }
        }
        return MoireResult(best, count, rects)
    }

    /**
     * Pixel-grid periodicity on a 3×3 grid of native-resolution tiles inside [region]:
     * returns the median over tiles of the strongest peak excess in luminance and in
     * red−blue chroma. Unlike moiré detection this includes the axes (a screen held
     * straight puts its grid exactly on them) but still skips the JPEG 8×8 block grid.
     */
    fun gridPeriodicity(bgr: Mat, region: Rect): Pair<Double, Double> {
        val r = clip(region, bgr.cols(), bgr.rows())
        val n = tileSize
        if (r.width < n || r.height < n) return 0.0 to 0.0
        val lumas = ArrayList<Double>(); val chromas = ArrayList<Double>()
        for (fy in doubleArrayOf(0.25, 0.5, 0.75)) for (fx in doubleArrayOf(0.25, 0.5, 0.75)) {
            val x = ((r.x + fx * r.width - n / 2).roundToInt().coerceIn(r.x, r.x + r.width - n)) / 8 * 8
            val y = ((r.y + fy * r.height - n / 2).roundToInt().coerceIn(r.y, r.y + r.height - n)) / 8 * 8
            val tile = bgr.submat(Rect(x, y, n, n))
            val f = Mat(); tile.convertTo(f, CvType.CV_32FC3); tile.release()
            val ch = mutableListOf<Mat>(); Core.split(f, ch); f.release()
            val luma = Mat()
            Core.addWeighted(ch[2], 0.299, ch[1], 0.587, 0.0, luma)
            Core.scaleAdd(ch[0], 0.114, luma, luma)
            val chroma = Mat()
            Core.subtract(ch[2], ch[0], chroma)
            ch.forEach { it.release() }
            lumas += maxExcess(luma); chromas += maxExcess(chroma)
            luma.release(); chroma.release()
        }
        return median(lumas) to median(chromas)
    }

    private fun median(v: List<Double>) = v.sorted()[v.size / 2]

    /** Strongest spectral peak of a float tile: ln(magnitude) minus the radial median. */
    private fun maxExcess(tileF: Mat): Double {
        val n = tileF.rows()
        val mean = Core.mean(tileF).`val`[0]
        val f = Mat()
        Core.subtract(tileF, Scalar(mean), f)
        val win = Mat()
        Imgproc.createHanningWindow(win, Size(n.toDouble(), n.toDouble()), CvType.CV_32F)
        Core.multiply(f, win, f); win.release()
        val spec = Mat()
        Core.dft(f, spec, Core.DFT_COMPLEX_OUTPUT); f.release()
        val mag = magnitudes(spec, n); spec.release()
        val minR = n / 21 // ≈24 for 512: ignore content's low frequencies
        val layout = radialLayout(n, minR)
        val values = FloatArray(layout.order.size)
        for (k in layout.order.indices) values[k] = mag[layout.order[k]]
        val block = n / 8
        var best = 0.0
        for (rad in minR..n / 2) {
            val from = layout.start[rad]; val to = layout.start[rad + 1]
            if (to - from < 8) continue
            val seg = values.copyOfRange(from, to)
            java.util.Arrays.sort(seg)
            val med = seg[seg.size / 2]
            for (k in from until to) {
                val idx = layout.order[k]
                val v = mag[idx] - med
                if (v <= best) continue
                val u = idx % n; val w = idx / n
                val fx = if (u <= n / 2) u else u - n; val fy = if (w <= n / 2) w else w - n
                if (nearMultiple(fx, block) && nearMultiple(fy, block)) continue
                best = v.toDouble()
            }
        }
        return best
    }

    private fun clip(r: Rect, w: Int, h: Int): Rect {
        val x0 = r.x.coerceIn(0, w); val y0 = r.y.coerceIn(0, h)
        return Rect(x0, y0, (r.x + r.width).coerceIn(0, w) - x0, (r.y + r.height).coerceIn(0, h) - y0)
    }

    /** Log-magnitude spectrum of the first tile, shifted so DC is centred. For the debug screen. */
    fun spectrumImage(gray: Mat, region: Rect): Mat {
        val r = tileRects(gray, region, tileSize, 1).firstOrNull() ?: Rect(0, 0, min(gray.cols(), tileSize), min(gray.rows(), tileSize))
        val tile = gray.submat(r)
        val (mag, n) = logMagnitude(tile)
        tile.release()
        val img = Mat(n, n, CvType.CV_32F)
        img.put(0, 0, shifted(mag, n))
        Core.normalize(img, img, 0.0, 255.0, Core.NORM_MINMAX)
        val out = Mat()
        img.convertTo(out, CvType.CV_8U)
        img.release()
        return out
    }

    /**
     * Notch-filters a float (CV_32F) single-channel image and returns the result.
     * Tiles of [size] with 50% overlap, Hann window, weighted overlap-add. Peaks are found
     * against a local spectral baseline (box blur of the log spectrum), which is cheap enough
     * to run on every tile.
     */
    fun removeMoire(src: Mat, size: Int = 256, logRatio: Double = 2.5): Mat {
        val half = size / 2
        val padded = Mat()
        Core.copyMakeBorder(src, padded, half, half + size, half, half + size, Core.BORDER_REFLECT)
        val w = padded.cols(); val h = padded.rows()
        val acc = Mat.zeros(h, w, CvType.CV_32F)
        val norm = Mat.zeros(h, w, CvType.CV_32F)
        val window = Mat()
        Imgproc.createHanningWindow(window, Size(size.toDouble(), size.toDouble()), CvType.CV_32F)
        Core.max(window, Scalar(1e-3), window)
        val window2 = Mat()
        Core.multiply(window, window, window2)

        val tile = Mat(); val spec = Mat(); val back = Mat(); val logMag = Mat(); val base = Mat()
        val planes = mutableListOf<Mat>()
        var y = 0
        while (y + size <= h) {
            var x = 0
            while (x + size <= w) {
                val roi = Rect(x, y, size, size)
                val srcTile = padded.submat(roi)
                val mean = Core.mean(srcTile).`val`[0]
                Core.subtract(srcTile, Scalar(mean), tile)
                Core.multiply(tile, window, tile)
                Core.dft(tile, spec, Core.DFT_COMPLEX_OUTPUT)

                Core.split(spec, planes)
                Core.magnitude(planes[0], planes[1], logMag)
                planes.forEach { it.release() }; planes.clear()
                Core.add(logMag, Scalar(1.0), logMag)
                Core.log(logMag, logMag)
                Imgproc.blur(logMag, base, Size(11.0, 11.0), Point(-1.0, -1.0), Core.BORDER_REFLECT)
                val peaks = localPeaks(ImageOps.floats(logMag), ImageOps.floats(base), size, logRatio, size / 16)

                val accRoi = acc.submat(roi)
                if (peaks.isEmpty()) {
                    // tile = (src - mean)·w  →  contribution src·w² = (tile + mean·w)·w
                    Core.scaleAdd(window, mean, tile, tile)
                } else {
                    applyNotches(spec, size, peaks)
                    Core.idft(spec, tile, Core.DFT_SCALE or Core.DFT_REAL_OUTPUT)
                    Core.scaleAdd(window, mean, tile, tile)
                }
                Core.multiply(tile, window, tile)
                Core.add(accRoi, tile, accRoi)
                val normRoi = norm.submat(roi)
                Core.add(normRoi, window2, normRoi)
                srcTile.release(); accRoi.release(); normRoi.release()
                x += half
            }
            y += half
        }
        listOf(tile, spec, back, logMag, base, window, window2, padded).forEach { it.release() }
        Core.divide(acc, norm, acc)
        val out = acc.submat(Rect(half, half, src.cols(), src.rows())).clone()
        acc.release(); norm.release()
        return out
    }

    /** Peaks above a local baseline, same exclusions as [findPeaks]. */
    private fun localPeaks(mag: FloatArray, base: FloatArray, n: Int, logRatio: Double, minRadius: Int): List<SpectralPeak> {
        val block = n / 8
        val peaks = ArrayList<SpectralPeak>()
        val r2min = minRadius * minRadius
        for (fy in 0 until n / 2) for (fx in -n / 2 + 1 until n / 2) {
            if (fy == 0 && fx <= 0) continue
            if (abs(fx) <= 2 || abs(fy) <= 2) continue
            if (fx * fx + fy * fy < r2min) continue
            val i = idx(fx, fy, n)
            val excess = mag[i] - base[i]
            if (excess < logRatio) continue
            if (nearMultiple(fx, block) && nearMultiple(fy, block)) continue
            var isMax = true
            loop@ for (dy in -1..1) for (dx in -1..1) {
                if (dx == 0 && dy == 0) continue
                if (mag[idx(fx + dx, fy + dy, n)] > mag[i]) { isMax = false; break@loop }
            }
            if (isMax) peaks += SpectralPeak(fx, fy, excess.toDouble())
        }
        return peaks.sortedByDescending { it.prominence }.take(12)
    }

    // ---- internals ----

    private fun logMagnitude(tile8u: Mat): Pair<FloatArray, Int> {
        val n = min(tile8u.cols(), tile8u.rows())
        val f = Mat()
        tile8u.submat(0, n, 0, n).convertTo(f, CvType.CV_32F)
        val mean = Core.mean(f).`val`[0]
        Core.subtract(f, org.opencv.core.Scalar(mean), f)
        val win = Mat()
        Imgproc.createHanningWindow(win, Size(n.toDouble(), n.toDouble()), CvType.CV_32F)
        Core.multiply(f, win, f)
        win.release()
        val spec = Mat()
        Core.dft(f, spec, Core.DFT_COMPLEX_OUTPUT)
        f.release()
        val mag = magnitudes(spec, n)
        spec.release()
        return mag to n
    }

    private fun magnitudes(spec: Mat, n: Int): FloatArray {
        val planes = mutableListOf<Mat>()
        Core.split(spec, planes)
        val mag = Mat()
        Core.magnitude(planes[0], planes[1], mag)
        planes.forEach { it.release() }
        Core.add(mag, org.opencv.core.Scalar(1.0), mag)
        Core.log(mag, mag)
        val out = ImageOps.floats(mag)
        mag.release()
        return out
    }

    internal fun detectPeaks(tile8u: Mat, logRatio: Double, minRadius: Int): List<SpectralPeak> {
        val (mag, n) = logMagnitude(tile8u)
        return findPeaks(mag, n, logRatio, minRadius)
    }

    /**
     * Peak test against a robust radial baseline: for every integer radius, median and MAD of the
     * log-magnitude. A peak must be [logRatio] above the median (e^3 ≈ 20×; pure noise maxima stay
     * near e^1.4) and also stand out from the ring's spread (anisotropic content such as text).
     * Excludes the axes (edges, UI lines) and the JPEG 8×8 block grid.
     */
    private fun findPeaks(mag: FloatArray, n: Int, logRatio: Double, minRadius: Int): List<SpectralPeak> {
        val maxR = n / 2
        val layout = radialLayout(n, minRadius)
        val values = FloatArray(layout.order.size)
        for (k in layout.order.indices) values[k] = mag[layout.order[k]]
        val med = DoubleArray(maxR + 1) { Double.MAX_VALUE }
        val mad = DoubleArray(maxR + 1) { 1.0 }
        val tmp = FloatArray(layout.maxBucket)
        for (r in minRadius..maxR) {
            val from = layout.start[r]; val to = layout.start[r + 1]
            val len = to - from
            if (len < 8) continue
            java.util.Arrays.sort(values, from, to)
            val m = values[from + len / 2]
            for (k in 0 until len) tmp[k] = abs(values[from + k] - m)
            java.util.Arrays.sort(tmp, 0, len)
            med[r] = m.toDouble()
            mad[r] = max(1e-3, 1.4826 * tmp[len / 2])
        }
        val block = n / 8
        val peaks = ArrayList<SpectralPeak>()
        // Upper half plane only (spectrum of a real signal is symmetric).
        for (fy in 0..n / 2 - 1) for (fx in -n / 2 + 1 until n / 2) {
            if (fy == 0 && fx <= 0) continue
            if (abs(fx) <= 2 || abs(fy) <= 2) continue
            if (nearMultiple(fx, block) && nearMultiple(fy, block)) continue
            val r = hypot(fx.toDouble(), fy.toDouble()).roundToInt()
            if (r < minRadius || r > maxR) continue
            val v = mag[idx(fx, fy, n)]
            val excess = v - med[r]
            if (excess < logRatio || excess / mad[r] < 4.0) continue
            var isMax = true
            loop@ for (dy in -1..1) for (dx in -1..1) {
                if (dx == 0 && dy == 0) continue
                if (mag[idx(fx + dx, fy + dy, n)] > v) { isMax = false; break@loop }
            }
            if (isMax) peaks += SpectralPeak(fx, fy, excess)
        }
        // Keep the strongest few; a real moiré has a handful of dominant peaks.
        return peaks.sortedByDescending { it.prominence }.take(12)
    }

    private fun applyNotches(spec: Mat, n: Int, peaks: List<SpectralPeak>) {
        val data = ImageOps.floats(spec) // interleaved re, im
        // Zero within 3 bins (the Hann main lobe plus off-bin leakage), cosine taper to 6.
        val inner = 3.0
        val outer = 6.0
        val reach = 6
        for (p in peaks) {
            for (sgn in intArrayOf(1, -1)) {
                val cx = p.fx * sgn; val cy = p.fy * sgn
                for (dy in -reach..reach) for (dx in -reach..reach) {
                    val d = kotlin.math.sqrt((dx * dx + dy * dy).toDouble())
                    val g = when {
                        d <= inner -> 0.0
                        d >= outer -> 1.0
                        else -> 0.5 - 0.5 * kotlin.math.cos(Math.PI * (d - inner) / (outer - inner))
                    }
                    val i = idx(cx + dx, cy + dy, n) * 2
                    data[i] = (data[i] * g).toFloat()
                    data[i + 1] = (data[i + 1] * g).toFloat()
                }
            }
        }
        spec.put(0, 0, data)
    }

    private fun idx(fx: Int, fy: Int, n: Int): Int {
        val u = ((fx % n) + n) % n
        val v = ((fy % n) + n) % n
        return v * n + u
    }

    private fun nearMultiple(f: Int, m: Int): Boolean {
        if (m <= 0) return false
        val r = ((abs(f) % m) + m) % m
        return r <= 1 || r >= m - 1
    }

    private fun shifted(mag: FloatArray, n: Int): FloatArray {
        val out = FloatArray(n * n)
        for (v in 0 until n) for (u in 0 until n) {
            out[((v + n / 2) % n) * n + (u + n / 2) % n] = mag[v * n + u]
        }
        return out
    }

    /** Pixel indices of an N×N spectrum grouped by integer radius, cached per (n, minRadius). */
    private class RadialLayout(val order: IntArray, val start: IntArray, val maxBucket: Int)

    private val layouts = HashMap<Long, RadialLayout>()

    private fun radialLayout(n: Int, minRadius: Int): RadialLayout = synchronized(layouts) {
        layouts.getOrPut(n.toLong() shl 32 or minRadius.toLong()) {
            val maxR = n / 2
            val counts = IntArray(maxR + 2)
            val rad = IntArray(n * n) { -1 }
            for (v in 0 until n) for (u in 0 until n) {
                val fx = if (u <= n / 2) u else u - n
                val fy = if (v <= n / 2) v else v - n
                val r = hypot(fx.toDouble(), fy.toDouble()).roundToInt()
                if (r in minRadius..maxR) { rad[v * n + u] = r; counts[r]++ }
            }
            val start = IntArray(maxR + 2)
            for (r in 1..maxR + 1) start[r] = start[r - 1] + counts[r - 1]
            val fill = start.copyOf()
            val order = IntArray(start[maxR + 1])
            for (i in rad.indices) if (rad[i] >= 0) order[fill[rad[i]]++] = i
            RadialLayout(order, start, counts.max())
        }
    }

    companion object {
        /** Up to [count] non-overlapping native-resolution tiles inside [region], spread over it. */
        fun tileRects(gray: Mat, region: Rect, size: Int, count: Int): List<Rect> {
            val r = region.intersect(gray.cols(), gray.rows())
            if (r.width < size || r.height < size) {
                val s = min(r.width, r.height) / 8 * 8
                return if (s >= 64) listOf(Rect(r.x / 8 * 8, r.y / 8 * 8, s, s)) else emptyList()
            }
            val positions = listOf(0.5 to 0.5, 0.25 to 0.3, 0.75 to 0.3, 0.25 to 0.7, 0.75 to 0.7, 0.5 to 0.2, 0.5 to 0.8)
            val out = ArrayList<Rect>()
            for ((fx, fy) in positions) {
                if (out.size >= count) break
                var x = (r.x + fx * r.width - size / 2).roundToInt().coerceIn(r.x, r.x + r.width - size)
                var y = (r.y + fy * r.height - size / 2).roundToInt().coerceIn(r.y, r.y + r.height - size)
                x = x / 8 * 8; y = y / 8 * 8 // align to the JPEG block grid
                val cand = Rect(x, y, size, size)
                if (out.none { overlapFraction(it, cand) > 0.25 }) out += cand
            }
            return out
        }

        private fun overlapFraction(a: Rect, b: Rect): Double {
            val ix = max(0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))
            val iy = max(0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))
            return ix.toDouble() * iy / (a.area())
        }

        private fun Rect.intersect(w: Int, h: Int): Rect {
            val x0 = x.coerceIn(0, w); val y0 = y.coerceIn(0, h)
            val x1 = (x + width).coerceIn(0, w); val y1 = (y + height).coerceIn(0, h)
            return Rect(x0, y0, x1 - x0, y1 - y0)
        }

    }
}
