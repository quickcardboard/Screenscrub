package com.screenscrub.vision

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.roundToInt

/** Small helpers shared by the vision code. All functions allocate new Mats; callers release them. */
internal object ImageOps {

    fun resizeLongEdge(src: Mat, longEdge: Int): Pair<Mat, Double> {
        val l = max(src.cols(), src.rows())
        if (l <= longEdge) return src.clone() to 1.0
        val f = longEdge.toDouble() / l
        val dst = Mat()
        Imgproc.resize(
            src, dst,
            Size((src.cols() * f).roundToInt().toDouble(), (src.rows() * f).roundToInt().toDouble()),
            0.0, 0.0, Imgproc.INTER_AREA,
        )
        return dst to f
    }

    fun resizeTo(src: Mat, w: Int, h: Int): Mat {
        val dst = Mat()
        val interp = if (w * h < src.cols() * src.rows()) Imgproc.INTER_AREA else Imgproc.INTER_CUBIC
        Imgproc.resize(src, dst, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, interp)
        return dst
    }

    fun gray(bgr: Mat): Mat {
        if (bgr.channels() == 1) return bgr.clone()
        val g = Mat()
        Imgproc.cvtColor(bgr, g, Imgproc.COLOR_BGR2GRAY)
        return g
    }

    fun warp(src: Mat, quad: Quad, w: Int, h: Int): Mat {
        val srcPts = MatOfPoint2f(*quad.points.toTypedArray())
        val dstPts = MatOfPoint2f(
            Point(0.0, 0.0), Point(w - 1.0, 0.0), Point(w - 1.0, h - 1.0), Point(0.0, h - 1.0),
        )
        val m = Imgproc.getPerspectiveTransform(srcPts, dstPts)
        val out = Mat()
        Imgproc.warpPerspective(
            src, out, m, Size(w.toDouble(), h.toDouble()),
            Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE,
        )
        srcPts.release(); dstPts.release(); m.release()
        return out
    }

    /** Central crop keeping [fraction] of each dimension. */
    fun centralRect(cols: Int, rows: Int, fraction: Double): Rect {
        val w = (cols * fraction).roundToInt()
        val h = (rows * fraction).roundToInt()
        return Rect((cols - w) / 2, (rows - h) / 2, w, h)
    }

    /** Reads an 8-bit single channel Mat into a ByteArray (row-major, no padding). */
    fun bytes(m: Mat): ByteArray {
        val c = if (m.isContinuous) m else m.clone()
        val out = ByteArray((c.total() * c.channels()).toInt())
        c.get(0, 0, out)
        if (c !== m) c.release()
        return out
    }

    fun floats(m: Mat): FloatArray {
        val c = if (m.isContinuous) m else m.clone()
        val out = FloatArray((c.total() * c.channels()).toInt())
        c.get(0, 0, out)
        if (c !== m) c.release()
        return out
    }

    /** Value at [p] (0..1) of an 8-bit single channel image, via histogram. */
    fun percentile8u(m: Mat, p: Double, mask: Mat? = null): Int {
        val hist = histogram8u(m, mask)
        val total = hist.sum()
        if (total == 0L) return 0
        val target = (total * p).toLong()
        var acc = 0L
        for (i in 0 until 256) {
            acc += hist[i]
            if (acc > target) return i
        }
        return 255
    }

    fun histogram8u(m: Mat, mask: Mat? = null): LongArray {
        val data = bytes(m)
        val maskData = mask?.let { bytes(it) }
        val hist = LongArray(256)
        for (i in data.indices) {
            if (maskData != null && maskData[i].toInt() == 0) continue
            hist[data[i].toInt() and 0xFF]++
        }
        return hist
    }

    fun odd(v: Int): Int = if (v % 2 == 0) v + 1 else v

    fun ellipse(size: Int): Mat =
        Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(size.toDouble(), size.toDouble()))

    fun rect(size: Int): Mat =
        Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(size.toDouble(), size.toDouble()))

    /** Large-kernel blur done at reduced resolution for speed. */
    fun fastBlur(src: Mat, sigma: Double): Mat {
        val f = (8.0 / sigma).coerceAtMost(1.0)
        if (f >= 1.0) {
            val out = Mat()
            Imgproc.GaussianBlur(src, out, Size(0.0, 0.0), sigma)
            return out
        }
        val small = Mat()
        Imgproc.resize(src, small, Size(0.0, 0.0), f, f, Imgproc.INTER_AREA)
        Imgproc.GaussianBlur(small, small, Size(0.0, 0.0), sigma * f)
        val out = Mat()
        Imgproc.resize(small, out, src.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
        small.release()
        return out
    }

    /** Max (or min) filter with a large square kernel, done at reduced resolution for speed. */
    fun fastMorph(src: Mat, kernel: Int, dilate: Boolean): Mat {
        val f = (16.0 / kernel).coerceAtMost(1.0)
        val small = Mat()
        Imgproc.resize(src, small, Size(0.0, 0.0), f, f, Imgproc.INTER_AREA)
        val k = rect(max(3, odd((kernel * f).roundToInt())))
        if (dilate) Imgproc.dilate(small, small, k) else Imgproc.erode(small, small, k)
        val out = Mat()
        Imgproc.resize(small, out, src.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
        small.release(); k.release()
        return out
    }

    fun toFloat(src: Mat): Mat {
        val f = Mat()
        src.convertTo(f, CvType.CV_32F)
        return f
    }
}
