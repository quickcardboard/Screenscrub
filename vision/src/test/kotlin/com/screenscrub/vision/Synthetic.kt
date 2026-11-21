package com.screenscrub.vision

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.PI
import kotlin.math.cos
import kotlin.random.Random

/** Synthetic test photos. Real tuning needs real photos; these only check the pipeline works. */
object Synthetic {
    private var loaded = false
    fun load() {
        if (!loaded) { nu.pattern.OpenCV.loadLocally(); loaded = true }
    }

    /** A light-mode UI: header bar, sidebar, text lines. */
    fun uiContent(w: Int, h: Int, dark: Boolean = false): Mat {
        val bg = if (dark) Scalar(30.0, 30.0, 30.0) else Scalar(250.0, 250.0, 250.0)
        val fg = if (dark) Scalar(220.0, 220.0, 220.0) else Scalar(40.0, 40.0, 40.0)
        val m = Mat(h, w, CvType.CV_8UC3, bg)
        Imgproc.rectangle(m, Rect(0, 0, w, h / 12), Scalar(200.0, 120.0, 30.0), -1)
        Imgproc.rectangle(m, Rect(0, h / 12, w / 5, h), if (dark) Scalar(45.0, 45.0, 45.0) else Scalar(235.0, 235.0, 235.0), -1)
        Imgproc.rectangle(m, Rect(w / 4, h / 7, w / 6, h / 18), Scalar(60.0, 160.0, 60.0), -1)
        val rnd = Random(1)
        var y = h / 4
        while (y < h - 20) {
            val words = 6 + rnd.nextInt(6)
            val sb = StringBuilder()
            repeat(words) { sb.append("lorem ipsum ".take(3 + rnd.nextInt(8))).append(' ') }
            Imgproc.putText(m, sb.toString(), Point(w / 4.0, y.toDouble()), Imgproc.FONT_HERSHEY_SIMPLEX, h / 1400.0, fg, 2)
            y += h / 30
        }
        return m
    }

    /** A printed page: white paper, black text lines. */
    fun pageContent(w: Int, h: Int): Mat {
        val m = Mat(h, w, CvType.CV_8UC3, Scalar(245.0, 245.0, 245.0))
        val rnd = Random(2)
        Imgproc.putText(m, "INVOICE 2026", Point(w * 0.1, h * 0.08), Imgproc.FONT_HERSHEY_DUPLEX, h / 900.0, Scalar(20.0, 20.0, 20.0), 3)
        var y = (h * 0.14).toInt()
        while (y < h * 0.93) {
            val sb = StringBuilder()
            repeat(9 + rnd.nextInt(4)) { sb.append("amount due ".take(2 + rnd.nextInt(8))).append(' ') }
            Imgproc.putText(m, sb.toString(), Point(w * 0.08, y.toDouble()), Imgproc.FONT_HERSHEY_SIMPLEX, h / 2200.0, Scalar(25.0, 25.0, 25.0), 2)
            y += h / 40
        }
        return m
    }

    /** Places [content] into a photo of [pw]×[ph] at the given corners over [surround]. */
    fun place(content: Mat, pw: Int, ph: Int, corners: List<Point>, surround: Scalar): Mat {
        val photo = Mat(ph, pw, CvType.CV_8UC3, surround)
        val src = MatOfPoint2f(
            Point(0.0, 0.0), Point(content.cols() - 1.0, 0.0),
            Point(content.cols() - 1.0, content.rows() - 1.0), Point(0.0, content.rows() - 1.0),
        )
        val dst = MatOfPoint2f(*corners.toTypedArray())
        val hm = Imgproc.getPerspectiveTransform(src, dst)
        Imgproc.warpPerspective(content, photo, hm, photo.size(), Imgproc.INTER_LINEAR, Core.BORDER_TRANSPARENT)
        return photo
    }

    /** Multiplies luminance with a rotated cosine grating: a stand-in for camera moiré. */
    fun addMoire(img: Mat, period: Double, angleDeg: Double, depth: Double, region: Rect? = null) {
        val r = region ?: Rect(0, 0, img.cols(), img.rows())
        val a = angleDeg * PI / 180
        val kx = cos(a) * 2 * PI / period; val ky = kotlin.math.sin(a) * 2 * PI / period
        val f = Mat()
        img.convertTo(f, CvType.CV_32FC3)
        val data = FloatArray(img.cols() * img.rows() * 3)
        f.get(0, 0, data)
        for (y in r.y until r.y + r.height) for (x in r.x until r.x + r.width) {
            val g = (1 + depth * cos(kx * x + ky * y)).toFloat()
            val i = (y * img.cols() + x) * 3
            data[i] *= g; data[i + 1] *= g; data[i + 2] *= g
        }
        f.put(0, 0, data)
        f.convertTo(img, CvType.CV_8UC3)
        f.release()
    }

    /** Smooth illumination gradient and sensor noise, so nothing is perfectly clean. */
    fun cameraize(img: Mat, warm: Boolean = false, seed: Int = 3) {
        val w = img.cols(); val h = img.rows()
        val grad = Mat(h, w, CvType.CV_32FC3)
        val data = FloatArray(w * h * 3)
        for (y in 0 until h) for (x in 0 until w) {
            val v = (0.75 + 0.25 * (1 - (x.toDouble() / w + y.toDouble() / h) / 2)).toFloat()
            val i = (y * w + x) * 3
            data[i] = v * if (warm) 0.85f else 1f; data[i + 1] = v; data[i + 2] = v * if (warm) 1.05f else 1f
        }
        grad.put(0, 0, data)
        val f = Mat()
        img.convertTo(f, CvType.CV_32FC3)
        Core.multiply(f, grad, f)
        val noise = Mat(h, w, CvType.CV_32FC3)
        Core.randn(noise, 0.0, 4.0)
        Core.add(f, noise, f)
        f.convertTo(img, CvType.CV_8UC3)
        Imgproc.GaussianBlur(img, img, Size(3.0, 3.0), 0.6)
        listOf(grad, f, noise).forEach { it.release() }
    }

    /**
     * Light mode: screen in a dim room. Dark mode: monitor with a black bezel against a grey wall.
     */
    /**
     * Corners of a physical rectangle (width [rw], height [rh]) rotated by [yaw]/[pitch] radians,
     * seen by a pinhole camera with focal length [f] px centred on a [w]×[h] sensor.
     */
    fun projectRect(rw: Double, rh: Double, yaw: Double, pitch: Double, dist: Double, f: Double, w: Int, h: Int): List<Point> {
        fun project(x: Double, y: Double): Point {
            val X = x * kotlin.math.cos(yaw); val Z0 = -x * kotlin.math.sin(yaw)
            val Y = y * kotlin.math.cos(pitch) - Z0 * kotlin.math.sin(pitch)
            val Z = y * kotlin.math.sin(pitch) + Z0 * kotlin.math.cos(pitch) + dist
            return Point(w / 2 + f * X / Z, h / 2 + f * Y / Z)
        }
        return listOf(project(-rw / 2, -rh / 2), project(rw / 2, -rh / 2), project(rw / 2, rh / 2), project(-rw / 2, rh / 2))
    }

    const val FOCAL_PX = 3200.0

    fun screenPhoto(dark: Boolean = false, moire: Boolean = true): Mat {
        val ui = uiContent(1920, 1080, dark)
        val corners = projectRect(1.6, 0.9, 0.35, -0.15, 2.0, FOCAL_PX, 4000, 3000)
        val photo: Mat
        if (dark) {
            val bezel = Mat()
            Core.copyMakeBorder(ui, bezel, 50, 50, 50, 50, Core.BORDER_CONSTANT, Scalar(14.0, 14.0, 14.0))
            // Corners of the padded content so that the panel lands where the light-mode panel does.
            val sx = 50.0 / 1920; val sy = 50.0 / 1080
            val c = corners
            fun lerp(a: Point, b: Point, t: Double) = Point(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
            val tl = lerp(lerp(c[0], c[1], -sx), lerp(c[3], c[2], -sx), -sy)
            val tr = lerp(lerp(c[0], c[1], 1 + sx), lerp(c[3], c[2], 1 + sx), -sy)
            val br = lerp(lerp(c[0], c[1], 1 + sx), lerp(c[3], c[2], 1 + sx), 1 + sy)
            val bl = lerp(lerp(c[0], c[1], -sx), lerp(c[3], c[2], -sx), 1 + sy)
            photo = place(bezel, 4000, 3000, listOf(tl, tr, br, bl), Scalar(120.0, 125.0, 130.0))
            bezel.release()
        } else {
            photo = place(ui, 4000, 3000, corners, Scalar(25.0, 22.0, 20.0))
        }
        ui.release()
        // Screens look slightly blue.
        val planes = mutableListOf<Mat>(); Core.split(photo, planes)
        planes[0].convertTo(planes[0], -1, 1.0, 12.0); planes[2].convertTo(planes[2], -1, 0.94, 0.0)
        Core.merge(planes, photo); planes.forEach { it.release() }
        if (moire) {
            val xs = corners.map { it.x }; val ys = corners.map { it.y }
            addMoire(photo, 4.3, 27.0, 0.18, Rect(xs.min().toInt(), ys.min().toInt(), (xs.max() - xs.min()).toInt(), (ys.max() - ys.min()).toInt()))
        }
        cameraize(photo)
        return photo
    }

    fun documentPhoto(): Mat {
        val page = pageContent(2100, 2970)
        val corners = projectRect(0.21, 0.297, -0.1, 0.4, 0.42, FOCAL_PX, 4000, 3000)
        val photo = place(page, 4000, 3000, corners, Scalar(60.0, 95.0, 140.0)) // wooden desk
        page.release()
        cameraize(photo, warm = true)
        return photo
    }

    /** A "normal" photo: sky, ground, blobs, texture. No quad, no moiré. */
    fun normalPhoto(seed: Int = 5): Mat {
        val w = 4000; val h = 3000
        val m = Mat(h, w, CvType.CV_8UC3)
        val data = ByteArray(w * h * 3)
        for (y in 0 until h) for (x in 0 until w) {
            val i = (y * w + x) * 3
            if (y < h * 0.55) {
                data[i] = (230 - y * 60 / h).toByte(); data[i + 1] = (180 - y * 40 / h).toByte(); data[i + 2] = (120).toByte()
            } else {
                data[i] = 40; data[i + 1] = (110 + (x % 37)).toByte(); data[i + 2] = 60
            }
        }
        m.put(0, 0, data)
        val rnd = Random(seed)
        repeat(40) {
            Imgproc.circle(
                m, Point(rnd.nextInt(w).toDouble(), rnd.nextInt(h / 2, h).toDouble()), rnd.nextInt(40, 220),
                Scalar(rnd.nextInt(256).toDouble(), rnd.nextInt(256).toDouble(), rnd.nextInt(256).toDouble()), -1,
            )
        }
        cameraize(m, seed = seed)
        return m
    }
}
