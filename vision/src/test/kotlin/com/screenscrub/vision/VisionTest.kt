package com.screenscrub.vision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.imgcodecs.Imgcodecs
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

class VisionTest {
    companion object {
        @BeforeClass @JvmStatic fun setup() = Synthetic.load()
    }

    private val scrub = ScreenScrub()

    @Test fun aspectRatioRecoveredFromPerspective() {
        // Project a 16:9 rectangle, rotated in 3D, with a known focal length.
        val f = 3000.0; val w = 4000; val h = 3000
        val rw = 1.6; val rh = 0.9
        val ry = 0.5; val rx = 0.3 // radians
        fun project(x: Double, y: Double): Point {
            // rotate around Y then X, push to z = 3
            var X = x * cos(ry); var Z = -x * sin(ry); var Y = y
            val Y2 = Y * cos(rx) - Z * sin(rx); val Z2 = Y * sin(rx) + Z * cos(rx)
            Y = Y2; Z = Z2 + 3.0
            return Point(w / 2 + f * X / Z, h / 2 + f * Y / Z)
        }
        val q = Quad(project(-rw / 2, -rh / 2), project(rw / 2, -rh / 2), project(rw / 2, rh / 2), project(-rw / 2, rh / 2))
        val focal35 = f / Math.hypot(w.toDouble(), h.toDouble()) * 43.27
        val withExif = Geometry.aspectRatio(q, w, h, focal35)
        val estimated = Geometry.aspectRatio(q, w, h, null)
        val naive = Geometry.edgeAspect(q)
        assertEquals(16.0 / 9, withExif, 0.03)
        assertEquals(16.0 / 9, estimated, 0.05)
        assertTrue("naive edge ratio should be worse ($naive)", abs(naive - 16.0 / 9) > abs(withExif - 16.0 / 9))
        assertEquals(16.0 / 9, Geometry.snap(withExif), 1e-9)
    }

    @Test fun pngEncodersRoundTrip() {
        val img = Synthetic.uiContent(640, 360)
        val color = OutputEncoder.palettePng(img, 32)
        val decoded = Imgcodecs.imdecode(MatOfByte(*color.bytes), Imgcodecs.IMREAD_COLOR)
        assertEquals(640, decoded.cols()); assertEquals(360, decoded.rows())
        val diff = Mat(); Core.absdiff(img, decoded, diff)
        assertTrue("palette error too high", Core.mean(diff).`val`.take(3).max() < 12)

        val gray = Mat(); org.opencv.imgproc.Imgproc.cvtColor(img, gray, org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY)
        val g = Imgcodecs.imdecode(MatOfByte(*OutputEncoder.grayPng(gray).bytes), Imgcodecs.IMREAD_GRAYSCALE)
        val d2 = Mat(); Core.absdiff(gray, g, d2)
        assertEquals(0.0, Core.norm(d2, Core.NORM_INF), 0.0)

        val bin = Mat(); org.opencv.imgproc.Imgproc.threshold(gray, bin, 128.0, 255.0, 0)
        val b = Imgcodecs.imdecode(MatOfByte(*OutputEncoder.bwPng(bin).bytes), Imgcodecs.IMREAD_GRAYSCALE)
        val d3 = Mat(); Core.absdiff(bin, b, d3)
        assertEquals(0.0, Core.norm(d3, Core.NORM_INF), 0.0)
    }

    @Test fun moireIsDetectedAndRemoved() {
        val base = Mat(1024, 1024, CvType.CV_8UC1, Scalar(180.0))
        val bgr = Mat(); org.opencv.imgproc.Imgproc.cvtColor(base, bgr, org.opencv.imgproc.Imgproc.COLOR_GRAY2BGR)
        Synthetic.addMoire(bgr, 6.1, 33.0, 0.2)
        val gray = ImageOps.gray(bgr)
        val analyzer = MoireAnalyzer()
        val before = analyzer.analyze(gray, Rect(0, 0, 1024, 1024))
        assertTrue("moiré not detected: ${before.maxProminence}", before.maxProminence > 9)

        val cleaned = analyzer.removeMoire(ImageOps.toFloat(gray))
        val c8 = Mat(); cleaned.convertTo(c8, CvType.CV_8U)
        val after = analyzer.analyze(c8, Rect(0, 0, 1024, 1024))
        assertTrue("moiré not removed: ${after.maxProminence}", after.maxProminence < before.maxProminence / 2)
        val stdBefore = org.opencv.core.MatOfDouble(); val stdAfter = org.opencv.core.MatOfDouble()
        Core.meanStdDev(gray, org.opencv.core.MatOfDouble(), stdBefore)
        Core.meanStdDev(c8, org.opencv.core.MatOfDouble(), stdAfter)
        assertTrue(stdAfter.toArray()[0] < stdBefore.toArray()[0] * 0.4)
    }

    @Test fun screenPhotoIsClassifiedAndCleaned() {
        val photo = Synthetic.screenPhoto()
        val meta = PhotoMeta(focalLength35mm = Synthetic.FOCAL_PX / Math.hypot(4000.0, 3000.0) * 43.27)
        val c = scrub.classify(photo, meta)
        println("screen: ${c.imageClass} s=${c.screenScore} d=${c.documentScore} ${c.features}")
        assertEquals(ImageClass.SCREEN, c.imageClass)
        assertNotNull(c.quad)
        val out = scrub.clean(photo, c, OutputStyle.AUTO, meta)!!
        val jpeg = OutputEncoder.jpeg(photo, 90)
        println("clean ${out.style} ${out.width}x${out.height} ${out.bytes.size / 1024} KB vs original ${jpeg.bytes.size / 1024} KB")
        assertTrue(out.width > out.height)
        assertEquals(16.0 / 9, out.width.toDouble() / out.height, 0.02)
        assertTrue("output should be smaller than the original", out.bytes.size < jpeg.bytes.size)
    }

    @Test fun darkModeScreenIsClassified() {
        val c = scrub.classify(Synthetic.screenPhoto(dark = true))
        println("dark screen: ${c.imageClass} s=${c.screenScore} d=${c.documentScore} ${c.features}")
        assertEquals(ImageClass.SCREEN, c.imageClass)
    }

    @Test fun documentPhotoIsClassifiedAndCleaned() {
        val photo = Synthetic.documentPhoto()
        val c = scrub.classify(photo)
        println("document: ${c.imageClass} s=${c.screenScore} d=${c.documentScore} ${c.features}")
        assertEquals(ImageClass.DOCUMENT, c.imageClass)
        val out = scrub.clean(photo, c, OutputStyle.AUTO)!!
        println("clean ${out.style} ${out.width}x${out.height} ${out.bytes.size / 1024} KB")
        assertEquals(OutputStyle.BW, out.style)
        assertTrue(out.height > out.width)
    }

    @Test fun normalPhotoIsOther() {
        for (seed in 1..3) {
            val c = scrub.classify(Synthetic.normalPhoto(seed))
            println("normal $seed: ${c.imageClass} s=${c.screenScore} d=${c.documentScore} ${c.features}")
            assertEquals(ImageClass.OTHER, c.imageClass)
        }
    }
}
