package com.screenscrub.vision

import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import kotlin.math.max

/**
 * Entry point for the app and the tuning harness. Pure: Mat in, result out.
 */
class ScreenScrub(val config: ClassifierConfig = ClassifierConfig()) {

    private val classifier = Classifier(config)
    private val enhancer = Enhancer(config)

    fun classify(fullBgr: Mat, meta: PhotoMeta = PhotoMeta()): Classification =
        classifier.classify(fullBgr, meta)

    /**
     * Cleans [fullBgr] as [asClass] (defaults to the classification's class).
     * Returns null for OTHER unless a class is forced.
     */
    fun clean(
        fullBgr: Mat,
        classification: Classification,
        style: OutputStyle,
        meta: PhotoMeta = PhotoMeta(),
        asClass: ImageClass? = null,
    ): EncodedImage? {
        val cls = asClass ?: classification.imageClass
        if (cls == ImageClass.OTHER) return null
        return enhancer.clean(fullBgr, cls, classification.quad, classification.features, style, meta)
    }

    /** Debug overlay: the detected quad and moiré tiles drawn on a copy of [fullBgr], downscaled. */
    fun debugOverlay(fullBgr: Mat, c: Classification, longEdge: Int = 1024): Mat {
        val (small, f) = ImageOps.resizeLongEdge(fullBgr, longEdge)
        val t = max(2, longEdge / 300)
        c.quad?.scaled(f)?.let { q ->
            Imgproc.polylines(small, listOf(MatOfPoint(*q.points.toTypedArray())), true, Scalar(0.0, 255.0, 0.0), t)
            q.points.forEach { Imgproc.circle(small, it, t * 3, Scalar(0.0, 0.0, 255.0), -1) }
        }
        Imgproc.putText(
            small, "${c.imageClass}  s=%.2f d=%.2f".format(c.screenScore, c.documentScore),
            Point(10.0, 30.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0.0, 255.0, 255.0), 2,
        )
        return small
    }

    /** FFT log-magnitude of the first moiré tile, 8-bit, DC centred. */
    fun debugSpectrum(fullBgr: Mat, c: Classification): Mat {
        val gray = ImageOps.gray(fullBgr)
        val region = c.quad?.let { q ->
            val xs = q.points.map { it.x }; val ys = q.points.map { it.y }
            Rect(xs.min().toInt(), ys.min().toInt(), (xs.max() - xs.min()).toInt(), (ys.max() - ys.min()).toInt())
        } ?: ImageOps.centralRect(gray.cols(), gray.rows(), 0.7)
        val out = MoireAnalyzer(config.moireTileSize, 1, config.moirePeakLogRatio).spectrumImage(gray, region)
        gray.release()
        return out
    }
}
