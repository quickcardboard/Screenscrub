package com.screenscrub.vision

import org.opencv.core.Point

enum class ImageClass { SCREEN, OTHER }

enum class OutputStyle {
    /** Pick per image: Photo for photographic content, otherwise Color. */
    AUTO,
    /** Palette of up to [ClassifierConfig.paletteColors] colours, indexed PNG. */
    COLOR,
    /** Adaptive threshold, 1-bit PNG. */
    BW,
    /** 8-bit grayscale PNG. */
    GRAY,
    /** JPEG, for photographic content. */
    PHOTO,
}

/** Metadata that helps analysis. Everything is optional. */
data class PhotoMeta(
    /** EXIF FocalLengthIn35mmFilm. */
    val focalLength35mm: Double? = null,
)

/**
 * Four corners in image pixels, ordered top-left, top-right, bottom-right, bottom-left.
 */
data class Quad(val tl: Point, val tr: Point, val br: Point, val bl: Point) {
    val points: List<Point> get() = listOf(tl, tr, br, bl)

    fun scaled(f: Double) = Quad(
        Point(tl.x * f, tl.y * f), Point(tr.x * f, tr.y * f),
        Point(br.x * f, br.y * f), Point(bl.x * f, bl.y * f),
    )

    fun area(): Double {
        val p = points
        var a = 0.0
        for (i in p.indices) {
            val j = (i + 1) % p.size
            a += p[i].x * p[j].y - p[j].x * p[i].y
        }
        return kotlin.math.abs(a) / 2.0
    }

    companion object {
        /** Orders four arbitrary points into TL, TR, BR, BL. */
        fun fromPoints(pts: List<Point>): Quad {
            require(pts.size == 4)
            val cx = pts.sumOf { it.x } / 4
            val cy = pts.sumOf { it.y } / 4
            val sorted = pts.sortedBy { kotlin.math.atan2(it.y - cy, it.x - cx) }
            // atan2 order from -pi: TL, TR, BR, BL for a typical upright quad (y grows downward).
            val tlIdx = sorted.indices.minBy { sorted[it].x + sorted[it].y }
            val r = List(4) { sorted[(tlIdx + it) % 4] }
            return Quad(r[0], r[1], r[2], r[3])
        }
    }
}

/** Every feature the classifier computed. Values are roughly 0..1 unless noted. */
data class Features(
    val quadAreaFraction: Double = 0.0,
    /** How well image edges support the quad's sides (0..1). */
    val quadEdgeSupport: Double = 0.0,
    /** Uniformity of the band just outside the quad: grey std (0..1); low for a bezel. */
    val ringStd: Double = 0.0,
    /** Mean luminance inside the quad minus the surrounding ring (0..1 scale). */
    val insideOutsideContrast: Double = 0.0,
    /** Strongest moiré peak: ln(peak / radial median). e^3 ≈ 20× above the baseline. */
    val moirePeak: Double = 0.0,
    /** Number of significant periodic peaks found across tiles. */
    val moirePeakCount: Int = 0,
    /**
     * Screen pixel grid: median over native-resolution tiles of the strongest periodic peak
     * (ln ratio over the radial median), in luminance and in red−blue chroma. A screen's
     * pixel and subpixel lattice shows up here even in close-ups with no visible edges.
     */
    val gridLuma: Double = 0.0,
    val gridChroma: Double = 0.0,
    /** Blue minus red of bright pixels inside the region (-1..1). */
    val blueShift: Double = 0.0,
    val meanSaturation: Double = 0.0,
    /** Fraction of strongly saturated pixels. */
    val colorfulFraction: Double = 0.0,
    /** Fraction of clipped pixels inside the region. */
    val glareFraction: Double = 0.0,
    /** Otsu between-class / total variance (0..1). */
    val bimodality: Double = 0.0,
    /**
     * Text lines: density of character-like components that have a neighbour of similar height
     * on the same baseline, normalised 0..1. Random texture rarely forms such chains.
     */
    val textScore: Double = 0.0,
    /** Fraction of the region that is "paper white" (bright, unsaturated). */
    val paperFraction: Double = 0.0,
    /** Median luminance inside the region (0..1). */
    val medianLuma: Double = 0.0,
) {
    fun asMap(): Map<String, Double> = linkedMapOf(
        "quadArea" to quadAreaFraction,
        "quadEdges" to quadEdgeSupport,
        "ringStd" to ringStd,
        "contrast" to insideOutsideContrast,
        "moirePeak" to moirePeak,
        "moirePeaks" to moirePeakCount.toDouble(),
        "gridLuma" to gridLuma,
        "gridChroma" to gridChroma,
        "blueShift" to blueShift,
        "saturation" to meanSaturation,
        "colorful" to colorfulFraction,
        "glare" to glareFraction,
        "bimodality" to bimodality,
        "text" to textScore,
        "paper" to paperFraction,
        "medianLuma" to medianLuma,
    )
}

data class Classification(
    val imageClass: ImageClass,
    val screenScore: Double,
    val features: Features,
    /** Quad in full-resolution image coordinates, if one was found. */
    val quad: Quad?,
)

/**
 * All thresholds and weights. Tune with the harness; [sensitivity] shifts both decision thresholds.
 */
data class ClassifierConfig(
    val analysisLongEdge: Int = 1024,
    val minQuadAreaFraction: Double = 0.15,
    val moireTileSize: Int = 512,
    val moireTiles: Int = 4,
    /** A moiré peak must exceed the radial median magnitude by e^this (3.0 ≈ 20×). */
    val moirePeakLogRatio: Double = 3.0,
    val screenThreshold: Double = 0.45,
    /** -1 (strict) .. +1 (eager). Lowers or raises the threshold by up to 0.15. */
    val sensitivity: Double = 0.0,
    val paletteColors: Int = 32,
) {
    val effectiveScreenThreshold get() = screenThreshold - 0.15 * sensitivity.coerceIn(-1.0, 1.0)
}

class EncodedImage(
    val bytes: ByteArray,
    val mimeType: String,
    val extension: String,
    val width: Int,
    val height: Int,
    val style: OutputStyle,
)
