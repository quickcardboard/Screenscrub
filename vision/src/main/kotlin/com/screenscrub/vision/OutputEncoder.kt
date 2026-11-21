package com.screenscrub.vision

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfInt
import org.opencv.core.TermCriteria
import org.opencv.imgcodecs.Imgcodecs
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.util.zip.CRC32
import java.util.zip.Deflater
import kotlin.math.min

/**
 * Encoders. Indexed and grayscale PNGs are written by hand: OpenCV and Android only write
 * truecolour PNGs, which are several times larger for palette content.
 */
object OutputEncoder {

    fun jpeg(bgr: Mat, quality: Int = 80): EncodedImage {
        val buf = MatOfByte()
        Imgcodecs.imencode(".jpg", bgr, buf, MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, quality))
        val bytes = buf.toArray()
        buf.release()
        return EncodedImage(bytes, "image/jpeg", "jpg", bgr.cols(), bgr.rows(), OutputStyle.PHOTO)
    }

    /** 8-bit grayscale PNG. */
    fun grayPng(gray: Mat): EncodedImage {
        val w = gray.cols(); val h = gray.rows()
        val px = ImageOps.bytes(gray)
        val bytes = png(w, h, bitDepth = 8, colorType = 0, palette = null, rows = { y, out ->
            // Filter "Up" for every row: cheap and effective for scanned content.
            out[0] = 2
            val o = y * w
            for (x in 0 until w) {
                val above = if (y == 0) 0 else px[o - w + x].toInt() and 0xFF
                out[1 + x] = ((px[o + x].toInt() and 0xFF) - above).toByte()
            }
        }, rowBytes = w)
        return EncodedImage(bytes, "image/png", "png", w, h, OutputStyle.GRAY)
    }

    /** 1-bit PNG from a binary image (0 = black, anything else = white). */
    fun bwPng(bin: Mat): EncodedImage {
        val w = bin.cols(); val h = bin.rows()
        val px = ImageOps.bytes(bin)
        val rowBytes = (w + 7) / 8
        val bytes = png(w, h, bitDepth = 1, colorType = 0, palette = null, rows = { y, out ->
            out[0] = 0
            java.util.Arrays.fill(out, 1, out.size, 0)
            val o = y * w
            for (x in 0 until w) {
                if (px[o + x].toInt() != 0) {
                    val i = 1 + (x shr 3)
                    out[i] = (out[i].toInt() or (0x80 ushr (x and 7))).toByte()
                }
            }
        }, rowBytes = rowBytes)
        return EncodedImage(bytes, "image/png", "png", w, h, OutputStyle.BW)
    }

    /** Palette-quantised colour PNG with at most [colors] entries (≤ 256). */
    fun palettePng(bgr: Mat, colors: Int): EncodedImage {
        val (palette, indices) = quantize(bgr, colors)
        val w = bgr.cols(); val h = bgr.rows()
        val n = palette.size / 3
        val depth = when { n <= 2 -> 1; n <= 4 -> 2; n <= 16 -> 4; else -> 8 }
        val perByte = 8 / depth
        val rowBytes = (w + perByte - 1) / perByte
        val bytes = png(w, h, bitDepth = depth, colorType = 3, palette = palette, rows = { y, out ->
            out[0] = 0
            java.util.Arrays.fill(out, 1, out.size, 0)
            val o = y * w
            for (x in 0 until w) {
                val v = indices[o + x].toInt() and 0xFF
                val i = 1 + x / perByte
                val shift = 8 - depth * (x % perByte + 1)
                out[i] = (out[i].toInt() or (v shl shift)).toByte()
            }
        }, rowBytes = rowBytes)
        return EncodedImage(bytes, "image/png", "png", w, h, OutputStyle.COLOR)
    }

    /**
     * k-means on a pixel sample, then a 15-bit lookup table maps every pixel to its nearest
     * centre. Returns (RGB palette bytes, one index byte per pixel).
     */
    fun quantize(bgr: Mat, colors: Int): Pair<ByteArray, ByteArray> {
        val k = colors.coerceIn(2, 256)
        val px = ImageOps.bytes(bgr)
        val count = px.size / 3
        val step = maxOf(1, count / 40_000)
        val sampleN = (count + step - 1) / step
        val samples = Mat(sampleN, 3, CvType.CV_32F)
        val sdata = FloatArray(sampleN * 3)
        var s = 0
        var i = 0
        while (i < count && s < sampleN) {
            sdata[s * 3] = (px[i * 3].toInt() and 0xFF).toFloat()
            sdata[s * 3 + 1] = (px[i * 3 + 1].toInt() and 0xFF).toFloat()
            sdata[s * 3 + 2] = (px[i * 3 + 2].toInt() and 0xFF).toFloat()
            s++; i += step
        }
        samples.put(0, 0, sdata)
        val kk = min(k, sampleN)
        val labels = Mat(); val centers = Mat()
        Core.kmeans(
            samples, kk, labels,
            TermCriteria(TermCriteria.EPS + TermCriteria.COUNT, 12, 1.0),
            2, Core.KMEANS_PP_CENTERS, centers,
        )
        val c = FloatArray(kk * 3)
        centers.get(0, 0, c)
        listOf(samples, labels, centers).forEach { it.release() }

        // Lookup table over 5 bits per channel.
        val lut = ByteArray(32 * 32 * 32)
        for (b in 0 until 32) for (g in 0 until 32) for (r in 0 until 32) {
            val bb = b * 8 + 4f; val gg = g * 8 + 4f; val rr = r * 8 + 4f
            var best = 0; var bestD = Float.MAX_VALUE
            for (j in 0 until kk) {
                val db = bb - c[j * 3]; val dg = gg - c[j * 3 + 1]; val dr = rr - c[j * 3 + 2]
                val d = db * db + dg * dg + dr * dr
                if (d < bestD) { bestD = d; best = j }
            }
            lut[(b shl 10) or (g shl 5) or r] = best.toByte()
        }
        val idx = ByteArray(count)
        for (p in 0 until count) {
            val b = (px[p * 3].toInt() and 0xFF) shr 3
            val g = (px[p * 3 + 1].toInt() and 0xFF) shr 3
            val r = (px[p * 3 + 2].toInt() and 0xFF) shr 3
            idx[p] = lut[(b shl 10) or (g shl 5) or r]
        }
        val palette = ByteArray(kk * 3)
        for (j in 0 until kk) {
            palette[j * 3] = c[j * 3 + 2].toInt().coerceIn(0, 255).toByte()     // R
            palette[j * 3 + 1] = c[j * 3 + 1].toInt().coerceIn(0, 255).toByte() // G
            palette[j * 3 + 2] = c[j * 3].toInt().coerceIn(0, 255).toByte()     // B
        }
        return palette to idx
    }

    private fun png(
        w: Int, h: Int, bitDepth: Int, colorType: Int, palette: ByteArray?,
        rows: (y: Int, out: ByteArray) -> Unit, rowBytes: Int,
    ): ByteArray {
        val out = ByteArrayOutputStream()
        out.write(byteArrayOf(0x89.toByte(), 'P'.code.toByte(), 'N'.code.toByte(), 'G'.code.toByte(), 13, 10, 26, 10))
        val ihdr = ByteArrayOutputStream().also {
            DataOutputStream(it).apply {
                writeInt(w); writeInt(h); writeByte(bitDepth); writeByte(colorType)
                writeByte(0); writeByte(0); writeByte(0)
            }
        }.toByteArray()
        chunk(out, "IHDR", ihdr)
        if (palette != null) chunk(out, "PLTE", palette)

        val deflater = Deflater(6)
        val compressed = ByteArrayOutputStream()
        val row = ByteArray(rowBytes + 1)
        val buf = ByteArray(64 * 1024)
        for (y in 0 until h) {
            rows(y, row)
            deflater.setInput(row)
            while (!deflater.needsInput()) {
                val n = deflater.deflate(buf)
                if (n > 0) compressed.write(buf, 0, n)
            }
        }
        deflater.finish()
        while (!deflater.finished()) {
            val n = deflater.deflate(buf)
            compressed.write(buf, 0, n)
        }
        deflater.end()
        chunk(out, "IDAT", compressed.toByteArray())
        chunk(out, "IEND", ByteArray(0))
        return out.toByteArray()
    }

    private fun chunk(out: ByteArrayOutputStream, type: String, data: ByteArray) {
        val d = DataOutputStream(out)
        d.writeInt(data.size)
        val t = type.toByteArray(Charsets.US_ASCII)
        d.write(t); d.write(data)
        val crc = CRC32()
        crc.update(t); crc.update(data)
        d.writeInt(crc.value.toInt())
    }
}
