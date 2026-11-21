package com.screenscrub.work

import android.graphics.Bitmap
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import com.screenscrub.data.AppSettings
import com.screenscrub.data.Origin
import com.screenscrub.data.ProcessedDao
import com.screenscrub.data.ProcessedImage
import com.screenscrub.data.SettingsStore
import com.screenscrub.media.MediaRepository
import com.screenscrub.media.SourceImage
import com.screenscrub.vision.Classification
import com.screenscrub.vision.ClassifierConfig
import com.screenscrub.vision.ImageClass
import com.screenscrub.vision.OutputStyle
import com.screenscrub.vision.PhotoMeta
import com.screenscrub.vision.ScreenScrub
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

/**
 * The one place that runs the pipeline: decode → classify → clean → save → record.
 * Used by the background worker, the in-app picker and the share sheet.
 */
class Processor(
    private val media: MediaRepository,
    private val dao: ProcessedDao,
    private val settings: SettingsStore,
) {
    /** One image at a time: a 16 MP decode plus working copies is a lot of memory. */
    private val lock = Mutex()

    data class Outcome(val record: ProcessedImage, val classification: Classification?)

    /**
     * Processes [source]. With [forceClass] the image is cleaned as that class even if it
     * classifies as OTHER (the user asked for it). [skipIfKnown] makes auto mode idempotent.
     */
    suspend fun process(
        source: SourceImage,
        origin: Origin,
        forceClass: ImageClass? = null,
        styleOverride: OutputStyle? = null,
        skipIfKnown: Boolean = false,
    ): Outcome? = withContext(Dispatchers.Default) {
        lock.withLock {
            if (skipIfKnown && dao.exists(source.id)) return@withLock null
            val s = settings.current()
            val started = SystemClock.elapsedRealtime()
            var mat: Mat? = null
            try {
                mat = decodeToMat(source.uri)
                val meta = PhotoMeta(focalLength35mm = media.focalLength35mm(source.uri))
                val scrub = ScreenScrub(config(s))
                val c = scrub.classify(mat, meta)
                val cls = forceClass ?: c.imageClass
                if (cls == ImageClass.OTHER) {
                    val row = record(source, c, origin, null, null, null, started)
                    dao.upsert(row)
                    return@withLock Outcome(row, c)
                }
                val style = styleOverride ?: if (cls == ImageClass.SCREEN) s.screenStyle else s.documentStyle
                val encoded = scrub.clean(mat, c, style, meta, asClass = cls)!!
                val name = source.displayName.substringBeforeLast('.') + "_clean." + encoded.extension
                val uri = media.save(encoded.bytes, encoded.mimeType, name, source.dateTaken)
                val row = record(source, c, origin, uri, encoded, cls, started)
                dao.upsert(row)
                Log.i(TAG, "${source.displayName}: $cls ${encoded.style} ${encoded.bytes.size / 1024} KB in ${row.durationMs} ms")
                Outcome(row, c)
            } catch (t: Throwable) {
                Log.e(TAG, "Failed on ${source.uri}", t)
                val row = ProcessedImage(
                    sourceId = source.id, sourceUri = source.uri.toString(), sourceName = source.displayName,
                    outputUri = null, outputMime = null, imageClass = "ERROR", screenScore = 0.0, documentScore = 0.0,
                    features = "", style = null, sourceBytes = source.sizeBytes, outputBytes = 0, outputWidth = 0,
                    outputHeight = 0, durationMs = SystemClock.elapsedRealtime() - started,
                    processedAt = System.currentTimeMillis(), origin = origin.name, error = t.message ?: t.javaClass.simpleName,
                )
                dao.upsert(row)
                Outcome(row, null)
            } finally {
                mat?.release()
            }
        }
    }

    /** Re-runs an existing result with another style or class, replacing its output file. */
    suspend fun reprocess(row: ProcessedImage, style: OutputStyle?, asClass: ImageClass?): Outcome? {
        row.outputUri?.let { runCatching { media.delete(Uri.parse(it)) } }
        dao.clearOutput(row.sourceId)
        val src = SourceImage(row.sourceId, Uri.parse(row.sourceUri), row.sourceName, null, row.sourceBytes)
        return process(src, Origin.valueOf(row.origin), forceClass = asClass ?: ImageClass.valueOf(row.imageClass).takeIf { it != ImageClass.OTHER }, styleOverride = style)
    }

    /** Debug images: quad overlay and FFT spectrum, computed on demand. */
    suspend fun debugImages(sourceUri: Uri): Pair<Bitmap, Bitmap> = withContext(Dispatchers.Default) {
        lock.withLock {
            val mat = decodeToMat(sourceUri)
            val scrub = ScreenScrub(config(settings.current()))
            val c = scrub.classify(mat, PhotoMeta(media.focalLength35mm(sourceUri)))
            val overlay = scrub.debugOverlay(mat, c)
            val spectrum = scrub.debugSpectrum(mat, c)
            mat.release()
            toBitmap(overlay) to toBitmap(spectrum)
        }
    }

    private fun config(s: AppSettings) = ClassifierConfig(sensitivity = s.sensitivity.toDouble())

    private fun decodeToMat(uri: Uri): Mat {
        val bmp = media.decode(uri)
        val argb = if (bmp.config == Bitmap.Config.ARGB_8888) bmp else bmp.copy(Bitmap.Config.ARGB_8888, false).also { bmp.recycle() }
        val rgba = Mat()
        Utils.bitmapToMat(argb, rgba)
        argb.recycle()
        val bgr = Mat()
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        rgba.release()
        return bgr
    }

    private fun toBitmap(m: Mat): Bitmap {
        val rgba = Mat()
        Imgproc.cvtColor(m, rgba, if (m.channels() == 1) Imgproc.COLOR_GRAY2RGBA else Imgproc.COLOR_BGR2RGBA)
        val bmp = Bitmap.createBitmap(m.cols(), m.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgba, bmp)
        rgba.release(); m.release()
        return bmp
    }

    private fun record(
        source: SourceImage, c: Classification, origin: Origin, uri: Uri?,
        encoded: com.screenscrub.vision.EncodedImage?, cls: ImageClass?, started: Long,
    ) = ProcessedImage(
        sourceId = source.id,
        sourceUri = source.uri.toString(),
        sourceName = source.displayName,
        outputUri = uri?.toString(),
        outputMime = encoded?.mimeType,
        imageClass = (cls ?: c.imageClass).name,
        screenScore = c.screenScore,
        documentScore = c.documentScore,
        features = c.features.asMap().entries.joinToString(";") { "${it.key}=${"%.3f".format(java.util.Locale.US, it.value)}" },
        style = encoded?.style?.name,
        sourceBytes = source.sizeBytes,
        outputBytes = encoded?.bytes?.size?.toLong() ?: 0,
        outputWidth = encoded?.width ?: 0,
        outputHeight = encoded?.height ?: 0,
        durationMs = SystemClock.elapsedRealtime() - started,
        processedAt = System.currentTimeMillis(),
        origin = origin.name,
        error = null,
    )

    companion object {
        private const val TAG = "ScreenScrub"
    }
}
