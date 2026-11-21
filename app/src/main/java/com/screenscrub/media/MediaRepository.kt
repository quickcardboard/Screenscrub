package com.screenscrub.media

import android.Manifest
import android.content.ContentUris
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.util.Size
import androidx.core.content.ContextCompat
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.sqrt

/** A camera photo found in MediaStore. */
data class SourceImage(
    val id: Long,
    val uri: Uri,
    val displayName: String,
    val dateTaken: Long?,
    val sizeBytes: Long,
    val generation: Long = -1,
    val dateAddedSec: Long = 0,
)

enum class PhotoAccess { FULL, PARTIAL, NONE }

/** Camera photos found by a scan, plus the cursor values to store for the next scan. */
data class ScanResult(val images: List<SourceImage>, val maxGeneration: Long, val maxDateAddedSec: Long)

/** Everything that touches MediaStore and image files. Originals are only ever opened for reading. */
class MediaRepository(private val context: Context) {

    private val resolver get() = context.contentResolver

    companion object {
        const val OUTPUT_DIR = "ScreenScrub"
        const val MAX_DECODE_PIXELS = 16_000_000L

        fun readPermissions(): Array<String> = when {
            Build.VERSION.SDK_INT >= 34 -> arrayOf(Manifest.permission.READ_MEDIA_IMAGES, Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED)
            Build.VERSION.SDK_INT >= 33 -> arrayOf(Manifest.permission.READ_MEDIA_IMAGES)
            Build.VERSION.SDK_INT <= 28 -> arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE)
            else -> arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE)
        }
    }

    fun access(): PhotoAccess {
        fun granted(p: String) = ContextCompat.checkSelfPermission(context, p) == PackageManager.PERMISSION_GRANTED
        return when {
            Build.VERSION.SDK_INT >= 33 && granted(Manifest.permission.READ_MEDIA_IMAGES) -> PhotoAccess.FULL
            Build.VERSION.SDK_INT >= 34 && granted(Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED) -> PhotoAccess.PARTIAL
            Build.VERSION.SDK_INT < 33 && granted(Manifest.permission.READ_EXTERNAL_STORAGE) -> PhotoAccess.FULL
            else -> PhotoAccess.NONE
        }
    }

    /**
     * New camera photos since the cursor. On API 30+ the cursor is GENERATION_MODIFIED, which also
     * catches photos that were still pending (being written) at the last scan.
     */
    fun newCameraImages(lastGeneration: Long, sinceDateAddedSec: Long): ScanResult {
        val collection = MediaStore.Images.Media.EXTERNAL_CONTENT_URI
        val cols = mutableListOf(
            MediaStore.Images.Media._ID, MediaStore.Images.Media.DISPLAY_NAME,
            MediaStore.Images.Media.DATE_TAKEN, MediaStore.Images.Media.DATE_ADDED, MediaStore.Images.Media.SIZE,
        )
        if (Build.VERSION.SDK_INT >= 29) cols += MediaStore.MediaColumns.RELATIVE_PATH
        else @Suppress("DEPRECATION") cols += MediaStore.MediaColumns.DATA
        if (Build.VERSION.SDK_INT >= 30) cols += MediaStore.MediaColumns.GENERATION_MODIFIED

        val (selection, args) = when {
            Build.VERSION.SDK_INT >= 30 && lastGeneration >= 0 ->
                "${MediaStore.MediaColumns.GENERATION_MODIFIED} > ? AND ${MediaStore.MediaColumns.IS_PENDING} = 0" to arrayOf(lastGeneration.toString())
            Build.VERSION.SDK_INT >= 29 ->
                "${MediaStore.Images.Media.DATE_ADDED} >= ? AND ${MediaStore.MediaColumns.IS_PENDING} = 0" to arrayOf(sinceDateAddedSec.toString())
            else -> "${MediaStore.Images.Media.DATE_ADDED} >= ?" to arrayOf(sinceDateAddedSec.toString())
        }

        val out = ArrayList<SourceImage>()
        var maxGen = lastGeneration
        var maxAdded = sinceDateAddedSec
        resolver.query(collection, cols.toTypedArray(), selection, args, "${MediaStore.Images.Media.DATE_ADDED} ASC")?.use { c ->
            val iId = c.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
            val iName = c.getColumnIndexOrThrow(MediaStore.Images.Media.DISPLAY_NAME)
            val iTaken = c.getColumnIndexOrThrow(MediaStore.Images.Media.DATE_TAKEN)
            val iAdded = c.getColumnIndexOrThrow(MediaStore.Images.Media.DATE_ADDED)
            val iSize = c.getColumnIndexOrThrow(MediaStore.Images.Media.SIZE)
            val iPath = c.getColumnIndexOrThrow(if (Build.VERSION.SDK_INT >= 29) MediaStore.MediaColumns.RELATIVE_PATH else @Suppress("DEPRECATION") MediaStore.MediaColumns.DATA)
            val iGen = if (Build.VERSION.SDK_INT >= 30) c.getColumnIndexOrThrow(MediaStore.MediaColumns.GENERATION_MODIFIED) else -1
            while (c.moveToNext()) {
                val path = c.getString(iPath) ?: ""
                val id = c.getLong(iId)
                val gen = if (iGen >= 0) c.getLong(iGen) else -1
                val added = c.getLong(iAdded)
                maxGen = maxOf(maxGen, gen)
                maxAdded = maxOf(maxAdded, added)
                if (!isCameraPath(path)) continue
                out += SourceImage(
                    id = id,
                    uri = ContentUris.withAppendedId(collection, id),
                    displayName = c.getString(iName) ?: "photo_$id",
                    dateTaken = if (c.isNull(iTaken)) null else c.getLong(iTaken),
                    sizeBytes = c.getLong(iSize),
                    generation = gen,
                    dateAddedSec = added,
                )
            }
        }
        return ScanResult(out, maxGen, maxAdded)
    }

    /** Camera folders differ by manufacturer, but they all live under DCIM. */
    private fun isCameraPath(path: String): Boolean {
        val p = path.lowercase(Locale.ROOT)
        return "dcim/" in p && "screenshot" !in p && OUTPUT_DIR.lowercase(Locale.ROOT) !in p
    }

    /** Builds a SourceImage for an arbitrary URI (photo picker, share sheet). */
    fun describe(uri: Uri): SourceImage {
        var name = "photo"
        var size = 0L
        var taken: Long? = null
        runCatching {
            resolver.query(uri, null, null, null, null)?.use { c ->
                if (c.moveToFirst()) {
                    c.getColumnIndex(OpenableColumns.DISPLAY_NAME).takeIf { it >= 0 }?.let { name = c.getString(it) ?: name }
                    c.getColumnIndex(OpenableColumns.SIZE).takeIf { it >= 0 }?.let { size = c.getLong(it) }
                    c.getColumnIndex(MediaStore.Images.Media.DATE_TAKEN).takeIf { it >= 0 && !c.isNull(it) }?.let { taken = c.getLong(it) }
                }
            }
        }
        // Negative IDs never collide with MediaStore IDs.
        val id = -((uri.toString().hashCode().toLong() and 0x7fffffffL) + 1)
        return SourceImage(id, uri, name, taken, size)
    }

    /**
     * Decodes with orientation applied, capped at [maxPixels]. ImageDecoder handles HEIC and
     * rotation on API 28+; older devices use BitmapFactory plus the EXIF orientation.
     */
    fun decode(uri: Uri, maxPixels: Long = MAX_DECODE_PIXELS): Bitmap {
        if (Build.VERSION.SDK_INT >= 28) {
            val src = ImageDecoder.createSource(resolver, uri)
            return ImageDecoder.decodeBitmap(src) { decoder, info, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                val px = info.size.width.toLong() * info.size.height
                if (px > maxPixels) {
                    val f = sqrt(maxPixels.toDouble() / px)
                    decoder.setTargetSize((info.size.width * f).toInt(), (info.size.height * f).toInt())
                }
            }
        }
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
        var sample = 1
        while (bounds.outWidth.toLong() * bounds.outHeight / (sample * sample) > maxPixels) sample *= 2
        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        val bmp = resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) }
            ?: error("Cannot decode $uri")
        val rotation = resolver.openInputStream(uri)?.use { ExifInterface(it).rotationDegrees } ?: 0
        if (rotation == 0) return bmp
        val m = Matrix().apply { postRotate(rotation.toFloat()) }
        return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true).also { if (it !== bmp) bmp.recycle() }
    }

    /** Small preview for lists. */
    fun thumbnail(uri: Uri, size: Int): Bitmap? = runCatching {
        if (Build.VERSION.SDK_INT >= 29) resolver.loadThumbnail(uri, Size(size, size), null)
        else decode(uri, size.toLong() * size * 2)
    }.getOrNull()

    /** EXIF FocalLengthIn35mmFilm, used to recover the true aspect ratio. GPS is never read. */
    fun focalLength35mm(uri: Uri): Double? = runCatching {
        resolver.openInputStream(uri)?.use { s ->
            ExifInterface(s).getAttributeInt(ExifInterface.TAG_FOCAL_LENGTH_IN_35MM_FILM, 0).takeIf { it > 0 }?.toDouble()
        }
    }.getOrNull()

    /**
     * Saves a new image to Pictures/ScreenScrub. Copies only the capture date: no GPS, no other EXIF.
     */
    fun save(bytes: ByteArray, mime: String, displayName: String, dateTaken: Long?): Uri {
        if (Build.VERSION.SDK_INT >= 29) {
            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, displayName)
                put(MediaStore.MediaColumns.MIME_TYPE, mime)
                put(MediaStore.MediaColumns.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/$OUTPUT_DIR")
                put(MediaStore.MediaColumns.IS_PENDING, 1)
                dateTaken?.let { put(MediaStore.MediaColumns.DATE_TAKEN, it) }
            }
            val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                ?: error("MediaStore insert failed")
            try {
                resolver.openOutputStream(uri)?.use { it.write(bytes) } ?: error("Cannot open $uri")
                if (mime == "image/jpeg" && dateTaken != null) writeExifDate(uri, dateTaken)
                resolver.update(uri, ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }, null, null)
            } catch (e: Exception) {
                resolver.delete(uri, null, null)
                throw e
            }
            return uri
        }
        @Suppress("DEPRECATION")
        val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), OUTPUT_DIR)
        dir.mkdirs()
        var file = File(dir, displayName)
        var n = 1
        while (file.exists()) file = File(dir, displayName.substringBeforeLast('.') + "_${n++}." + displayName.substringAfterLast('.'))
        file.writeBytes(bytes)
        if (mime == "image/jpeg" && dateTaken != null) runCatching {
            ExifInterface(file.absolutePath).apply { setAttribute(ExifInterface.TAG_DATETIME_ORIGINAL, exifDate(dateTaken)); saveAttributes() }
        }
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, file.name)
            put(MediaStore.MediaColumns.MIME_TYPE, mime)
            @Suppress("DEPRECATION") put(MediaStore.MediaColumns.DATA, file.absolutePath)
            dateTaken?.let { put(MediaStore.Images.Media.DATE_TAKEN, it) }
        }
        return resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: Uri.fromFile(file)
    }

    private fun writeExifDate(uri: Uri, dateTaken: Long) = runCatching {
        resolver.openFileDescriptor(uri, "rw")?.use { pfd ->
            ExifInterface(pfd.fileDescriptor).apply {
                setAttribute(ExifInterface.TAG_DATETIME_ORIGINAL, exifDate(dateTaken))
                saveAttributes()
            }
        }
    }

    private fun exifDate(ms: Long) = SimpleDateFormat("yyyy:MM:dd HH:mm:ss", Locale.US).format(Date(ms))

    /** Deletes one of our outputs. May throw RecoverableSecurityException after a reinstall. */
    fun delete(uri: Uri): Boolean = resolver.delete(uri, null, null) > 0

    fun exists(uri: Uri): Boolean = runCatching {
        resolver.query(uri, arrayOf(MediaStore.MediaColumns._ID), null, null, null)?.use { it.count > 0 } ?: false
    }.getOrDefault(false)
}
