package com.screenscrub.ui

import android.graphics.Bitmap
import android.net.Uri
import android.util.LruCache
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.material3.MaterialTheme
import com.screenscrub.ScreenScrubApp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

private val cache = object : LruCache<String, Bitmap>(48 * 1024 * 1024) {
    override fun sizeOf(key: String, value: Bitmap) = value.byteCount
}

/** Loads a content URI off the main thread; [size] is the longest edge wanted. */
@Composable
fun UriImage(
    uri: Uri?,
    size: Int,
    modifier: Modifier = Modifier,
    contentScale: ContentScale = ContentScale.Crop,
    contentDescription: String? = null,
) {
    val app = LocalContext.current.applicationContext as ScreenScrubApp
    val key = "$uri@$size"
    val bmp by produceState(cache.get(key), key) {
        if (uri == null || value != null) return@produceState
        value = withContext(Dispatchers.IO) {
            val b = if (size <= 512) app.media.thumbnail(uri, size)
            else runCatching { app.media.decode(uri, size.toLong() * size) }.getOrNull()
            b?.also { cache.put(key, it) }
        }
    }
    Box(modifier.background(MaterialTheme.colorScheme.surfaceVariant)) {
        bmp?.let {
            Image(it.asImageBitmap(), contentDescription, Modifier.matchParentSize(), contentScale = contentScale)
        }
    }
}
