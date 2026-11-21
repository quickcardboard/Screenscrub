package com.screenscrub.ui

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import com.screenscrub.ScreenScrubApp
import com.screenscrub.data.Origin
import com.screenscrub.vision.ImageClass
import com.screenscrub.work.Notifications
import kotlinx.coroutines.launch

/**
 * "Clean & share": receives images from any app's share sheet, cleans them, then opens the
 * share sheet again with the cleaned copies.
 */
class ShareActivity : ComponentActivity() {
    private var status by mutableStateOf("Cleaning…")
    private var askFor by mutableStateOf<Uri?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val uris = incoming(intent)
        if (uris.isEmpty()) { finish(); return }
        setContent {
            ScreenScrubTheme {
                val ask = askFor
                if (ask == null) {
                    AlertDialog(
                        onDismissRequest = {},
                        confirmButton = {},
                        text = {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                CircularProgressIndicator(Modifier.padding(end = 16.dp))
                                Text(status)
                            }
                        },
                    )
                } else {
                    AlertDialog(
                        onDismissRequest = { finish() },
                        title = { Text("Not sure what this is") },
                        text = { Text("This photo doesn't look like a screen or a document.") },
                        confirmButton = {
                            Column(horizontalAlignment = Alignment.End) {
                                TextButton(onClick = { askFor = null; run(listOf(ask), ImageClass.SCREEN) }) { Text("Clean as screen") }
                                TextButton(onClick = { askFor = null; run(listOf(ask), ImageClass.DOCUMENT) }) { Text("Clean as document") }
                                TextButton(onClick = { share(listOf(ask), listOf(contentResolver.getType(ask) ?: "image/*")) }) { Text("Send original") }
                            }
                        },
                    )
                }
            }
        }
        run(uris, null)
    }

    private fun run(uris: List<Uri>, force: ImageClass?) = lifecycleScope.launch {
        val app = application as ScreenScrubApp
        val out = mutableListOf<Uri>()
        val mimes = mutableListOf<String>()
        uris.forEachIndexed { i, uri ->
            status = if (uris.size > 1) "Cleaning ${i + 1} of ${uris.size}…" else "Cleaning…"
            val res = app.processor.process(app.media.describe(uri), Origin.SHARE, forceClass = force)
            val clean = res?.record?.outputUri?.let(Uri::parse)
            when {
                clean != null -> { out += clean; mimes += res.record.outputMime ?: "image/*" }
                uris.size == 1 && res?.record?.error == null -> { askFor = uri; return@launch }
                else -> { out += uri; mimes += contentResolver.getType(uri) ?: "image/*" } // send the original
            }
        }
        share(out, mimes)
    }

    private fun share(uris: List<Uri>, mimes: List<String>) {
        startActivity(Notifications.shareIntent(this, uris, mimes))
        finish()
    }

    private fun incoming(intent: Intent): List<Uri> = when (intent.action) {
        Intent.ACTION_SEND -> listOfNotNull(
            if (Build.VERSION.SDK_INT >= 33) intent.getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)
            else @Suppress("DEPRECATION") intent.getParcelableExtra(Intent.EXTRA_STREAM),
        )
        Intent.ACTION_SEND_MULTIPLE ->
            (if (Build.VERSION.SDK_INT >= 33) intent.getParcelableArrayListExtra(Intent.EXTRA_STREAM, Uri::class.java)
            else @Suppress("DEPRECATION") intent.getParcelableArrayListExtra(Intent.EXTRA_STREAM)) ?: emptyList()
        else -> emptyList()
    }
}
