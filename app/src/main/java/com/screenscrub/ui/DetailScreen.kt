package com.screenscrub.ui

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.text.format.Formatter
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.IntentSenderRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.screenscrub.BuildConfig
import com.screenscrub.data.ProcessedImage
import com.screenscrub.vision.ImageClass
import com.screenscrub.vision.OutputStyle
import com.screenscrub.work.Notifications
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DetailScreen(vm: MainViewModel, id: Long) {
    val row by remember(id) { vm.observe(id) }.collectAsState(initial = null)
    val busy by vm.busy.collectAsStateWithLifecycle()
    val ctx = LocalContext.current
    val r = row

    val deleteLauncher = rememberLauncherForActivityResult(ActivityResultContracts.StartIntentSenderForResult()) { res ->
        if (res.resultCode == Activity.RESULT_OK && r != null) vm.outputDeletedExternally(r)
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(if (r?.imageClass == "DOCUMENT") "Document" else "Screen") },
                navigationIcon = { IconButton(onClick = { vm.back() }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") } },
                actions = {
                    if (r?.outputUri != null) {
                        IconButton(onClick = {
                            vm.deleteOutput(r) { uri ->
                                if (Build.VERSION.SDK_INT >= 30) {
                                    val pi = MediaStore.createDeleteRequest(ctx.contentResolver, listOf(uri))
                                    deleteLauncher.launch(IntentSenderRequest.Builder(pi.intentSender).build())
                                }
                            }
                        }) { Icon(Icons.Default.Delete, "Delete cleaned copy") }
                    }
                },
            )
        },
    ) { pad ->
        if (r == null) return@Scaffold
        Column(Modifier.fillMaxSize().padding(pad).verticalScroll(rememberScrollState()).padding(16.dp)) {
            if (busy) LinearProgressIndicator(Modifier.fillMaxWidth().padding(bottom = 8.dp))
            BeforeAfter(r)
            Text(
                "${Formatter.formatShortFileSize(ctx, r.sourceBytes)} → ${Formatter.formatShortFileSize(ctx, r.outputBytes)}" +
                    " · ${r.outputWidth}×${r.outputHeight} · ${r.style?.lowercase() ?: ""}",
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.padding(top = 8.dp),
            )
            r.error?.let { Text("Error: $it", color = MaterialTheme.colorScheme.error) }

            Row(Modifier.padding(top = 16.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(
                    onClick = { r.outputUri?.let { ctx.startActivity(Notifications.shareIntent(ctx, listOf(Uri.parse(it)), listOfNotNull(r.outputMime))) } },
                    enabled = r.outputUri != null,
                ) {
                    Icon(Icons.Default.Share, null); Text("  Share")
                }
                OutlinedButton(
                    onClick = {
                        r.outputUri?.let {
                            ctx.startActivity(Intent(Intent.ACTION_VIEW).setDataAndType(Uri.parse(it), r.outputMime).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION))
                        }
                    },
                    enabled = r.outputUri != null,
                ) { Text("Open") }
            }

            ReprocessRow(r, onReprocess = { style, cls -> vm.reprocess(r, style, cls) })

            if (BuildConfig.DEBUG) DebugPanel(vm, r)
        }
    }
}

/** Shows the cleaned image; press and hold to see the original. */
@Composable
private fun BeforeAfter(r: ProcessedImage) {
    var showOriginal by remember { mutableStateOf(false) }
    val aspect = if (r.outputWidth > 0 && r.outputHeight > 0) r.outputWidth.toFloat() / r.outputHeight else 4f / 3
    Column {
        Box(
            Modifier.fillMaxWidth().aspectRatio(aspect.coerceIn(0.5f, 2.5f)).pointerInput(Unit) {
                detectTapGestures(onPress = { showOriginal = true; tryAwaitRelease(); showOriginal = false })
            },
        ) {
            UriImage(
                Uri.parse(if (showOriginal || r.outputUri == null) r.sourceUri else r.outputUri),
                2048, Modifier.fillMaxSize(), contentScale = ContentScale.Fit,
                contentDescription = if (showOriginal) "Original" else "Cleaned",
            )
        }
        Text(
            if (showOriginal) "Original" else "Cleaned — hold to compare with the original",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 4.dp),
        )
    }
}

@Composable
private fun ReprocessRow(r: ProcessedImage, onReprocess: (OutputStyle?, ImageClass?) -> Unit) {
    var styleMenu by remember { mutableStateOf(false) }
    var classMenu by remember { mutableStateOf(false) }
    Column(Modifier.padding(top = 16.dp)) {
        HorizontalDivider()
        Text("Not right? Redo it", style = MaterialTheme.typography.titleSmall, modifier = Modifier.padding(top = 12.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Box {
                TextButton(onClick = { styleMenu = true }) { Text("Output style") }
                DropdownMenu(styleMenu, onDismissRequest = { styleMenu = false }) {
                    OutputStyle.entries.forEach { s ->
                        DropdownMenuItem(text = { Text(styleLabel(s)) }, onClick = { styleMenu = false; onReprocess(s, null) })
                    }
                }
            }
            Box {
                TextButton(onClick = { classMenu = true }) { Text("Treat as…") }
                DropdownMenu(classMenu, onDismissRequest = { classMenu = false }) {
                    DropdownMenuItem(text = { Text("Screen") }, onClick = { classMenu = false; onReprocess(null, ImageClass.SCREEN) })
                    DropdownMenuItem(text = { Text("Document") }, onClick = { classMenu = false; onReprocess(null, ImageClass.DOCUMENT) })
                }
            }
        }
    }
}

fun styleLabel(s: OutputStyle) = when (s) {
    OutputStyle.AUTO -> "Automatic"
    OutputStyle.COLOR -> "Color (small PNG)"
    OutputStyle.BW -> "Black & white"
    OutputStyle.GRAY -> "Grayscale"
    OutputStyle.PHOTO -> "Photo (JPEG)"
}

/** Dev builds only: scores, features, quad overlay and FFT spectrum for threshold tuning. */
@Composable
private fun DebugPanel(vm: MainViewModel, r: ProcessedImage) {
    val scope = rememberCoroutineScope()
    var images by remember(r.sourceId) { mutableStateOf<Pair<Bitmap, Bitmap>?>(null) }
    var loading by remember { mutableStateOf(false) }
    Column(Modifier.padding(top = 24.dp)) {
        HorizontalDivider()
        Text("Debug", style = MaterialTheme.typography.titleSmall, modifier = Modifier.padding(top = 12.dp))
        Text(
            "class=${r.imageClass}  screen=%.2f  document=%.2f  %d ms  origin=%s".format(r.screenScore, r.documentScore, r.durationMs, r.origin),
            fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall,
        )
        r.featureMap().forEach { (k, v) ->
            Text("$k = $v", fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall)
        }
        TextButton(
            onClick = { loading = true; scope.launch { images = vm.debugImages(r); loading = false } },
            enabled = !loading,
        ) { Text(if (loading) "Computing…" else "Show quad overlay and spectrum") }
        images?.let { (overlay, spectrum) ->
            Image(overlay.asImageBitmap(), "Quad overlay", Modifier.fillMaxWidth().height(260.dp), contentScale = ContentScale.Fit)
            Image(spectrum.asImageBitmap(), "FFT spectrum", Modifier.fillMaxWidth().height(260.dp).padding(top = 8.dp), contentScale = ContentScale.Fit)
        }
    }
}
