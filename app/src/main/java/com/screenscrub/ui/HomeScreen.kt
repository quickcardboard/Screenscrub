package com.screenscrub.ui

import android.Manifest
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.provider.Settings
import android.text.format.DateUtils
import android.text.format.Formatter
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.screenscrub.data.ProcessedImage
import com.screenscrub.media.MediaRepository
import com.screenscrub.media.PhotoAccess
import com.screenscrub.vision.ImageClass
import com.screenscrub.work.Notifications

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(vm: MainViewModel) {
    val ctx = LocalContext.current
    val results by vm.results.collectAsStateWithLifecycle()
    val settings by vm.settings.collectAsStateWithLifecycle()
    val access by vm.access.collectAsStateWithLifecycle()
    val busy by vm.busy.collectAsStateWithLifecycle()
    val unclassified by vm.unclassified.collectAsStateWithLifecycle()
    val message by vm.message.collectAsStateWithLifecycle()
    val snackbar = remember { SnackbarHostState() }

    LaunchedEffect(message) {
        message?.let { snackbar.showSnackbar(it); vm.consumeMessage() }
    }

    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
        vm.refreshAccess()
    }
    val notifLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {}
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let { vm.processPicked(it) }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("ScreenScrub") },
                actions = {
                    IconButton(onClick = { vm.navigate(Screen.Settings) }) { Icon(Icons.Default.Settings, "Settings") }
                },
            )
        },
        floatingActionButton = {
            ExtendedFloatingActionButton(
                onClick = { picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                icon = { Icon(Icons.Default.AddPhotoAlternate, null) },
                text = { Text("Clean a photo") },
            )
        },
        snackbarHost = { SnackbarHost(snackbar) },
    ) { pad ->
        LazyColumn(
            Modifier.fillMaxSize().padding(pad),
            contentPadding = PaddingValues(16.dp, 8.dp, 16.dp, 96.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            if (busy) item { LinearProgressIndicator(Modifier.fillMaxWidth()) }

            item {
                AutoCard(
                    enabled = settings.autoEnabled,
                    access = access,
                    onToggle = { on ->
                        if (on && access != PhotoAccess.FULL) permissionLauncher.launch(MediaRepository.readPermissions())
                        if (on && Build.VERSION.SDK_INT >= 33) notifLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
                        vm.setAuto(on)
                    },
                )
            }

            if (access != PhotoAccess.FULL) {
                item {
                    PermissionCard(access) { permissionLauncher.launch(MediaRepository.readPermissions()) }
                }
            }

            if (results.isEmpty()) {
                item { EmptyState() }
            } else {
                item {
                    Text(
                        "Cleaned photos", style = MaterialTheme.typography.titleMedium,
                        modifier = Modifier.padding(top = 8.dp),
                    )
                }
                items(results, key = { it.sourceId }) { row ->
                    ResultRow(
                        row,
                        onOpen = { vm.navigate(Screen.Detail(row.sourceId)) },
                        onShare = {
                            row.outputUri?.let { u ->
                                ctx.startActivity(Notifications.shareIntent(ctx, listOf(Uri.parse(u)), listOfNotNull(row.outputMime)))
                            }
                        },
                    )
                }
            }
        }
    }

    unclassified?.let { u ->
        AlertDialog(
            onDismissRequest = vm::dismissUnclassified,
            title = { Text("Not sure what this is") },
            text = { Text("This photo doesn't look like a screen or a document. Clean it anyway?") },
            confirmButton = {
                Row {
                    TextButton(onClick = { vm.processPicked(u.uri, ImageClass.SCREEN) }) { Text("As screen") }
                    TextButton(onClick = { vm.processPicked(u.uri, ImageClass.DOCUMENT) }) { Text("As document") }
                }
            },
            dismissButton = { TextButton(onClick = vm::dismissUnclassified) { Text("Cancel") } },
        )
    }
}

@Composable
private fun AutoCard(enabled: Boolean, access: PhotoAccess, onToggle: (Boolean) -> Unit) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)) {
        Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("Clean new photos automatically", style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.size(4.dp))
                Text(
                    when {
                        enabled && access == PhotoAccess.FULL ->
                            "On. Photos of screens and documents you take are cleaned in the background and saved to Pictures/ScreenScrub."
                        enabled -> "Waiting for full photo access."
                        else -> "Off. You can still clean photos by hand or with \"Clean & share\" in the share menu."
                    },
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
            Spacer(Modifier.width(12.dp))
            Switch(checked = enabled, onCheckedChange = onToggle)
        }
    }
}

@Composable
private fun PermissionCard(access: PhotoAccess, onRequest: () -> Unit) {
    val ctx = LocalContext.current
    Card {
        Column(Modifier.padding(16.dp)) {
            Text(
                if (access == PhotoAccess.PARTIAL) "Only some photos are shared with ScreenScrub" else "Photo access needed for automatic mode",
                style = MaterialTheme.typography.titleMedium,
            )
            Spacer(Modifier.size(4.dp))
            Text(
                if (access == PhotoAccess.PARTIAL)
                    "Automatic cleaning needs to see each new photo as you take it, which \"selected photos\" access doesn't allow. " +
                        "Choose \"Allow all\" to turn it on. Nothing leaves your phone: ScreenScrub has no internet access."
                else
                    "ScreenScrub looks at new camera photos on your phone to find screens and documents. " +
                        "It has no internet access, and your originals are never changed.",
                style = MaterialTheme.typography.bodyMedium,
            )
            Row(Modifier.padding(top = 8.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilledTonalButton(onClick = onRequest) { Text("Allow access") }
                TextButton(onClick = {
                    ctx.startActivity(
                        Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", ctx.packageName, null)),
                    )
                }) { Text("App settings") }
            }
        }
    }
}

@Composable
private fun EmptyState() {
    Column(Modifier.fillMaxWidth().padding(vertical = 48.dp), horizontalAlignment = Alignment.CenterHorizontally) {
        Text("No cleaned photos yet", style = MaterialTheme.typography.titleMedium)
        Spacer(Modifier.size(8.dp))
        Text(
            "Take a photo of a screen or a document, or tap \"Clean a photo\".",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun ResultRow(row: ProcessedImage, onOpen: () -> Unit, onShare: () -> Unit) {
    val ctx = LocalContext.current
    Card(Modifier.fillMaxWidth().clickable(onClick = onOpen)) {
        Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            UriImage(Uri.parse(row.sourceUri), 256, Modifier.size(64.dp).clip(RoundedCornerShape(8.dp)), contentDescription = "Original")
            Icon(
                Icons.AutoMirrored.Filled.ArrowForward, null,
                Modifier.padding(horizontal = 6.dp).size(16.dp), tint = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            UriImage(row.outputUri?.let(Uri::parse), 256, Modifier.size(64.dp).clip(RoundedCornerShape(8.dp)), contentDescription = "Cleaned")
            Column(Modifier.weight(1f).padding(start = 12.dp)) {
                Text(row.sourceName, maxLines = 1, overflow = TextOverflow.Ellipsis, fontWeight = FontWeight.Medium)
                Text(
                    "${Formatter.formatShortFileSize(ctx, row.sourceBytes)} → ${Formatter.formatShortFileSize(ctx, row.outputBytes)} · " +
                        DateUtils.getRelativeTimeSpanString(row.processedAt),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Box(Modifier.padding(top = 4.dp)) {
                    AssistChip(onClick = onOpen, label = { Text(if (row.imageClass == "SCREEN") "Screen" else "Document") })
                }
            }
            IconButton(onClick = onShare) { Icon(Icons.Default.Share, "Share") }
        }
    }
}
