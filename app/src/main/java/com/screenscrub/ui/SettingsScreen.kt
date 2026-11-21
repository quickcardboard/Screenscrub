package com.screenscrub.ui

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.screenscrub.vision.OutputStyle

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(vm: MainViewModel) {
    val s by vm.settings.collectAsStateWithLifecycle()
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Settings") },
                navigationIcon = { IconButton(onClick = { vm.back() }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") } },
            )
        },
    ) { pad ->
        Column(Modifier.fillMaxSize().padding(pad).verticalScroll(rememberScrollState()).padding(horizontal = 16.dp)) {
            Section("Output")
            StylePicker("Screen photos", s.screenStyle, vm::setScreenStyle)
            StylePicker("Document photos", s.documentStyle, vm::setDocumentStyle)

            Section("Detection")
            var sens by remember(s.sensitivity) { mutableFloatStateOf(s.sensitivity) }
            Text("Sensitivity", style = MaterialTheme.typography.bodyLarge)
            Text(
                "Stricter misses some photos; more eager may clean photos that aren't screens or documents.",
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Slider(value = sens, onValueChange = { sens = it }, onValueChangeFinished = { vm.setSensitivity(sens) }, valueRange = -1f..1f, steps = 3)
            Row { Text("Stricter", Modifier.weight(1f), style = MaterialTheme.typography.labelSmall); Text("More eager", style = MaterialTheme.typography.labelSmall) }

            Section("Background")
            SwitchRow("Only while charging", "Saves battery, but photos are cleaned later.", s.chargingOnly, vm::setChargingOnly)
            SwitchRow("Notifications", "Tell me when a photo has been cleaned, with a Share button.", s.notify, vm::setNotify)

            Section("Privacy")
            Text(
                "ScreenScrub has no internet permission. Originals are never changed. Cleaned copies keep the capture date only: no location or other metadata.",
                style = MaterialTheme.typography.bodyMedium, modifier = Modifier.padding(bottom = 24.dp),
            )
        }
    }
}

@Composable
private fun Section(title: String) {
    HorizontalDivider(Modifier.padding(top = 16.dp))
    Text(title, style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary, modifier = Modifier.padding(vertical = 12.dp))
}

@Composable
private fun StylePicker(label: String, value: OutputStyle, onPick: (OutputStyle) -> Unit) {
    var open by remember { mutableStateOf(false) }
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(label, Modifier.weight(1f), style = MaterialTheme.typography.bodyLarge)
        Box {
            TextButton(onClick = { open = true }) { Text(styleLabel(value)) }
            DropdownMenu(open, onDismissRequest = { open = false }) {
                OutputStyle.entries.forEach { st ->
                    DropdownMenuItem(text = { Text(styleLabel(st)) }, onClick = { open = false; onPick(st) })
                }
            }
        }
    }
}

@Composable
private fun SwitchRow(title: String, subtitle: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(Modifier.padding(vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.bodyLarge)
            Text(subtitle, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Switch(checked, onChange)
    }
}
