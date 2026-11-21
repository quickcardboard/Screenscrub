package com.screenscrub.ui

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.screenscrub.ScreenScrubApp
import com.screenscrub.data.AppSettings
import com.screenscrub.data.Origin
import com.screenscrub.data.ProcessedImage
import com.screenscrub.media.PhotoAccess
import com.screenscrub.vision.ImageClass
import com.screenscrub.vision.OutputStyle
import com.screenscrub.work.Scheduler
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

sealed interface Screen {
    data object Home : Screen
    data class Detail(val id: Long) : Screen
    data object Settings : Screen
}

/** A picked image that looked like neither a screen nor a document: ask the user. */
data class Unclassified(val uri: Uri)

class MainViewModel(application: Application) : AndroidViewModel(application) {
    private val app = application as ScreenScrubApp

    val results: StateFlow<List<ProcessedImage>> =
        app.db.processed().results().stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())
    val settings: StateFlow<AppSettings> =
        app.settings.flow.stateIn(viewModelScope, SharingStarted.Eagerly, AppSettings())

    private val _screen = MutableStateFlow<Screen>(Screen.Home)
    val screen = _screen.asStateFlow()

    private val _access = MutableStateFlow(app.media.access())
    val access = _access.asStateFlow()

    private val _busy = MutableStateFlow(false)
    val busy = _busy.asStateFlow()

    private val _unclassified = MutableStateFlow<Unclassified?>(null)
    val unclassified = _unclassified.asStateFlow()

    private val _message = MutableStateFlow<String?>(null)
    val message = _message.asStateFlow()

    fun navigate(s: Screen) { _screen.value = s }
    fun back(): Boolean {
        if (_screen.value == Screen.Home) return false
        _screen.value = Screen.Home
        return true
    }
    fun consumeMessage() { _message.value = null }

    fun refreshAccess() {
        _access.value = app.media.access()
        if (_access.value == PhotoAccess.FULL && settings.value.autoEnabled) Scheduler.catchUp(app)
    }

    fun observe(id: Long) = app.db.processed().observe(id)

    fun setAuto(on: Boolean) = viewModelScope.launch {
        app.settings.setAuto(on)
        if (on) {
            Scheduler.arm(app, settings.value.chargingOnly, replace = true)
        } else {
            Scheduler.disarm(app)
        }
    }

    fun setChargingOnly(v: Boolean) = viewModelScope.launch {
        app.settings.setChargingOnly(v)
        if (settings.value.autoEnabled) Scheduler.arm(app, v, replace = true)
    }
    fun setNotify(v: Boolean) = viewModelScope.launch { app.settings.setNotify(v) }
    fun setSensitivity(v: Float) = viewModelScope.launch { app.settings.setSensitivity(v) }
    fun setScreenStyle(s: OutputStyle) = viewModelScope.launch { app.settings.setScreenStyle(s) }
    fun setDocumentStyle(s: OutputStyle) = viewModelScope.launch { app.settings.setDocumentStyle(s) }

    /** "Clean a photo": photo picker result. */
    fun processPicked(uri: Uri, forceClass: ImageClass? = null) = viewModelScope.launch {
        _unclassified.value = null
        _busy.value = true
        try {
            val src = app.media.describe(uri)
            val out = app.processor.process(src, Origin.MANUAL, forceClass = forceClass)
            when {
                out == null -> Unit
                out.record.error != null -> _message.value = "Couldn't process this photo: ${out.record.error}"
                out.record.outputUri == null -> _unclassified.value = Unclassified(uri)
                else -> _screen.value = Screen.Detail(out.record.sourceId)
            }
        } finally {
            _busy.value = false
        }
    }

    fun dismissUnclassified() { _unclassified.value = null }

    fun reprocess(row: ProcessedImage, style: OutputStyle?, asClass: ImageClass?) = viewModelScope.launch {
        _busy.value = true
        try {
            val out = app.processor.reprocess(row, style, asClass)
            if (out?.record?.error != null) _message.value = "Reprocessing failed: ${out.record.error}"
        } finally {
            _busy.value = false
        }
    }

    /** Deletes the cleaned copy (never the original). Returns false if the system must confirm. */
    fun deleteOutput(row: ProcessedImage, onNeedsConfirmation: (Uri) -> Unit) = viewModelScope.launch {
        val uri = row.outputUri?.let(Uri::parse) ?: return@launch
        try {
            app.media.delete(uri)
            app.db.processed().clearOutput(row.sourceId)
            _screen.value = Screen.Home
        } catch (e: SecurityException) {
            onNeedsConfirmation(uri)
        }
    }

    fun outputDeletedExternally(row: ProcessedImage) = viewModelScope.launch {
        app.db.processed().clearOutput(row.sourceId)
        _screen.value = Screen.Home
    }

    suspend fun debugImages(row: ProcessedImage) = app.processor.debugImages(Uri.parse(row.sourceUri))
}
