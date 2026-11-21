package com.screenscrub.data

import android.content.Context
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.floatPreferencesKey
import androidx.datastore.preferences.core.longPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.screenscrub.vision.OutputStyle
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map

data class AppSettings(
    val autoEnabled: Boolean = false,
    /** Only camera photos added after this moment are auto-processed (ms since epoch). */
    val autoEnabledAt: Long = 0,
    val screenStyle: OutputStyle = OutputStyle.AUTO,
    val documentStyle: OutputStyle = OutputStyle.AUTO,
    /** -1 strict .. +1 eager. */
    val sensitivity: Float = 0f,
    val chargingOnly: Boolean = false,
    val notify: Boolean = true,
    /** MediaStore scan cursors. */
    val lastGeneration: Long = -1,
    val lastDateAddedSec: Long = 0,
)

private val Context.dataStore by preferencesDataStore("settings")

class SettingsStore(private val context: Context) {
    private object K {
        val auto = booleanPreferencesKey("auto")
        val autoAt = longPreferencesKey("auto_at")
        val screenStyle = stringPreferencesKey("screen_style")
        val docStyle = stringPreferencesKey("doc_style")
        val sensitivity = floatPreferencesKey("sensitivity")
        val charging = booleanPreferencesKey("charging_only")
        val notify = booleanPreferencesKey("notify")
        val generation = longPreferencesKey("last_generation")
        val dateAdded = longPreferencesKey("last_date_added")
    }

    val flow: Flow<AppSettings> = context.dataStore.data.map { it.toSettings() }

    suspend fun current(): AppSettings = flow.first()

    private fun Preferences.toSettings() = AppSettings(
        autoEnabled = this[K.auto] ?: false,
        autoEnabledAt = this[K.autoAt] ?: 0,
        screenStyle = style(this[K.screenStyle]),
        documentStyle = style(this[K.docStyle]),
        sensitivity = this[K.sensitivity] ?: 0f,
        chargingOnly = this[K.charging] ?: false,
        notify = this[K.notify] ?: true,
        lastGeneration = this[K.generation] ?: -1,
        lastDateAddedSec = this[K.dateAdded] ?: 0,
    )

    private fun style(s: String?) = s?.let { runCatching { OutputStyle.valueOf(it) }.getOrNull() } ?: OutputStyle.AUTO

    suspend fun setAuto(enabled: Boolean) = context.dataStore.edit {
        val wasOn = it[K.auto] ?: false
        it[K.auto] = enabled
        if (enabled && !wasOn) {
            // Start from "now": don't chew through the existing library.
            it[K.autoAt] = System.currentTimeMillis()
            it.remove(K.generation)
            it[K.dateAdded] = System.currentTimeMillis() / 1000
        }
    }

    suspend fun setScreenStyle(s: OutputStyle) = context.dataStore.edit { it[K.screenStyle] = s.name }
    suspend fun setDocumentStyle(s: OutputStyle) = context.dataStore.edit { it[K.docStyle] = s.name }
    suspend fun setSensitivity(v: Float) = context.dataStore.edit { it[K.sensitivity] = v }
    suspend fun setChargingOnly(v: Boolean) = context.dataStore.edit { it[K.charging] = v }
    suspend fun setNotify(v: Boolean) = context.dataStore.edit { it[K.notify] = v }

    suspend fun setCursor(generation: Long, dateAddedSec: Long) = context.dataStore.edit {
        it[K.generation] = generation
        it[K.dateAdded] = dateAddedSec
    }
}
