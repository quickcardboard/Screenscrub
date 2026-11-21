package com.screenscrub.ui

import android.content.Intent
import android.database.ContentObserver
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.screenscrub.work.Scheduler

class MainActivity : ComponentActivity() {
    private val vm: MainViewModel by viewModels()
    private val handler = Handler(Looper.getMainLooper())
    private val catchUp = Runnable { if (vm.settings.value.autoEnabled) Scheduler.catchUp(this) }

    /** While the app is open, react to new photos immediately instead of waiting for WorkManager. */
    private val observer = object : ContentObserver(handler) {
        override fun onChange(selfChange: Boolean) {
            handler.removeCallbacks(catchUp)
            handler.postDelayed(catchUp, 800)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        handleIntent(intent)
        setContent {
            ScreenScrubTheme {
                val screen by vm.screen.collectAsStateWithLifecycle()
                BackHandler(enabled = screen != Screen.Home) { vm.back() }
                when (val s = screen) {
                    Screen.Home -> HomeScreen(vm)
                    is Screen.Detail -> DetailScreen(vm, s.id)
                    Screen.Settings -> SettingsScreen(vm)
                }
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleIntent(intent)
    }

    private fun handleIntent(intent: Intent?) {
        val id = intent?.getLongExtra(EXTRA_OPEN_ID, Long.MIN_VALUE) ?: return
        if (id != Long.MIN_VALUE) vm.navigate(Screen.Detail(id))
    }

    override fun onStart() {
        super.onStart()
        vm.refreshAccess()
        contentResolver.registerContentObserver(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, true, observer)
    }

    override fun onStop() {
        contentResolver.unregisterContentObserver(observer)
        super.onStop()
    }

    companion object {
        const val EXTRA_OPEN_ID = "open_id"
    }
}

@Composable
fun ScreenScrubTheme(content: @Composable () -> Unit) {
    val dark = isSystemInDarkTheme()
    val ctx = LocalContext.current
    val scheme = when {
        android.os.Build.VERSION.SDK_INT >= 31 -> if (dark) dynamicDarkColorScheme(ctx) else dynamicLightColorScheme(ctx)
        dark -> darkColorScheme(primary = Color(0xFF9DB4FF))
        else -> lightColorScheme(primary = Color(0xFF1E4FD8))
    }
    MaterialTheme(colorScheme = scheme, content = content)
}
