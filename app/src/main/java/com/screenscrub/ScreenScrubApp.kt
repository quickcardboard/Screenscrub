package com.screenscrub

import android.app.Application
import android.util.Log
import com.screenscrub.data.AppDatabase
import com.screenscrub.data.SettingsStore
import com.screenscrub.media.MediaRepository
import com.screenscrub.work.Notifications
import com.screenscrub.work.Processor
import com.screenscrub.work.Scheduler
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import org.opencv.android.OpenCVLoader

/** Holds the app-wide singletons (a tiny service locator; no DI framework needed). */
class ScreenScrubApp : Application() {
    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    val db by lazy { AppDatabase.create(this) }
    val settings by lazy { SettingsStore(this) }
    val media by lazy { MediaRepository(this) }
    val processor by lazy { Processor(media, db.processed(), settings) }

    override fun onCreate() {
        super.onCreate()
        if (!OpenCVLoader.initLocal()) Log.e("ScreenScrub", "OpenCV failed to load")
        Notifications.createChannel(this)
        scope.launch {
            val s = settings.current()
            if (s.autoEnabled) {
                Scheduler.arm(this@ScreenScrubApp, s.chargingOnly)
                Scheduler.catchUp(this@ScreenScrubApp)
            }
        }
    }
}
