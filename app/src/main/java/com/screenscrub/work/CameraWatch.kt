package com.screenscrub.work

import android.content.Context
import android.provider.MediaStore
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import com.screenscrub.ScreenScrubApp
import com.screenscrub.data.Origin
import com.screenscrub.media.PhotoAccess
import java.time.Duration

/**
 * Wakes when MediaStore images change, cleans new camera photos, then re-arms itself.
 * Content-URI triggers are one-shot by design, hence the re-arm at the end of every run.
 */
class CameraWatchWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {

    override suspend fun doWork(): Result {
        val app = applicationContext as ScreenScrubApp
        val rearm = inputData.getBoolean(KEY_REARM, false)
        val s = app.settings.current()
        if (!s.autoEnabled) return Result.success()
        try {
            if (app.media.access() == PhotoAccess.FULL) {
                val scan = app.media.newCameraImages(s.lastGeneration, maxOf(s.lastDateAddedSec, s.autoEnabledAt / 1000))
                val cleaned = mutableListOf<com.screenscrub.data.ProcessedImage>()
                for (img in scan.images) {
                    if (img.dateAddedSec * 1000 < s.autoEnabledAt - 60_000) continue
                    val out = app.processor.process(img, Origin.AUTO, skipIfKnown = true) ?: continue
                    if (out.record.outputUri != null) cleaned += out.record
                }
                app.settings.setCursor(scan.maxGeneration, scan.maxDateAddedSec)
                if (cleaned.isNotEmpty() && s.notify) Notifications.showCleaned(applicationContext, cleaned)
            }
        } finally {
            if (rearm) Scheduler.arm(applicationContext, s.chargingOnly, append = true)
        }
        return Result.success()
    }

    companion object {
        const val KEY_REARM = "rearm"
    }
}

object Scheduler {
    private const val WATCH = "camera-watch"
    private const val CATCH_UP = "camera-catch-up"

    /** Waits for the next MediaStore change. */
    fun arm(context: Context, chargingOnly: Boolean, append: Boolean = false, replace: Boolean = false) {
        val constraints = Constraints.Builder()
            .addContentUriTrigger(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, true)
            // Fire quickly: the user may share the photo seconds after taking it.
            .setTriggerContentUpdateDelay(Duration.ofMillis(500))
            .setTriggerContentMaxDelay(Duration.ofSeconds(3))
            .setRequiresCharging(chargingOnly)
            .build()
        val req = OneTimeWorkRequestBuilder<CameraWatchWorker>()
            .setConstraints(constraints)
            .setInputData(workDataOf(CameraWatchWorker.KEY_REARM to true))
            .build()
        val policy = when {
            append -> ExistingWorkPolicy.APPEND_OR_REPLACE
            replace -> ExistingWorkPolicy.REPLACE
            else -> ExistingWorkPolicy.KEEP
        }
        WorkManager.getInstance(context).enqueueUniqueWork(WATCH, policy, req)
    }

    /** Scans now (app opened, settings changed, foreground observer). Does not re-arm. */
    fun catchUp(context: Context) {
        val req = OneTimeWorkRequestBuilder<CameraWatchWorker>().build()
        WorkManager.getInstance(context).enqueueUniqueWork(CATCH_UP, ExistingWorkPolicy.KEEP, req)
    }

    fun disarm(context: Context) {
        WorkManager.getInstance(context).cancelUniqueWork(WATCH)
    }
}
