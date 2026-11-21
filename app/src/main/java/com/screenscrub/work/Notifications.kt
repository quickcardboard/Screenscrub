package com.screenscrub.work

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.ClipData
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import com.screenscrub.R
import com.screenscrub.ScreenScrubApp
import com.screenscrub.data.ProcessedImage
import com.screenscrub.ui.MainActivity
import com.screenscrub.ui.ShareActivity

object Notifications {
    private const val CHANNEL = "cleaned"
    private const val ID = 1

    fun createChannel(context: Context) {
        val ch = NotificationChannel(CHANNEL, context.getString(R.string.channel_name), NotificationManager.IMPORTANCE_DEFAULT)
        ch.description = context.getString(R.string.channel_description)
        ch.setSound(null, null)
        context.getSystemService(NotificationManager::class.java).createNotificationChannel(ch)
    }

    fun showCleaned(context: Context, rows: List<ProcessedImage>) {
        if (Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) return
        val latest = rows.last()
        val uris = rows.mapNotNull { it.outputUri?.let(Uri::parse) }
        val title = when {
            rows.size > 1 -> "${rows.size} photos cleaned"
            latest.imageClass == "DOCUMENT" -> "Document photo cleaned"
            else -> "Screen photo cleaned"
        }
        val open = PendingIntent.getActivity(
            context, 0,
            Intent(context, MainActivity::class.java).putExtra(MainActivity.EXTRA_OPEN_ID, latest.sourceId)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val share = PendingIntent.getActivity(
            context, 1, shareIntent(context, uris, rows.mapNotNull { it.outputMime }),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val app = context.applicationContext as ScreenScrubApp
        val thumb = uris.lastOrNull()?.let { app.media.thumbnail(it, 256) }
        val n = NotificationCompat.Builder(context, CHANNEL)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(title)
            .setContentText("Saved to Pictures/ScreenScrub. Tap Share to send it.")
            .setLargeIcon(thumb)
            .setContentIntent(open)
            .setAutoCancel(true)
            .addAction(0, if (uris.size > 1) "Share all" else "Share", share)
            .build()
        runCatching { NotificationManagerCompat.from(context).notify(ID, n) }
    }

    /** Share-sheet intent for cleaned outputs. Excludes our own "Clean & share" target. */
    fun shareIntent(context: Context, uris: List<Uri>, mimes: List<String>): Intent {
        val type = mimes.distinct().singleOrNull() ?: "image/*"
        val send = if (uris.size == 1) {
            Intent(Intent.ACTION_SEND).putExtra(Intent.EXTRA_STREAM, uris[0])
        } else {
            Intent(Intent.ACTION_SEND_MULTIPLE).putParcelableArrayListExtra(Intent.EXTRA_STREAM, ArrayList(uris))
        }
        send.type = type
        send.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        if (uris.isNotEmpty()) {
            val clip = ClipData.newRawUri(null, uris[0])
            uris.drop(1).forEach { clip.addItem(ClipData.Item(it)) }
            send.clipData = clip
        }
        return Intent.createChooser(send, null)
            .putExtra(Intent.EXTRA_EXCLUDE_COMPONENTS, arrayOf(ComponentName(context, ShareActivity::class.java)))
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
}
