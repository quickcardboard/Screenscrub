package com.screenscrub.data

import android.content.Context
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Entity
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.Upsert
import kotlinx.coroutines.flow.Flow

/** How an image reached the pipeline. */
enum class Origin { AUTO, MANUAL, SHARE }

/**
 * One row per analysed source image, including those classified OTHER, so nothing is
 * analysed twice. [outputUri] is null when nothing was saved.
 */
@Entity(tableName = "processed")
data class ProcessedImage(
    /** MediaStore ID for camera images; a negative hash of the URI for picked/shared images. */
    @PrimaryKey val sourceId: Long,
    val sourceUri: String,
    val sourceName: String,
    val outputUri: String?,
    val outputMime: String?,
    val imageClass: String,
    val screenScore: Double,
    val documentScore: Double,
    /** Feature name → value, as "k=v;k=v" (kept simple and readable in the debug screen). */
    val features: String,
    val style: String?,
    val sourceBytes: Long,
    val outputBytes: Long,
    val outputWidth: Int,
    val outputHeight: Int,
    val durationMs: Long,
    val processedAt: Long,
    val origin: String,
    val error: String?,
) {
    fun featureMap(): Map<String, String> = features.split(';').filter { '=' in it }
        .associate { it.substringBefore('=') to it.substringAfter('=') }
}

@Dao
interface ProcessedDao {
    @Query("SELECT * FROM processed WHERE outputUri IS NOT NULL ORDER BY processedAt DESC")
    fun results(): Flow<List<ProcessedImage>>

    @Query("SELECT * FROM processed WHERE sourceId = :id")
    fun observe(id: Long): Flow<ProcessedImage?>

    @Query("SELECT * FROM processed WHERE sourceId = :id")
    suspend fun get(id: Long): ProcessedImage?

    @Query("SELECT EXISTS(SELECT 1 FROM processed WHERE sourceId = :id)")
    suspend fun exists(id: Long): Boolean

    @Upsert
    suspend fun upsert(row: ProcessedImage)

    @Query("UPDATE processed SET outputUri = NULL, outputMime = NULL, outputBytes = 0 WHERE sourceId = :id")
    suspend fun clearOutput(id: Long)

    @Query("SELECT * FROM processed WHERE outputUri = :uri LIMIT 1")
    suspend fun byOutput(uri: String): ProcessedImage?
}

@Database(entities = [ProcessedImage::class], version = 1, exportSchema = true)
abstract class AppDatabase : RoomDatabase() {
    abstract fun processed(): ProcessedDao

    companion object {
        fun create(context: Context): AppDatabase =
            Room.databaseBuilder(context, AppDatabase::class.java, "screenscrub.db").build()
    }
}
