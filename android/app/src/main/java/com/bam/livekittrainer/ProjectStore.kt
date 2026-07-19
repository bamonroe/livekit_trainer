package com.bam.livekittrainer

import android.content.ContentValues
import android.content.Context
import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import org.json.JSONArray
import org.json.JSONException
import java.io.File

class ProjectStore(context: Context) {
    private val appContext = context.applicationContext
    private val dbHelper = StoreDbHelper(appContext)
    private val legacyPrefs = appContext.getSharedPreferences(LEGACY_PREFS, Context.MODE_PRIVATE)

    init {
        migrateLegacyPreferences()
    }

    fun loadProjects(): List<WakeWordProject> {
        val db = dbHelper.readableDatabase
        return db.query(
            TABLE_PROJECTS,
            null,
            null,
            null,
            null,
            null,
            "created_at_millis DESC",
        ).use { cursor ->
            buildList {
                while (cursor.moveToNext()) {
                    add(cursor.toProject())
                }
            }
        }
    }

    fun addProject(project: WakeWordProject) {
        dbHelper.writableDatabase.insertWithOnConflict(
            TABLE_PROJECTS,
            null,
            project.toContentValues(),
            SQLiteDatabase.CONFLICT_REPLACE,
        )
    }

    fun loadClips(projectId: String): List<ClipRecord> {
        val db = dbHelper.readableDatabase
        return db.query(
            TABLE_CLIPS,
            null,
            "project_id = ?",
            arrayOf(projectId),
            null,
            null,
            "recorded_at_millis DESC",
        ).use { cursor ->
            buildList {
                while (cursor.moveToNext()) {
                    add(cursor.toClip())
                }
            }
        }
    }

    fun loadBulkRecordings(projectId: String): List<BulkRecording> {
        val db = dbHelper.readableDatabase
        return db.query(
            TABLE_BULK_RECORDINGS,
            null,
            "project_id = ?",
            arrayOf(projectId),
            null,
            null,
            "recorded_at_millis DESC",
        ).use { cursor ->
            buildList {
                while (cursor.moveToNext()) {
                    add(cursor.toBulkRecording())
                }
            }
        }
    }

    fun addClip(clip: ClipRecord) {
        dbHelper.writableDatabase.insertWithOnConflict(
            TABLE_CLIPS,
            null,
            clip.toContentValues(),
            SQLiteDatabase.CONFLICT_REPLACE,
        )
    }

    fun addBulkRecording(recording: BulkRecording) {
        dbHelper.writableDatabase.insertWithOnConflict(
            TABLE_BULK_RECORDINGS,
            null,
            recording.toContentValues(),
            SQLiteDatabase.CONFLICT_REPLACE,
        )
    }

    fun loadTestRecordings(projectId: String): List<BulkRecording> {
        val db = dbHelper.readableDatabase
        return db.query(
            TABLE_TEST_RECORDINGS,
            null,
            "project_id = ?",
            arrayOf(projectId),
            null,
            null,
            "recorded_at_millis DESC",
        ).use { cursor ->
            buildList {
                while (cursor.moveToNext()) {
                    add(cursor.toBulkRecording())
                }
            }
        }
    }

    fun addTestRecording(recording: BulkRecording) {
        dbHelper.writableDatabase.insertWithOnConflict(
            TABLE_TEST_RECORDINGS,
            null,
            recording.toContentValues(),
            SQLiteDatabase.CONFLICT_REPLACE,
        )
    }

    fun deleteTestRecording(recording: BulkRecording) {
        dbHelper.writableDatabase.delete(TABLE_TEST_RECORDINGS, "id = ?", arrayOf(recording.id))
    }

    fun loadBackgroundRecordings(projectId: String): List<BackgroundRecording> {
        val db = dbHelper.readableDatabase
        return db.query(
            TABLE_BACKGROUND_RECORDINGS,
            null,
            "project_id = ?",
            arrayOf(projectId),
            null,
            null,
            "recorded_at_millis DESC",
        ).use { cursor ->
            buildList {
                while (cursor.moveToNext()) {
                    add(cursor.toBackgroundRecording())
                }
            }
        }
    }

    fun addBackgroundRecording(recording: BackgroundRecording) {
        dbHelper.writableDatabase.insertWithOnConflict(
            TABLE_BACKGROUND_RECORDINGS,
            null,
            recording.toContentValues(),
            SQLiteDatabase.CONFLICT_REPLACE,
        )
    }

    fun deleteBackgroundRecording(recording: BackgroundRecording) {
        dbHelper.writableDatabase.delete(
            TABLE_BACKGROUND_RECORDINGS,
            "id = ?",
            arrayOf(recording.id),
        )
    }

    fun deleteClip(clip: ClipRecord) {
        dbHelper.writableDatabase.delete(TABLE_CLIPS, "id = ?", arrayOf(clip.id))
    }

    fun deleteBulkRecording(recording: BulkRecording) {
        dbHelper.writableDatabase.delete(TABLE_BULK_RECORDINGS, "id = ?", arrayOf(recording.id))
    }

    fun resetAllData(): Int {
        val clips = loadAllClips()
        clips.forEach { clip ->
            File(clip.filePath).delete()
        }

        File(appContext.filesDir, "clips").deleteRecursively()
        File(appContext.filesDir, "bulk").deleteRecursively()
        File(appContext.filesDir, "background").deleteRecursively()
        File(appContext.filesDir, "test").deleteRecursively()
        File(appContext.filesDir, "exports").deleteRecursively()
        File(appContext.cacheDir, "sync").deleteRecursively()

        val db = dbHelper.writableDatabase
        db.beginTransaction()
        try {
            db.delete(TABLE_PROMPT_STATE, null, null)
            db.delete(TABLE_TEST_RECORDINGS, null, null)
            db.delete(TABLE_BACKGROUND_RECORDINGS, null, null)
            db.delete(TABLE_BULK_RECORDINGS, null, null)
            db.delete(TABLE_CLIPS, null, null)
            db.delete(TABLE_PROJECTS, null, null)
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }

        legacyPrefs.edit()
            .clear()
            .putBoolean(KEY_MIGRATED, true)
            .apply()

        return clips.size
    }

    private fun loadAllClips(): List<ClipRecord> {
        val db = dbHelper.readableDatabase
        return db.query(
            TABLE_CLIPS,
            null,
            null,
            null,
            null,
            null,
            null,
        ).use { cursor ->
            buildList {
                while (cursor.moveToNext()) {
                    add(cursor.toClip())
                }
            }
        }
    }

    fun promptIndex(projectId: String, promptCount: Int): Int {
        if (promptCount <= 0) return 0

        val db = dbHelper.readableDatabase
        val index = db.query(
            TABLE_PROMPT_STATE,
            arrayOf("prompt_index"),
            "project_id = ?",
            arrayOf(projectId),
            null,
            null,
            null,
        ).use { cursor ->
            if (cursor.moveToFirst()) cursor.getIntValue("prompt_index") else 0
        }

        return index.coerceIn(0, promptCount - 1)
    }

    fun promptBatch(projectId: String): Int {
        val db = dbHelper.readableDatabase
        return db.query(
            TABLE_PROMPT_STATE,
            arrayOf("prompt_batch"),
            "project_id = ?",
            arrayOf(projectId),
            null,
            null,
            null,
        ).use { cursor ->
            if (cursor.moveToFirst()) cursor.getIntValue("prompt_batch") else 0
        }.coerceAtLeast(0)
    }

    fun advancePrompt(projectId: String, promptCount: Int) {
        if (promptCount <= 0) return
        val currentIndex = promptIndex(projectId, promptCount)
        val currentBatch = promptBatch(projectId)
        if (currentIndex + 1 >= promptCount) {
            setPromptState(projectId, promptIndex = 0, promptBatch = currentBatch + 1)
        } else {
            setPromptState(projectId, promptIndex = currentIndex + 1, promptBatch = currentBatch)
        }
    }

    fun setPromptIndex(projectId: String, promptIndex: Int) {
        setPromptState(projectId, promptIndex, promptBatch(projectId))
    }

    private fun setPromptState(projectId: String, promptIndex: Int, promptBatch: Int) {
        dbHelper.writableDatabase.insertWithOnConflict(
            TABLE_PROMPT_STATE,
            null,
            ContentValues().apply {
                put("project_id", projectId)
                put("prompt_index", promptIndex.coerceAtLeast(0))
                put("prompt_batch", promptBatch.coerceAtLeast(0))
            },
            SQLiteDatabase.CONFLICT_REPLACE,
        )
    }

    private fun migrateLegacyPreferences() {
        if (legacyPrefs.getBoolean(KEY_MIGRATED, false)) return

        val projects = try {
            JSONArray(legacyPrefs.getString(KEY_PROJECTS, "[]") ?: "[]")
        } catch (_: JSONException) {
            // Nothing recoverable from a malformed top-level list; don't retry.
            legacyPrefs.edit().putBoolean(KEY_MIGRATED, true).apply()
            return
        }

        val db = dbHelper.writableDatabase
        db.beginTransaction()
        try {
            for (index in 0 until projects.length()) {
                try {
                    val item = projects.getJSONObject(index)
                    val project = WakeWordProject(
                        id = item.getString("id"),
                        phrase = item.getString("phrase"),
                        slug = item.getString("slug"),
                        createdAtMillis = item.getLong("created_at_millis"),
                    )
                    db.insertWithOnConflict(
                        TABLE_PROJECTS,
                        null,
                        project.toContentValues(),
                        SQLiteDatabase.CONFLICT_IGNORE,
                    )
                    migrateLegacyClips(db, project.id)
                    migrateLegacyPromptIndex(db, project.id)
                } catch (_: JSONException) {
                    // Skip a single malformed project and keep migrating the rest.
                }
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
        // Only mark migration complete after the good records are committed.
        legacyPrefs.edit().putBoolean(KEY_MIGRATED, true).apply()
    }

    private fun migrateLegacyClips(db: SQLiteDatabase, projectId: String) {
        val rawClips = legacyPrefs.getString(legacyClipsKey(projectId), "[]") ?: "[]"
        val clips = JSONArray(rawClips)
        for (index in 0 until clips.length()) {
            try {
                val item = clips.getJSONObject(index)
                val clip = ClipRecord(
                    id = item.getString("id"),
                    projectId = item.getString("project_id"),
                    projectSlug = item.getString("project_slug"),
                    filePath = item.getString("file_path"),
                    label = ClipLabel.valueOf(item.getString("label")),
                    prompt = item.getString("prompt"),
                    spokenPhrase = item.getString("spoken_phrase"),
                    recordedAtMillis = item.getLong("recorded_at_millis"),
                    durationMs = item.getLong("duration_ms"),
                    sampleRateHz = item.getInt("sample_rate_hz"),
                    channels = item.getInt("channels"),
                    encoding = item.getString("encoding"),
                    conditions = emptyList(),
                )
                db.insertWithOnConflict(
                    TABLE_CLIPS,
                    null,
                    clip.toContentValues(),
                    SQLiteDatabase.CONFLICT_IGNORE,
                )
            } catch (_: JSONException) {
                // Skip a single malformed clip.
            } catch (_: IllegalArgumentException) {
                // Skip a clip with an unknown label enum.
            }
        }
    }

    private fun migrateLegacyPromptIndex(db: SQLiteDatabase, projectId: String) {
        if (!legacyPrefs.contains(legacyPromptIndexKey(projectId))) return
        db.insertWithOnConflict(
            TABLE_PROMPT_STATE,
            null,
            ContentValues().apply {
                put("project_id", projectId)
                put("prompt_index", legacyPrefs.getInt(legacyPromptIndexKey(projectId), 0))
            },
            SQLiteDatabase.CONFLICT_IGNORE,
        )
    }

    private class StoreDbHelper(context: Context) :
        SQLiteOpenHelper(context, DATABASE_NAME, null, DATABASE_VERSION) {
        override fun onCreate(db: SQLiteDatabase) {
            db.execSQL(
                """
                CREATE TABLE $TABLE_PROJECTS (
                    id TEXT PRIMARY KEY,
                    phrase TEXT NOT NULL,
                    slug TEXT NOT NULL,
                    created_at_millis INTEGER NOT NULL
                )
                """.trimIndent(),
            )
            db.execSQL(
                """
                CREATE TABLE $TABLE_CLIPS (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    project_slug TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    label TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    spoken_phrase TEXT NOT NULL,
                    recorded_at_millis INTEGER NOT NULL,
                    duration_ms INTEGER NOT NULL,
                    sample_rate_hz INTEGER NOT NULL,
                    channels INTEGER NOT NULL,
                    encoding TEXT NOT NULL,
                    condition_tags TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(project_id) REFERENCES $TABLE_PROJECTS(id) ON DELETE CASCADE
                )
                """.trimIndent(),
            )
            db.execSQL(
                """
                CREATE TABLE $TABLE_PROMPT_STATE (
                    project_id TEXT PRIMARY KEY,
                    prompt_index INTEGER NOT NULL,
                    prompt_batch INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(project_id) REFERENCES $TABLE_PROJECTS(id) ON DELETE CASCADE
                )
                """.trimIndent(),
            )
            createBulkRecordingsTable(db)
            createBackgroundRecordingsTable(db)
            createTestRecordingsTable(db)
            db.execSQL("CREATE INDEX clips_project_id_idx ON $TABLE_CLIPS(project_id)")
        }

        override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
            if (oldVersion < 2) {
                db.execSQL("ALTER TABLE $TABLE_CLIPS ADD COLUMN condition_tags TEXT NOT NULL DEFAULT ''")
            }
            if (oldVersion < 3) {
                db.execSQL("ALTER TABLE $TABLE_PROMPT_STATE ADD COLUMN prompt_batch INTEGER NOT NULL DEFAULT 0")
            }
            if (oldVersion < 4) {
                createBulkRecordingsTable(db)
            }
            if (oldVersion < 5) {
                createBackgroundRecordingsTable(db)
            }
            if (oldVersion < 6) {
                addCaptureColumns(db, TABLE_BULK_RECORDINGS)
                addCaptureColumns(db, TABLE_BACKGROUND_RECORDINGS)
            }
            if (oldVersion < 7) {
                createTestRecordingsTable(db)
            }
        }

        override fun onConfigure(db: SQLiteDatabase) {
            super.onConfigure(db)
            db.setForeignKeyConstraintsEnabled(true)
        }

        private fun createBulkRecordingsTable(db: SQLiteDatabase) {
            db.execSQL(
                """
                CREATE TABLE IF NOT EXISTS $TABLE_BULK_RECORDINGS (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    project_slug TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    script TEXT NOT NULL,
                    recorded_at_millis INTEGER NOT NULL,
                    duration_ms INTEGER NOT NULL,
                    sample_rate_hz INTEGER NOT NULL,
                    channels INTEGER NOT NULL,
                    encoding TEXT NOT NULL,
                    condition_tags TEXT NOT NULL DEFAULT '',
                    $CAPTURE_COLUMNS_SQL,
                    FOREIGN KEY(project_id) REFERENCES $TABLE_PROJECTS(id) ON DELETE CASCADE
                )
                """.trimIndent(),
            )
            db.execSQL("CREATE INDEX IF NOT EXISTS bulk_recordings_project_id_idx ON $TABLE_BULK_RECORDINGS(project_id)")
        }

        private fun createBackgroundRecordingsTable(db: SQLiteDatabase) {
            db.execSQL(
                """
                CREATE TABLE IF NOT EXISTS $TABLE_BACKGROUND_RECORDINGS (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    project_slug TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    recorded_at_millis INTEGER NOT NULL,
                    duration_ms INTEGER NOT NULL,
                    sample_rate_hz INTEGER NOT NULL,
                    channels INTEGER NOT NULL,
                    encoding TEXT NOT NULL,
                    $CAPTURE_COLUMNS_SQL,
                    FOREIGN KEY(project_id) REFERENCES $TABLE_PROJECTS(id) ON DELETE CASCADE
                )
                """.trimIndent(),
            )
            db.execSQL("CREATE INDEX IF NOT EXISTS background_recordings_project_id_idx ON $TABLE_BACKGROUND_RECORDINGS(project_id)")
        }

        // Test takes share the bulk-recording shape (they carry a script) but live
        // in their own table so they can never be swept into a training upload.
        private fun createTestRecordingsTable(db: SQLiteDatabase) {
            db.execSQL(
                """
                CREATE TABLE IF NOT EXISTS $TABLE_TEST_RECORDINGS (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    project_slug TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    script TEXT NOT NULL,
                    recorded_at_millis INTEGER NOT NULL,
                    duration_ms INTEGER NOT NULL,
                    sample_rate_hz INTEGER NOT NULL,
                    channels INTEGER NOT NULL,
                    encoding TEXT NOT NULL,
                    condition_tags TEXT NOT NULL DEFAULT '',
                    $CAPTURE_COLUMNS_SQL,
                    FOREIGN KEY(project_id) REFERENCES $TABLE_PROJECTS(id) ON DELETE CASCADE
                )
                """.trimIndent(),
            )
            db.execSQL("CREATE INDEX IF NOT EXISTS test_recordings_project_id_idx ON $TABLE_TEST_RECORDINGS(project_id)")
        }

        private fun addCaptureColumns(db: SQLiteDatabase, table: String) {
            val existing = existingColumns(db, table)
            for ((name, type) in CAPTURE_COLUMNS) {
                if (name in existing) continue
                db.execSQL("ALTER TABLE $table ADD COLUMN $name $type")
            }
        }

        private fun existingColumns(db: SQLiteDatabase, table: String): Set<String> {
            val columns = mutableSetOf<String>()
            db.rawQuery("PRAGMA table_info($table)", null).use { cursor ->
                val nameIndex = cursor.getColumnIndex("name")
                while (cursor.moveToNext()) {
                    columns.add(cursor.getString(nameIndex))
                }
            }
            return columns
        }
    }

    private companion object {
        const val DATABASE_NAME = "wake_word_collection.db"
        const val DATABASE_VERSION = 7
        const val LEGACY_PREFS = "wake_word_projects"

        // Per-take provenance columns shared by the bulk and background tables.
        // Kept in one place so the CREATE definitions and the version-6 ALTER
        // migration stay in lockstep.
        val CAPTURE_COLUMNS: List<Pair<String, String>> = listOf(
            "capture_device_manufacturer" to "TEXT NOT NULL DEFAULT ''",
            "capture_device_model" to "TEXT NOT NULL DEFAULT ''",
            "capture_os_version" to "TEXT NOT NULL DEFAULT ''",
            "capture_app_version" to "TEXT NOT NULL DEFAULT ''",
            "capture_input_route" to "TEXT NOT NULL DEFAULT ''",
            "capture_source_sample_rate_hz" to "INTEGER NOT NULL DEFAULT 0",
            "capture_source_channels" to "INTEGER NOT NULL DEFAULT 0",
            "capture_session_id" to "TEXT NOT NULL DEFAULT ''",
        )
        val CAPTURE_COLUMNS_SQL: String =
            CAPTURE_COLUMNS.joinToString(",\n                    ") { (name, type) -> "$name $type" }
        const val KEY_PROJECTS = "projects"
        const val KEY_MIGRATED = "sqlite_migrated"
        const val TABLE_PROJECTS = "projects"
        const val TABLE_CLIPS = "clips"
        const val TABLE_PROMPT_STATE = "prompt_state"
        const val TABLE_BULK_RECORDINGS = "bulk_recordings"
        const val TABLE_BACKGROUND_RECORDINGS = "background_recordings"
        const val TABLE_TEST_RECORDINGS = "test_recordings"

        fun legacyClipsKey(projectId: String): String = "clips_$projectId"

        fun legacyPromptIndexKey(projectId: String): String = "prompt_index_$projectId"
    }
}

private fun WakeWordProject.toContentValues(): ContentValues =
    ContentValues().apply {
        put("id", id)
        put("phrase", phrase)
        put("slug", slug)
        put("created_at_millis", createdAtMillis)
    }

private fun ClipRecord.toContentValues(): ContentValues =
    ContentValues().apply {
        put("id", id)
        put("project_id", projectId)
        put("project_slug", projectSlug)
        put("file_path", filePath)
        put("label", label.name)
        put("prompt", prompt)
        put("spoken_phrase", spokenPhrase)
        put("recorded_at_millis", recordedAtMillis)
        put("duration_ms", durationMs)
        put("sample_rate_hz", sampleRateHz)
        put("channels", channels)
        put("encoding", encoding)
        put("condition_tags", conditions.joinToString(",") { it.name })
    }

private fun BulkRecording.toContentValues(): ContentValues =
    ContentValues().apply {
        put("id", id)
        put("project_id", projectId)
        put("project_slug", projectSlug)
        put("file_path", filePath)
        put("script", script)
        put("recorded_at_millis", recordedAtMillis)
        put("duration_ms", durationMs)
        put("sample_rate_hz", sampleRateHz)
        put("channels", channels)
        put("encoding", encoding)
        put("condition_tags", conditions.joinToString(",") { it.name })
        putCapture(capture)
    }

private fun BackgroundRecording.toContentValues(): ContentValues =
    ContentValues().apply {
        put("id", id)
        put("project_id", projectId)
        put("project_slug", projectSlug)
        put("file_path", filePath)
        put("recorded_at_millis", recordedAtMillis)
        put("duration_ms", durationMs)
        put("sample_rate_hz", sampleRateHz)
        put("channels", channels)
        put("encoding", encoding)
        putCapture(capture)
    }

private fun ContentValues.putCapture(capture: CaptureMetadata) {
    put("capture_device_manufacturer", capture.deviceManufacturer)
    put("capture_device_model", capture.deviceModel)
    put("capture_os_version", capture.osVersion)
    put("capture_app_version", capture.appVersion)
    put("capture_input_route", capture.inputRoute)
    put("capture_source_sample_rate_hz", capture.sourceSampleRateHz)
    put("capture_source_channels", capture.sourceChannels)
    put("capture_session_id", capture.sessionId)
}

private fun Cursor.toCaptureMetadata(): CaptureMetadata =
    CaptureMetadata(
        deviceManufacturer = getOptionalStringValue("capture_device_manufacturer"),
        deviceModel = getOptionalStringValue("capture_device_model"),
        osVersion = getOptionalStringValue("capture_os_version"),
        appVersion = getOptionalStringValue("capture_app_version"),
        inputRoute = getOptionalStringValue("capture_input_route"),
        sourceSampleRateHz = getOptionalIntValue("capture_source_sample_rate_hz"),
        sourceChannels = getOptionalIntValue("capture_source_channels"),
        sessionId = getOptionalStringValue("capture_session_id"),
    )

private fun Cursor.toProject(): WakeWordProject =
    WakeWordProject(
        id = getStringValue("id"),
        phrase = getStringValue("phrase"),
        slug = getStringValue("slug"),
        createdAtMillis = getLongValue("created_at_millis"),
    )

private fun Cursor.toClip(): ClipRecord =
    ClipRecord(
        id = getStringValue("id"),
        projectId = getStringValue("project_id"),
        projectSlug = getStringValue("project_slug"),
        filePath = getStringValue("file_path"),
        label = ClipLabel.valueOf(getStringValue("label")),
        prompt = getStringValue("prompt"),
        spokenPhrase = getStringValue("spoken_phrase"),
        recordedAtMillis = getLongValue("recorded_at_millis"),
        durationMs = getLongValue("duration_ms"),
        sampleRateHz = getIntValue("sample_rate_hz"),
        channels = getIntValue("channels"),
    encoding = getStringValue("encoding"),
    conditions = getOptionalStringValue("condition_tags")
        .split(',')
        .mapNotNull { raw ->
            raw.takeIf { it.isNotBlank() }?.let { runCatching { ClipCondition.valueOf(it) }.getOrNull() }
        },
)

private fun Cursor.toBulkRecording(): BulkRecording =
    BulkRecording(
        id = getStringValue("id"),
        projectId = getStringValue("project_id"),
        projectSlug = getStringValue("project_slug"),
        filePath = getStringValue("file_path"),
        script = getStringValue("script"),
        recordedAtMillis = getLongValue("recorded_at_millis"),
        durationMs = getLongValue("duration_ms"),
        sampleRateHz = getIntValue("sample_rate_hz"),
        channels = getIntValue("channels"),
        encoding = getStringValue("encoding"),
        conditions = getOptionalStringValue("condition_tags")
            .split(',')
            .mapNotNull { raw ->
                raw.takeIf { it.isNotBlank() }?.let { runCatching { ClipCondition.valueOf(it) }.getOrNull() }
            },
        capture = toCaptureMetadata(),
    )

private fun Cursor.toBackgroundRecording(): BackgroundRecording =
    BackgroundRecording(
        id = getStringValue("id"),
        projectId = getStringValue("project_id"),
        projectSlug = getStringValue("project_slug"),
        filePath = getStringValue("file_path"),
        recordedAtMillis = getLongValue("recorded_at_millis"),
        durationMs = getLongValue("duration_ms"),
        sampleRateHz = getIntValue("sample_rate_hz"),
        channels = getIntValue("channels"),
        encoding = getStringValue("encoding"),
        capture = toCaptureMetadata(),
    )

private fun Cursor.getStringValue(column: String): String =
    getString(getColumnIndexOrThrow(column))

private fun Cursor.getOptionalStringValue(column: String): String {
    val index = getColumnIndex(column)
    return if (index >= 0) getString(index) ?: "" else ""
}

private fun Cursor.getIntValue(column: String): Int =
    getInt(getColumnIndexOrThrow(column))

private fun Cursor.getOptionalIntValue(column: String): Int {
    val index = getColumnIndex(column)
    return if (index >= 0) getInt(index) else 0
}

private fun Cursor.getLongValue(column: String): Long =
    getLong(getColumnIndexOrThrow(column))
