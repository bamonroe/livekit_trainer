package com.bam.livekittrainer

import android.content.ContentValues
import android.content.Context
import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import org.json.JSONArray
import org.json.JSONException

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

    fun addClip(clip: ClipRecord) {
        dbHelper.writableDatabase.insertWithOnConflict(
            TABLE_CLIPS,
            null,
            clip.toContentValues(),
            SQLiteDatabase.CONFLICT_REPLACE,
        )
    }

    fun deleteClip(clip: ClipRecord) {
        dbHelper.writableDatabase.delete(TABLE_CLIPS, "id = ?", arrayOf(clip.id))
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

    fun advancePrompt(projectId: String, promptCount: Int) {
        if (promptCount <= 0) return
        val next = (promptIndex(projectId, promptCount) + 1) % promptCount
        dbHelper.writableDatabase.insertWithOnConflict(
            TABLE_PROMPT_STATE,
            null,
            ContentValues().apply {
                put("project_id", projectId)
                put("prompt_index", next)
            },
            SQLiteDatabase.CONFLICT_REPLACE,
        )
    }

    private fun migrateLegacyPreferences() {
        if (legacyPrefs.getBoolean(KEY_MIGRATED, false)) return

        val rawProjects = legacyPrefs.getString(KEY_PROJECTS, "[]") ?: "[]"
        val db = dbHelper.writableDatabase
        db.beginTransaction()
        try {
            val projects = JSONArray(rawProjects)
            for (index in 0 until projects.length()) {
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
            }
            db.setTransactionSuccessful()
            legacyPrefs.edit().putBoolean(KEY_MIGRATED, true).apply()
        } catch (_: JSONException) {
            legacyPrefs.edit().putBoolean(KEY_MIGRATED, true).apply()
        } finally {
            db.endTransaction()
        }
    }

    private fun migrateLegacyClips(db: SQLiteDatabase, projectId: String) {
        val rawClips = legacyPrefs.getString(legacyClipsKey(projectId), "[]") ?: "[]"
        val clips = JSONArray(rawClips)
        for (index in 0 until clips.length()) {
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
            )
            db.insertWithOnConflict(
                TABLE_CLIPS,
                null,
                clip.toContentValues(),
                SQLiteDatabase.CONFLICT_IGNORE,
            )
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
                    FOREIGN KEY(project_id) REFERENCES $TABLE_PROJECTS(id) ON DELETE CASCADE
                )
                """.trimIndent(),
            )
            db.execSQL(
                """
                CREATE TABLE $TABLE_PROMPT_STATE (
                    project_id TEXT PRIMARY KEY,
                    prompt_index INTEGER NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES $TABLE_PROJECTS(id) ON DELETE CASCADE
                )
                """.trimIndent(),
            )
            db.execSQL("CREATE INDEX clips_project_id_idx ON $TABLE_CLIPS(project_id)")
        }

        override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
            db.execSQL("DROP TABLE IF EXISTS $TABLE_PROMPT_STATE")
            db.execSQL("DROP TABLE IF EXISTS $TABLE_CLIPS")
            db.execSQL("DROP TABLE IF EXISTS $TABLE_PROJECTS")
            onCreate(db)
        }

        override fun onConfigure(db: SQLiteDatabase) {
            super.onConfigure(db)
            db.setForeignKeyConstraintsEnabled(true)
        }
    }

    private companion object {
        const val DATABASE_NAME = "wake_word_collection.db"
        const val DATABASE_VERSION = 1
        const val LEGACY_PREFS = "wake_word_projects"
        const val KEY_PROJECTS = "projects"
        const val KEY_MIGRATED = "sqlite_migrated"
        const val TABLE_PROJECTS = "projects"
        const val TABLE_CLIPS = "clips"
        const val TABLE_PROMPT_STATE = "prompt_state"

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
    }

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
    )

private fun Cursor.getStringValue(column: String): String =
    getString(getColumnIndexOrThrow(column))

private fun Cursor.getIntValue(column: String): Int =
    getInt(getColumnIndexOrThrow(column))

private fun Cursor.getLongValue(column: String): Long =
    getLong(getColumnIndexOrThrow(column))
