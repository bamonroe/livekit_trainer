package com.bam.livekittrainer

import android.content.Context
import android.os.Build
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.time.Instant
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

class BundleExporter(private val context: Context) {
    fun exportProject(
        project: WakeWordProject,
        clips: List<ClipRecord>,
        bulkRecordings: List<BulkRecording> = emptyList(),
        backgroundRecordings: List<BackgroundRecording> = emptyList(),
    ): File {
        val exportRoot = File(
            context.filesDir,
            "exports/${project.slug}_${System.currentTimeMillis()}",
        )
        val audioDir = File(exportRoot, "audio")
        val bulkAudioDir = File(exportRoot, "bulk_audio")
        val backgroundAudioDir = File(exportRoot, "background_audio")
        audioDir.mkdirs()
        bulkAudioDir.mkdirs()
        backgroundAudioDir.mkdirs()

        val exportedClips = JSONArray()
        clips.filter { File(it.filePath).isFile }.forEach { clip ->
            val source = File(clip.filePath)
            val exportedName = "${clip.id}.wav"
            val exportedFile = File(audioDir, exportedName)
            source.copyTo(exportedFile, overwrite = true)
            exportedClips.put(clip.toManifestJson("audio/$exportedName"))
        }

        val exportedBulkRecordings = JSONArray()
        bulkRecordings.filter { File(it.filePath).isFile }.forEach { recording ->
            val source = File(recording.filePath)
            val exportedName = "${recording.id}.wav"
            val exportedFile = File(bulkAudioDir, exportedName)
            source.copyTo(exportedFile, overwrite = true)
            exportedBulkRecordings.put(recording.toManifestJson("bulk_audio/$exportedName"))
        }

        val exportedBackgroundRecordings = JSONArray()
        backgroundRecordings.filter { File(it.filePath).isFile }.forEach { recording ->
            val source = File(recording.filePath)
            val exportedName = "${recording.id}.wav"
            val exportedFile = File(backgroundAudioDir, exportedName)
            source.copyTo(exportedFile, overwrite = true)
            exportedBackgroundRecordings.put(recording.toManifestJson("background_audio/$exportedName"))
        }

        File(exportRoot, "manifest.json").writeText(
            manifest(
                project,
                exportedClips,
                exportedBulkRecordings,
                exportedBackgroundRecordings,
            ).toString(2),
            Charsets.UTF_8,
        )
        return exportRoot
    }

    fun exportProjectZip(
        project: WakeWordProject,
        clips: List<ClipRecord>,
        bulkRecordings: List<BulkRecording> = emptyList(),
        backgroundRecordings: List<BackgroundRecording> = emptyList(),
    ): File {
        val exportRoot = exportProject(project, clips, bulkRecordings, backgroundRecordings)
        val syncDir = File(context.cacheDir, "sync").apply { mkdirs() }
        val zip = File(syncDir, "${exportRoot.name}.zip")
        zipDirectory(exportRoot, zip)
        return zip
    }

    private fun manifest(
        project: WakeWordProject,
        clips: JSONArray,
        bulkRecordings: JSONArray,
        backgroundRecordings: JSONArray,
    ): JSONObject {
        val packageInfo = context.packageManager.getPackageInfo(context.packageName, 0)
        return JSONObject()
            .put("schema_version", 1)
            .put("exported_at", Instant.now().toString())
            .put(
                "app",
                JSONObject()
                    .put("package", context.packageName)
                    .put("version_name", packageInfo.versionName ?: "0")
                    .put("version_code", packageInfo.longVersionCode),
            )
            .put(
                "device",
                JSONObject()
                    .put("manufacturer", Build.MANUFACTURER)
                    .put("model", Build.MODEL)
                    .put("android_sdk", Build.VERSION.SDK_INT),
            )
            .put(
                "wake_word",
                JSONObject()
                    .put("id", project.id)
                    .put("slug", project.slug)
                    .put("phrase", project.phrase)
                    .put("target_phrases", JSONArray().put(project.phrase))
                    .put("negative_phrases", JSONArray()),
            )
            .put(
                "prompt_batch",
                JSONObject()
                    .put("id", "initial_${project.id}")
                    .put("created_at", Instant.ofEpochMilli(project.createdAtMillis).toString())
                    .put("strategy", "mixed_v1"),
            )
            .put("clips", clips)
            .put("bulk_recordings", bulkRecordings)
            .put("background_recordings", backgroundRecordings)
    }

    private fun ClipRecord.toManifestJson(file: String): JSONObject {
        return JSONObject()
            .put("id", id)
            .put("file", file)
            .put("label", label.name.lowercase())
            .put("prompt", prompt)
            .put("spoken_phrase", spokenPhrase)
            .put("recorded_at", Instant.ofEpochMilli(recordedAtMillis).toString())
            .put("duration_ms", durationMs)
            .put("sample_rate_hz", sampleRateHz)
            .put("channels", channels)
            .put("encoding", encoding)
            .put(
                "conditions",
                JSONArray().apply {
                    conditions.forEach { condition -> put(condition.name.lowercase()) }
                },
            )
            .put("session_id", "initial")
            .put("notes", "")
    }

    private fun BulkRecording.toManifestJson(file: String): JSONObject {
        return JSONObject()
            .put("id", id)
            .put("file", file)
            .put("script", script)
            .put("recorded_at", Instant.ofEpochMilli(recordedAtMillis).toString())
            .put("duration_ms", durationMs)
            .put("sample_rate_hz", sampleRateHz)
            .put("channels", channels)
            .put("encoding", encoding)
            .put(
                "conditions",
                JSONArray().apply {
                    conditions.forEach { condition -> put(condition.name.lowercase()) }
                },
            )
            .put("capture", capture.toManifestJson())
            .put("session_id", capture.sessionId)
            .put("notes", "")
    }

    private fun BackgroundRecording.toManifestJson(file: String): JSONObject {
        return JSONObject()
            .put("id", id)
            .put("file", file)
            .put("recorded_at", Instant.ofEpochMilli(recordedAtMillis).toString())
            .put("duration_ms", durationMs)
            .put("sample_rate_hz", sampleRateHz)
            .put("channels", channels)
            .put("encoding", encoding)
            .put("capture", capture.toManifestJson())
            .put("session_id", capture.sessionId)
            .put("notes", "")
    }

    private fun zipDirectory(sourceDir: File, output: File) {
        ZipOutputStream(FileOutputStream(output)).use { zip ->
            sourceDir.walkTopDown()
                .filter { it.isFile }
                .forEach { file ->
                    val relative = sourceDir.toPath().relativize(file.toPath()).toString()
                    zip.putNextEntry(ZipEntry(relative))
                    FileInputStream(file).use { input ->
                        input.copyTo(zip)
                    }
                    zip.closeEntry()
                }
        }
    }
}
