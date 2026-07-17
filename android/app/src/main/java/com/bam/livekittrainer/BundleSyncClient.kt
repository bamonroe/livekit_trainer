package com.bam.livekittrainer

import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import org.json.JSONObject

class BundleSyncClient(
    private val serverUrl: String,
    private val whisperServerUrl: String = "",
) {
    fun saveSettings(): String {
        val endpoint = URL(serverUrl.trimEnd('/') + "/settings")
        val body = """
            {
              "sync_server_url": ${jsonString(serverUrl)},
              "whisper_server_url": ${jsonString(whisperServerUrl)}
            }
        """.trimIndent().toByteArray(Charsets.UTF_8)
        val connection = endpoint.openConnection() as HttpURLConnection
        connection.requestMethod = "POST"
        connection.connectTimeout = 10_000
        connection.readTimeout = 30_000
        connection.doOutput = true
        connection.setRequestProperty("Content-Type", "application/json")
        connection.setFixedLengthStreamingMode(body.size)

        try {
            connection.outputStream.use { output ->
                output.write(body)
            }
        } catch (error: IOException) {
            connection.disconnect()
            throw error
        }

        return readResponse(connection, "Save settings failed")
    }

    fun upload(bundleZip: File): String {
        val endpoint = URL(serverUrl.trimEnd('/') + "/sync")
        val connection = endpoint.openConnection() as HttpURLConnection
        connection.requestMethod = "POST"
        connection.connectTimeout = 10_000
        connection.readTimeout = 120_000
        connection.doOutput = true
        connection.setRequestProperty("Content-Type", "application/zip")
        connection.setRequestProperty("X-Bundle-Name", bundleZip.name)
        if (whisperServerUrl.isNotBlank()) {
            connection.setRequestProperty("X-Whisper-Server-Url", whisperServerUrl)
        }
        connection.setFixedLengthStreamingMode(bundleZip.length())

        try {
            bundleZip.inputStream().use { input ->
                connection.outputStream.use { output ->
                    input.copyTo(output)
                }
            }
        } catch (error: IOException) {
            connection.disconnect()
            throw error
        }

        return readResponse(connection, "Sync failed")
    }

    fun loadProjects(): List<WakeWordProject> {
        val endpoint = URL(serverUrl.trimEnd('/') + "/projects")
        val connection = endpoint.openConnection() as HttpURLConnection
        connection.requestMethod = "GET"
        connection.connectTimeout = 10_000
        connection.readTimeout = 30_000
        val response = readResponse(connection, "Load server projects failed")
        val projects = JSONObject(response).getJSONArray("projects")
        return buildList {
            for (index in 0 until projects.length()) {
                val item = projects.getJSONObject(index)
                val slug = item.getString("slug")
                add(
                    WakeWordProject(
                        id = item.optString("id", slug),
                        phrase = item.optString("phrase", slug),
                        slug = slug,
                        createdAtMillis = item.optLong("created_at_millis", 0L),
                    ),
                )
            }
        }
    }

    fun syncedBulkRecordingIds(wakeWordSlug: String): Set<String> {
        val endpoint = URL(serverUrl.trimEnd('/') + "/bulk/${urlPart(wakeWordSlug)}/recordings")
        val connection = endpoint.openConnection() as HttpURLConnection
        connection.requestMethod = "GET"
        connection.connectTimeout = 10_000
        connection.readTimeout = 30_000
        val response = readResponse(connection, "Load synced bulk recordings failed")
        val ids = JSONObject(response).getJSONArray("recording_ids")
        return buildSet {
            for (index in 0 until ids.length()) {
                add(ids.getString(index))
            }
        }
    }

    fun loadBulkReview(wakeWordSlug: String): List<BulkReviewClip> {
        val endpoint = URL(serverUrl.trimEnd('/') + "/review/${urlPart(wakeWordSlug)}/bulk")
        val connection = endpoint.openConnection() as HttpURLConnection
        connection.requestMethod = "GET"
        connection.connectTimeout = 10_000
        connection.readTimeout = 30_000
        val response = readResponse(connection, "Load bulk review failed")
        val clips = JSONObject(response).getJSONArray("clips")
        return buildList {
            for (index in 0 until clips.length()) {
                val item = clips.getJSONObject(index)
                val category = item.getString("category")
                val fileName = item.getString("file_name")
                add(
                    BulkReviewClip(
                        id = item.getString("id"),
                        label = item.getString("label"),
                        spokenPhrase = item.optString("spoken_phrase"),
                        sourceRecording = item.optString("source_recording"),
                        sourceStartSec = item.optDouble("source_start_sec", 0.0),
                        sourceEndSec = item.optDouble("source_end_sec", 0.0),
                        durationMs = item.optLong("duration_ms", 0L),
                        averageProbability = item.optDouble("average_probability", 0.0),
                        wordCount = item.optInt("word_count", 0),
                        category = category,
                        fileName = fileName,
                        audioUrl = reviewClipUrl(wakeWordSlug, category, fileName),
                    ),
                )
            }
        }
    }

    fun loadBulkAlignment(wakeWordSlug: String, sourceRecording: String): BulkAlignment {
        val endpoint = URL(serverUrl.trimEnd('/') + "/review/${urlPart(wakeWordSlug)}/bulk/${urlPart(sourceRecording)}/alignment")
        val connection = endpoint.openConnection() as HttpURLConnection
        connection.requestMethod = "GET"
        connection.connectTimeout = 10_000
        connection.readTimeout = 30_000
        val response = readResponse(connection, "Load bulk alignment failed")
        val root = JSONObject(response)
        val words = root.getJSONArray("words")
        val cuts = root.getJSONArray("cuts")
        return BulkAlignment(
            sourceRecording = root.getString("source_recording"),
            script = root.optString("script"),
            words = buildList {
                for (index in 0 until words.length()) {
                    val item = words.getJSONObject(index)
                    add(
                        BulkAlignmentWord(
                            word = item.optString("word").trim(),
                            startSec = item.optDouble("start", 0.0),
                            endSec = item.optDouble("end", 0.0),
                        ),
                    )
                }
            },
            cuts = buildList {
                for (index in 0 until cuts.length()) {
                    val item = cuts.getJSONObject(index)
                    add(
                        BulkAlignmentCut(
                            id = item.getString("id"),
                            label = item.getString("label"),
                            startSec = item.optDouble("start_sec", 0.0),
                            endSec = item.optDouble("end_sec", 0.0),
                        ),
                    )
                }
            },
            audioUrl = bulkSourceAudioUrl(wakeWordSlug, sourceRecording),
        )
    }

    fun deleteBulkReviewClip(wakeWordSlug: String, clip: BulkReviewClip): String {
        val endpoint = URL(reviewClipUrl(wakeWordSlug, clip.category, clip.fileName))
        val connection = endpoint.openConnection() as HttpURLConnection
        connection.requestMethod = "DELETE"
        connection.connectTimeout = 10_000
        connection.readTimeout = 30_000
        return readResponse(connection, "Delete bulk review clip failed")
    }

    private fun readResponse(connection: HttpURLConnection, errorPrefix: String): String {
        try {
            val code = connection.responseCode
            val response = if (code in 200..299) {
                connection.inputStream.bufferedReader().use { it.readText() }
            } else {
                connection.errorStream?.bufferedReader()?.use { it.readText() }.orEmpty()
            }
            if (code !in 200..299) {
                throw IOException("$errorPrefix HTTP $code: $response")
            }
            return response
        } finally {
            connection.disconnect()
        }
    }

    private fun jsonString(value: String): String {
        val escaped = buildString {
            value.forEach { char ->
                when (char) {
                    '\\' -> append("\\\\")
                    '"' -> append("\\\"")
                    '\b' -> append("\\b")
                    '\n' -> append("\\n")
                    '\r' -> append("\\r")
                    '\t' -> append("\\t")
                    else -> {
                        if (char.code < 0x20) {
                            append("\\u")
                            append(char.code.toString(16).padStart(4, '0'))
                        } else {
                            append(char)
                        }
                    }
                }
            }
        }
        return "\"$escaped\""
    }

    private fun reviewClipUrl(wakeWordSlug: String, category: String, fileName: String): String {
        return serverUrl.trimEnd('/') +
            "/review/${urlPart(wakeWordSlug)}/${urlPart(category)}/${urlPart(fileName)}"
    }

    private fun bulkSourceAudioUrl(wakeWordSlug: String, sourceRecording: String): String {
        return serverUrl.trimEnd('/') +
            "/review/${urlPart(wakeWordSlug)}/bulk/${urlPart(sourceRecording)}/audio"
    }

    private fun urlPart(value: String): String {
        return URLEncoder.encode(value, Charsets.UTF_8.name()).replace("+", "%20")
    }
}
