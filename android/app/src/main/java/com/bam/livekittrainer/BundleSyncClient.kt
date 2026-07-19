package com.bam.livekittrainer

import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import org.json.JSONObject

class BundleSyncClient(
    private val serverUrl: String,
) {
    fun saveSettings(): String {
        val endpoint = URL(serverUrl.trimEnd('/') + "/settings")
        val body = """
            {
              "sync_server_url": ${jsonString(serverUrl)}
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

    fun loadProjectCounts(): Map<String, ProjectCounts> {
        val endpoint = URL(serverUrl.trimEnd('/') + "/projects")
        val connection = endpoint.openConnection() as HttpURLConnection
        connection.requestMethod = "GET"
        connection.connectTimeout = 10_000
        connection.readTimeout = 30_000
        val response = readResponse(connection, "Load project counts failed")
        val projects = JSONObject(response).getJSONArray("projects")
        return buildMap {
            for (index in 0 until projects.length()) {
                val item = projects.getJSONObject(index)
                put(
                    item.getString("slug"),
                    ProjectCounts(
                        positive = item.optInt("positive_count", 0),
                        negative = item.optInt("negative_count", 0),
                        background = item.optInt("background_count", 0),
                        pooledNegative = item.optInt("pooled_negative_count", 0),
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

    /**
     * Map of recording id to the source-WAV SHA-256 the server holds. A null
     * value means the server has that recording but recorded it before
     * checksums existed (legacy). Ids absent from the map are not on the server.
     */
    fun loadServerChecksums(wakeWordSlug: String): Map<String, String?> {
        val endpoint = URL(serverUrl.trimEnd('/') + "/bulk/${urlPart(wakeWordSlug)}/recordings")
        val connection = endpoint.openConnection() as HttpURLConnection
        connection.requestMethod = "GET"
        connection.connectTimeout = 10_000
        connection.readTimeout = 30_000
        val response = readResponse(connection, "Load server checksums failed")
        val root = JSONObject(response)
        if (!root.has("checksums")) return emptyMap()
        val checksums = root.getJSONArray("checksums")
        return buildMap {
            for (index in 0 until checksums.length()) {
                val item = checksums.getJSONObject(index)
                put(item.getString("id"), item.optStringOrNull("sha256"))
            }
        }
    }

    fun loadServerRecordings(wakeWordSlug: String): List<ServerRecording> {
        val endpoint = URL(serverUrl.trimEnd('/') + "/bulk/${urlPart(wakeWordSlug)}/recordings/detail")
        val connection = endpoint.openConnection() as HttpURLConnection
        connection.requestMethod = "GET"
        connection.connectTimeout = 10_000
        connection.readTimeout = 30_000
        val response = readResponse(connection, "Load server recordings failed")
        val recordings = JSONObject(response).getJSONArray("recordings")
        return buildList {
            for (index in 0 until recordings.length()) {
                val item = recordings.getJSONObject(index)
                add(
                    ServerRecording(
                        id = item.getString("id"),
                        isBackground = item.optBoolean("is_background", false),
                        isTest = item.optBoolean("is_test", false),
                        recordedAtMillis = parseIsoMillis(item.optString("recorded_at")),
                        durationMs = item.optLong("duration_ms", 0L),
                        positiveCount = item.optInt("positive_count", 0),
                        negativeCount = item.optInt("negative_count", 0),
                        backgroundCount = item.optInt("background_count", 0),
                        deviceManufacturer = item.optStringOrNull("device_manufacturer"),
                        deviceModel = item.optStringOrNull("device_model"),
                        appVersion = item.optStringOrNull("app_version"),
                        inputRoute = item.optStringOrNull("input_route"),
                        sessionId = item.optStringOrNull("session_id"),
                    ),
                )
            }
        }
    }

    private fun JSONObject.optStringOrNull(key: String): String? =
        if (isNull(key)) null else optString(key).ifBlank { null }

    private fun parseIsoMillis(value: String): Long =
        if (value.isBlank()) {
            0L
        } else {
            try {
                java.time.Instant.parse(value).toEpochMilli()
            } catch (_: Exception) {
                0L
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

    /**
     * Score an already-uploaded, already-transcribed take against the trained
     * model. `mode` is "full" (continuous rolling window, the honest streaming
     * test) or "reset" (silence-padded per step). Returns the detection curve
     * plus per-utterance peak scores. Scoring replays the whole take through the
     * model, so this can take a while for long recordings.
     */
    fun loadScore(
        wakeWordSlug: String,
        sourceRecording: String,
        mode: String = "full",
        threshold: Double = 0.5,
    ): ScoreResult {
        val endpoint = URL(
            serverUrl.trimEnd('/') +
                "/score/${urlPart(wakeWordSlug)}/${urlPart(sourceRecording)}" +
                "?mode=${urlPart(mode)}&threshold=$threshold",
        )
        val connection = endpoint.openConnection() as HttpURLConnection
        connection.requestMethod = "GET"
        connection.connectTimeout = 10_000
        connection.readTimeout = 180_000
        val response = readResponse(connection, "Score failed")
        val root = JSONObject(response)
        val timesArr = root.getJSONArray("times_ms")
        val scoresArr = root.getJSONArray("scores")
        val targetsArr = root.getJSONArray("targets")
        return ScoreResult(
            sourceRecording = root.optString("source_recording", sourceRecording),
            phrase = root.optString("phrase"),
            mode = root.optString("mode", mode),
            threshold = root.optDouble("threshold", threshold),
            durationMs = root.optDouble("duration_ms", 0.0),
            windowMs = root.optInt("window_ms", 2000),
            stepMs = root.optInt("step_ms", 0),
            timesMs = buildList {
                for (index in 0 until timesArr.length()) add(timesArr.getDouble(index))
            },
            scores = buildList {
                for (index in 0 until scoresArr.length()) add(scoresArr.getDouble(index))
            },
            targets = buildList {
                for (index in 0 until targetsArr.length()) {
                    val item = targetsArr.getJSONObject(index)
                    add(
                        ScoreTarget(
                            text = item.optString("text").trim(),
                            startMs = item.optDouble("start_ms", 0.0),
                            endMs = item.optDouble("end_ms", 0.0),
                            peakScore = item.optDouble("peak_score", 0.0),
                            peakTimeMs = item.optDouble("peak_time_ms", 0.0),
                            detected = item.optBoolean("detected", false),
                        ),
                    )
                }
            },
            truePositives = root.optInt("true_positives", 0),
            falseNegatives = root.optInt("false_negatives", 0),
            falsePositives = root.optInt("false_positives", 0),
        )
    }

    /** Streaming URL for a stored bulk take's full source audio. */
    fun sourceAudioUrl(wakeWordSlug: String, sourceRecording: String): String {
        return bulkSourceAudioUrl(wakeWordSlug, sourceRecording)
    }

    /** Re-run alignment and slicing on the server from already-uploaded audio. */
    fun reprocessProject(wakeWordSlug: String): String {
        return postReprocess(serverUrl.trimEnd('/') + "/reprocess/${urlPart(wakeWordSlug)}")
    }

    /** Re-run alignment and slicing for a single already-uploaded recording. */
    fun reprocessRecording(wakeWordSlug: String, sourceRecording: String): String {
        return postReprocess(
            serverUrl.trimEnd('/') +
                "/reprocess/${urlPart(wakeWordSlug)}/${urlPart(sourceRecording)}",
        )
    }

    private fun postReprocess(url: String): String {
        val connection = URL(url).openConnection() as HttpURLConnection
        connection.requestMethod = "POST"
        connection.connectTimeout = 10_000
        connection.readTimeout = 120_000
        // Some servers require a content length for POST; send an empty body.
        connection.doOutput = true
        connection.setFixedLengthStreamingMode(0)
        val response = readResponse(connection, "Reprocess failed")
        return JSONObject(response).optString("alignment_output", response)
    }

    /** Delete a recording, its slices, and stored source WAV on the server. */
    fun deleteRecording(wakeWordSlug: String, sourceRecording: String): String {
        val endpoint = URL(
            serverUrl.trimEnd('/') + "/bulk/${urlPart(wakeWordSlug)}/${urlPart(sourceRecording)}",
        )
        val connection = endpoint.openConnection() as HttpURLConnection
        connection.requestMethod = "DELETE"
        connection.connectTimeout = 10_000
        connection.readTimeout = 30_000
        return readResponse(connection, "Delete recording failed")
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
