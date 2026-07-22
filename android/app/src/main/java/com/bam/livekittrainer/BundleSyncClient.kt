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

    /** Register a project on the server so it propagates to the user's other
     *  devices even before any recording exists. Idempotent on slug. */
    fun createProject(project: WakeWordProject) {
        val endpoint = URL(serverUrl.trimEnd('/') + "/projects")
        val payload = JSONObject().apply {
            put("id", project.id)
            put("slug", project.slug)
            put("phrase", project.phrase)
        }
        val body = payload.toString().toByteArray(Charsets.UTF_8)
        val connection = endpoint.openConnection() as HttpURLConnection
        connection.requestMethod = "POST"
        connection.connectTimeout = 10_000
        connection.readTimeout = 30_000
        connection.doOutput = true
        connection.setRequestProperty("Content-Type", "application/json")
        connection.setFixedLengthStreamingMode(body.size)
        try {
            connection.outputStream.use { it.write(body) }
        } catch (error: IOException) {
            connection.disconnect()
            throw error
        }
        readResponse(connection, "Create server project failed")
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
                        kind = item.optString("kind", "mixed").ifBlank { "mixed" },
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

    private fun JSONObject.optDoubleOrNull(key: String): Double? =
        if (has(key) && !isNull(key)) optDouble(key).takeIf { !it.isNaN() } else null

    private fun JSONObject.optIntOrNull(key: String): Int? =
        if (has(key) && !isNull(key)) optInt(key) else null

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

    /**
     * Fetch a representative sample of the F5 synthetic positives for review. The
     * server enumerates the synth bucket and returns an evenly-spaced spread; each
     * sample's [SyntheticSample.audioUrl] streams the clip bytes from the server.
     */
    fun loadSyntheticSamples(wakeWordSlug: String): List<SyntheticSample> {
        val endpoint = URL(serverUrl.trimEnd('/') + "/synth/${urlPart(wakeWordSlug)}/sample")
        val connection = endpoint.openConnection() as HttpURLConnection
        connection.requestMethod = "GET"
        connection.connectTimeout = 10_000
        connection.readTimeout = 30_000
        val response = readResponse(connection, "Load synthetic samples failed")
        val samples = JSONObject(response).getJSONArray("samples")
        return buildList {
            for (index in 0 until samples.length()) {
                val item = samples.getJSONObject(index)
                val fileName = item.getString("file_name")
                add(
                    SyntheticSample(
                        id = item.getString("id"),
                        fileName = fileName,
                        text = item.optString("text"),
                        audioUrl = syntheticAudioUrl(wakeWordSlug, fileName),
                    ),
                )
            }
        }
    }

    /**
     * Kick off an F5 voice-cloned positive batch for [wakeWordSlug], seeded by the
     * enrollment take. Returns immediately with the requested count; the server
     * runs the batch in the background and [syntheticGenerationStatus] polls it.
     */
    fun startSyntheticGeneration(wakeWordSlug: String, count: Int): Int {
        val endpoint = URL(
            serverUrl.trimEnd('/') + "/synth/${urlPart(wakeWordSlug)}/generate?count=$count",
        )
        val connection = endpoint.openConnection() as HttpURLConnection
        connection.requestMethod = "POST"
        connection.connectTimeout = 10_000
        connection.readTimeout = 30_000
        connection.doOutput = true
        connection.setFixedLengthStreamingMode(0)
        val response = readResponse(connection, "Start synthetic generation failed")
        return JSONObject(response).optInt("requested", count)
    }

    /** Poll the state of a slug's F5 generation run. */
    fun syntheticGenerationStatus(wakeWordSlug: String): SyntheticGenStatus {
        val endpoint = URL(
            serverUrl.trimEnd('/') + "/synth/${urlPart(wakeWordSlug)}/generate/status",
        )
        val connection = endpoint.openConnection() as HttpURLConnection
        connection.requestMethod = "GET"
        connection.connectTimeout = 10_000
        connection.readTimeout = 30_000
        val root = JSONObject(readResponse(connection, "Synthetic generation status failed"))
        return SyntheticGenStatus(
            running = root.optBoolean("running", false),
            requested = root.optInt("requested", 0),
            wrote = root.optInt("wrote", 0),
            error = root.optStringOrNull("error"),
            idle = root.optBoolean("idle", true),
        )
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
        noCache: Boolean = false,
        run: String? = null,
    ): ScoreResult {
        val endpoint = URL(
            serverUrl.trimEnd('/') +
                "/score/${urlPart(wakeWordSlug)}/${urlPart(sourceRecording)}" +
                "?mode=${urlPart(mode)}&threshold=$threshold" +
                (if (noCache) "&nocache=1" else "") +
                (if (run != null) "&run=${urlPart(run)}" else ""),
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
            run = root.optStringOrNull("run"),
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

    /**
     * Every archived trained model version for a wake word, newest first, from
     * the server's `/models/runs` provenance ledger. Each carries its own eval
     * scores so the Test page can pick a specific past model to score against
     * instead of only the current one. The endpoint returns runs across all wake
     * words; this filters to the given slug.
     */
    fun loadModelRuns(wakeWordSlug: String): List<ModelRun> {
        val endpoint = URL(serverUrl.trimEnd('/') + "/models/runs")
        val connection = endpoint.openConnection() as HttpURLConnection
        connection.requestMethod = "GET"
        connection.connectTimeout = 10_000
        connection.readTimeout = 30_000
        val response = readResponse(connection, "Model runs failed")
        val runs = JSONObject(response).optJSONArray("runs") ?: return emptyList()
        return buildList {
            for (index in 0 until runs.length()) {
                val item = runs.getJSONObject(index)
                if (item.optString("slug") != wakeWordSlug) continue
                val runId = item.optStringOrNull("run_id") ?: continue
                val eval = item.optJSONObject("eval")
                add(
                    ModelRun(
                        slug = wakeWordSlug,
                        runId = runId,
                        isCurrent = item.optInt("is_current", 0) == 1,
                        finishedAt = item.optStringOrNull("finished_at"),
                        steps = if (item.isNull("steps")) null else item.optInt("steps"),
                        personal = item.optBoolean("personal", false),
                        contextFix = if (item.isNull("context_fix")) null else item.optBoolean("context_fix"),
                        recall = eval?.optDoubleOrNull("recall"),
                        fpph = eval?.optDoubleOrNull("fpph"),
                        modelSize = item.optStringOrNull("model_size"),
                        modelType = item.optStringOrNull("model_type"),
                        positiveBoost = item.optIntOrNull("positive_boost"),
                        realPositive = item.optIntOrNull("real_positive"),
                        realNegative = item.optIntOrNull("real_negative"),
                        realBackground = item.optIntOrNull("real_background"),
                    ),
                )
            }
        }
    }

    /**
     * The stored grades for a model's test takes plus the model totals (misses,
     * false positives), read back without re-running the scorer. `run` null =
     * the current deployed model. Backs the Model test view's per-take scores and
     * statistics.
     */
    fun loadModelGrades(wakeWordSlug: String, mode: String = "full", run: String? = null): ModelGrades {
        val endpoint = URL(
            serverUrl.trimEnd('/') +
                "/score-grades/${urlPart(wakeWordSlug)}?mode=${urlPart(mode)}" +
                (if (run != null) "&run=${urlPart(run)}" else ""),
        )
        val connection = endpoint.openConnection() as HttpURLConnection
        connection.requestMethod = "GET"
        connection.connectTimeout = 10_000
        connection.readTimeout = 30_000
        return parseModelGrades(readResponse(connection, "Model grades failed"), wakeWordSlug, mode)
    }

    /**
     * Score every test take for a wake word against one model in a single
     * request and return the resulting grades and totals. The server runs the
     * takes one at a time through the scorer, so this can take a while for a
     * large test set — hence the long read timeout.
     */
    fun scoreAllTestTakes(
        wakeWordSlug: String,
        mode: String = "full",
        threshold: Double = 0.5,
        run: String? = null,
    ): ModelGrades {
        val endpoint = URL(
            serverUrl.trimEnd('/') +
                "/score-all/${urlPart(wakeWordSlug)}?mode=${urlPart(mode)}&threshold=$threshold" +
                (if (run != null) "&run=${urlPart(run)}" else ""),
        )
        val connection = endpoint.openConnection() as HttpURLConnection
        connection.requestMethod = "POST"
        connection.connectTimeout = 10_000
        connection.readTimeout = 600_000
        connection.doOutput = true
        connection.setFixedLengthStreamingMode(0)
        return parseModelGrades(readResponse(connection, "Score all failed"), wakeWordSlug, mode)
    }

    private fun parseModelGrades(response: String, fallbackSlug: String, fallbackMode: String): ModelGrades {
        val root = JSONObject(response)
        val totals = root.optJSONObject("totals")
        val gradesArr = root.optJSONArray("grades")
        return ModelGrades(
            slug = root.optString("wake_word_slug", fallbackSlug),
            run = root.optStringOrNull("run"),
            mode = root.optString("mode", fallbackMode),
            testTakes = totals?.optInt("test_takes", 0) ?: 0,
            graded = totals?.optInt("graded", 0) ?: 0,
            targets = totals?.optInt("targets", 0) ?: 0,
            truePositives = totals?.optInt("true_positives", 0) ?: 0,
            falseNegatives = totals?.optInt("false_negatives", 0) ?: 0,
            falsePositives = totals?.optInt("false_positives", 0) ?: 0,
            detections = totals?.optInt("detections", 0) ?: 0,
            grades = buildList {
                for (index in 0 until (gradesArr?.length() ?: 0)) {
                    val item = gradesArr!!.getJSONObject(index)
                    add(
                        ScoreGrade(
                            recordingId = item.optString("recording_id"),
                            run = item.optStringOrNull("run"),
                            threshold = item.optDouble("threshold", 0.5),
                            peakScore = item.optDouble("peak_score", 0.0),
                            maxScore = item.optDouble("max_score", 0.0),
                            hasTarget = item.optBoolean("has_target", false),
                            targetCount = item.optInt("target_count", 0),
                            truePositives = item.optInt("true_positives", 0),
                            falseNegatives = item.optInt("false_negatives", 0),
                            falsePositives = item.optInt("false_positives", 0),
                            detected = item.optBoolean("detected", false),
                            scoredAtMs = item.optLong("scored_at_ms", 0L),
                        ),
                    )
                }
            },
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

    /** Kick off a full training run with the given hyperparameter JSON body. */
    fun startTraining(wakeWordSlug: String, bodyJson: String): JSONObject {
        val endpoint = URL(serverUrl.trimEnd('/') + "/train/${urlPart(wakeWordSlug)}")
        val body = bodyJson.toByteArray(Charsets.UTF_8)
        val connection = endpoint.openConnection() as HttpURLConnection
        connection.requestMethod = "POST"
        connection.connectTimeout = 10_000
        connection.readTimeout = 60_000
        connection.doOutput = true
        connection.setRequestProperty("Content-Type", "application/json")
        connection.setFixedLengthStreamingMode(body.size)
        try {
            connection.outputStream.use { it.write(body) }
        } catch (error: IOException) {
            connection.disconnect()
            throw error
        }
        return JSONObject(readResponse(connection, "Start training failed"))
    }

    /** Current training status for a wake word (state, step, message, …). */
    fun trainingStatus(wakeWordSlug: String): JSONObject {
        val endpoint = URL(serverUrl.trimEnd('/') + "/train/${urlPart(wakeWordSlug)}/status")
        val connection = endpoint.openConnection() as HttpURLConnection
        connection.requestMethod = "GET"
        connection.connectTimeout = 10_000
        connection.readTimeout = 30_000
        return JSONObject(readResponse(connection, "Training status failed"))
    }

    /** Cancel the running and queued training jobs for a wake word. */
    fun cancelTraining(wakeWordSlug: String): JSONObject {
        val endpoint = URL(serverUrl.trimEnd('/') + "/train/${urlPart(wakeWordSlug)}/cancel")
        val connection = endpoint.openConnection() as HttpURLConnection
        connection.requestMethod = "POST"
        connection.connectTimeout = 10_000
        connection.readTimeout = 30_000
        connection.doOutput = true
        connection.setFixedLengthStreamingMode(0)
        return JSONObject(readResponse(connection, "Cancel training failed"))
    }

    /** The live training pipeline: every queued or running job, oldest first. */
    fun trainingQueue(): JSONObject {
        val endpoint = URL(serverUrl.trimEnd('/') + "/queue")
        val connection = endpoint.openConnection() as HttpURLConnection
        connection.requestMethod = "GET"
        connection.connectTimeout = 10_000
        connection.readTimeout = 30_000
        return JSONObject(readResponse(connection, "Training queue failed"))
    }

    /** Cancel one queue entry by its id. */
    fun deleteQueueEntry(id: Long): JSONObject {
        val endpoint = URL(serverUrl.trimEnd('/') + "/queue/$id")
        val connection = endpoint.openConnection() as HttpURLConnection
        connection.requestMethod = "DELETE"
        connection.connectTimeout = 10_000
        connection.readTimeout = 30_000
        return JSONObject(readResponse(connection, "Cancel queue entry failed"))
    }

    /** Tail of the training log for a wake word. */
    fun trainingLog(wakeWordSlug: String, tail: Int = 200): String {
        val endpoint = URL(
            serverUrl.trimEnd('/') + "/train/${urlPart(wakeWordSlug)}/log?tail=$tail",
        )
        val connection = endpoint.openConnection() as HttpURLConnection
        connection.requestMethod = "GET"
        connection.connectTimeout = 10_000
        connection.readTimeout = 30_000
        return readResponse(connection, "Training log failed")
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

    private fun syntheticAudioUrl(wakeWordSlug: String, fileName: String): String {
        return serverUrl.trimEnd('/') +
            "/synth/${urlPart(wakeWordSlug)}/audio/${urlPart(fileName)}"
    }

    private fun bulkSourceAudioUrl(wakeWordSlug: String, sourceRecording: String): String {
        return serverUrl.trimEnd('/') +
            "/review/${urlPart(wakeWordSlug)}/bulk/${urlPart(sourceRecording)}/audio"
    }

    private fun urlPart(value: String): String {
        return URLEncoder.encode(value, Charsets.UTF_8.name()).replace("+", "%20")
    }
}
