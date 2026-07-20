package com.bam.livekittrainer

/**
 * A model-test score run for one recorded take, returned by the sync server's
 * `/score/:slug/:recording_id` endpoint. `timesMs`/`scores` are the raw rolling
 * detection curve; `targets` are the spoken wake phrases located in the
 * transcript and tagged with the model's peak score in each one's window.
 */
data class ScoreResult(
    val sourceRecording: String,
    val phrase: String,
    /** Archived run id that was scored, or null when the current model was used. */
    val run: String?,
    val mode: String,
    val threshold: Double,
    val durationMs: Double,
    val windowMs: Int,
    val stepMs: Int,
    val timesMs: List<Double>,
    val scores: List<Double>,
    val targets: List<ScoreTarget>,
    val truePositives: Int,
    val falseNegatives: Int,
    val falsePositives: Int,
)

data class ScoreTarget(
    val text: String,
    val startMs: Double,
    val endMs: Double,
    val peakScore: Double,
    val peakTimeMs: Double,
    val detected: Boolean,
)

/**
 * One archived trained model version for a wake word, from the sync server's
 * `/models/runs` provenance ledger. Every training run is kept under its own
 * `runs/<runId>/` with its own `.onnx` and eval scores, so past models are never
 * overwritten and any of them can be re-scored and compared. `recall`/`fpph` are
 * that run's synthetic eval numbers; `isCurrent` marks the one currently exported
 * as the wake word's default model.
 */
data class ModelRun(
    val slug: String,
    val runId: String,
    val isCurrent: Boolean,
    val finishedAt: String?,
    val steps: Int?,
    val personal: Boolean,
    val contextFix: Boolean?,
    val recall: Double?,
    val fpph: Double?,
)
