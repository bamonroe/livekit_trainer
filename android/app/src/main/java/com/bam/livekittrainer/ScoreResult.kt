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
