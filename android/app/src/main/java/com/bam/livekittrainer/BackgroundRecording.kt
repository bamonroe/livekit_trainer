package com.bam.livekittrainer

/**
 * A long ambient/background noise take. Unlike a bulk script recording it has no
 * spoken script; the sync server slices it into short background training clips
 * without transcription. Attached to a project slug for transport, but the
 * trainer pools background across every wake word.
 */
data class BackgroundRecording(
    val id: String,
    val projectId: String,
    val projectSlug: String,
    val filePath: String,
    val recordedAtMillis: Long,
    val durationMs: Long,
    val sampleRateHz: Int,
    val channels: Int,
    val encoding: String,
    val capture: CaptureMetadata = CaptureMetadata.EMPTY,
)
