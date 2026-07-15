package com.bam.livekittrainer

data class ClipRecord(
    val id: String,
    val projectId: String,
    val projectSlug: String,
    val filePath: String,
    val label: ClipLabel,
    val prompt: String,
    val spokenPhrase: String,
    val recordedAtMillis: Long,
    val durationMs: Long,
    val sampleRateHz: Int,
    val channels: Int,
    val encoding: String,
)
