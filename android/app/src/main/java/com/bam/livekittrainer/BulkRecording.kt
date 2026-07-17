package com.bam.livekittrainer

data class BulkRecording(
    val id: String,
    val projectId: String,
    val projectSlug: String,
    val filePath: String,
    val script: String,
    val recordedAtMillis: Long,
    val durationMs: Long,
    val sampleRateHz: Int,
    val channels: Int,
    val encoding: String,
    val conditions: List<ClipCondition> = emptyList(),
)
