package com.bam.livekittrainer

data class BulkReviewClip(
    val id: String,
    val label: String,
    val spokenPhrase: String,
    val sourceRecording: String,
    val sourceStartSec: Double,
    val sourceEndSec: Double,
    val durationMs: Long,
    val averageProbability: Double,
    val wordCount: Int,
    val category: String,
    val fileName: String,
    val audioUrl: String,
)
