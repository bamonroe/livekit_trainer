package com.bam.livekittrainer

data class BulkAlignment(
    val sourceRecording: String,
    val script: String,
    val words: List<BulkAlignmentWord>,
    val cuts: List<BulkAlignmentCut>,
    val audioUrl: String,
)

data class BulkAlignmentWord(
    val word: String,
    val startSec: Double,
    val endSec: Double,
)

data class BulkAlignmentCut(
    val id: String,
    val label: String,
    val startSec: Double,
    val endSec: Double,
)
