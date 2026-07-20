package com.bam.livekittrainer

data class BulkRecording(
    val id: String,
    val projectId: String,
    val projectSlug: String,
    val filePath: String,
    val script: String,
    /**
     * How this take should be sliced by the server: [KIND_POSITIVE] (wake word
     * repeated → many positives), [KIND_NEGATIVE] (ordinary speech → negatives),
     * or [KIND_HARD_NEGATIVE] (near-miss phrases → hard negatives). Test takes
     * carry [KIND_TEST]. Only hard-negative takes have a non-empty [script].
     */
    val kind: String = KIND_POSITIVE,
    val recordedAtMillis: Long,
    val durationMs: Long,
    val sampleRateHz: Int,
    val channels: Int,
    val encoding: String,
    val conditions: List<ClipCondition> = emptyList(),
    val capture: CaptureMetadata = CaptureMetadata.EMPTY,
) {
    companion object {
        const val KIND_POSITIVE = "positive"
        const val KIND_NEGATIVE = "negative"
        const val KIND_HARD_NEGATIVE = "hard_negative"
        const val KIND_TEST = "test"
    }
}
