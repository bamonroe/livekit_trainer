package com.bam.livekittrainer

/**
 * A recording as the server knows it. The server is the master record, so this
 * can describe takes captured on a different device (another phone or a tablet)
 * that were never stored on this one. [deviceModel]/[deviceManufacturer]
 * identify where it came from so any device can recognize and manage it.
 */
data class ServerRecording(
    val id: String,
    val isBackground: Boolean,
    val isTest: Boolean = false,
    /**
     * How the take was sliced: [BulkRecording.KIND_POSITIVE], `KIND_NEGATIVE`,
     * `KIND_HARD_NEGATIVE`, `background`, `test`, or `mixed` for legacy
     * single-script takes. Lets Review group recordings by kind.
     */
    val kind: String = "mixed",
    val recordedAtMillis: Long,
    val durationMs: Long,
    val positiveCount: Int,
    val negativeCount: Int,
    val backgroundCount: Int,
    val deviceManufacturer: String?,
    val deviceModel: String?,
    val appVersion: String?,
    val inputRoute: String?,
    val sessionId: String?,
) {
    /** A short human label for the capturing device, or null if unknown. */
    val deviceLabel: String?
        get() = when {
            !deviceModel.isNullOrBlank() -> deviceModel
            !deviceManufacturer.isNullOrBlank() -> deviceManufacturer
            else -> null
        }
}
