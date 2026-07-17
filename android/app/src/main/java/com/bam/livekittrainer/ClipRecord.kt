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
    val conditions: List<ClipCondition> = emptyList(),
)

enum class ClipCondition(val displayName: String) {
    QUIET_ROOM("Quiet room"),
    BACKGROUND_NOISE("Background noise"),
    BACKGROUND_SPEECH("Background speech"),
    MUSIC("Music"),
    FAN_OR_APPLIANCE("Fan or appliance"),
    CLOSE_MIC("Close mic"),
    FAR_FIELD("Far field"),
    SOFT_VOICE("Soft voice"),
    LOUD_VOICE("Loud voice"),
    TIRED_VOICE("Tired voice"),
}
