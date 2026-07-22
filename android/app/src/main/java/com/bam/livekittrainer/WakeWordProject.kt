package com.bam.livekittrainer

data class WakeWordProject(
    val id: String,
    val phrase: String,
    val slug: String,
    val createdAtMillis: Long,
    /**
     * When true, this wake word is a non-lexical sound (e.g. a fast "beep beep")
     * that Whisper cannot transcribe reliably, so positive takes are sliced by
     * sound-burst energy instead of Whisper word timestamps. Drives the Record
     * page toggle and stamps [BulkRecording.ENERGY_POSITIVE_MARKER] onto positive
     * takes so the server slices them by energy. Local-only preference; it is not
     * synced to the server (the per-take marker carries the decision instead).
     */
    val energyPositives: Boolean = false,
)

fun slugifyPhrase(input: String): String {
    val slug = input
        .trim()
        .lowercase()
        .replace(Regex("[^a-z0-9]+"), "_")
        .trim('_')
    return slug.ifBlank { "wake_word" }
}
