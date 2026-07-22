package com.bam.livekittrainer

/**
 * One F5-TTS voice-cloned positive, sampled from the server for spot-checking on
 * the Review page. There can be thousands of these on disk, so the server
 * returns only a representative spread; [audioUrl] streams the clip bytes.
 */
data class SyntheticSample(
    val id: String,
    val fileName: String,
    val text: String,
    val audioUrl: String,
)

/**
 * State of a server-side F5 synthetic-positive generation run for one wake word,
 * polled while the enrollment detail page is generating a batch. [idle] means the
 * server has no record of a run for this slug (never started, or since restarted).
 */
data class SyntheticGenStatus(
    val running: Boolean,
    val requested: Int,
    val wrote: Int,
    val error: String?,
    val idle: Boolean,
)
