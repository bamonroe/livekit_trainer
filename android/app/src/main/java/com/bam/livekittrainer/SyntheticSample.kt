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
