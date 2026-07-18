package com.bam.livekittrainer

import org.json.JSONObject

/**
 * Provenance for one recording take: which device and input route captured it,
 * the microphone's native audio format before our 16 kHz mono conversion, and a
 * session id that groups every take recorded in one app sitting. Stored next to
 * the recording and forwarded in the sync manifest so the trainer can later
 * trace, filter, or rebalance clips by their source hardware and session.
 */
data class CaptureMetadata(
    val deviceManufacturer: String = "",
    val deviceModel: String = "",
    val osVersion: String = "",
    val appVersion: String = "",
    val inputRoute: String = "",
    val sourceSampleRateHz: Int = 0,
    val sourceChannels: Int = 0,
    val sessionId: String = "",
) {
    fun toManifestJson(): JSONObject = JSONObject()
        .put("device_manufacturer", deviceManufacturer)
        .put("device_model", deviceModel)
        .put("os_version", osVersion)
        .put("app_version", appVersion)
        .put("input_route", inputRoute)
        .put("source_sample_rate_hz", sourceSampleRateHz)
        .put("source_channels", sourceChannels)
        .put("session_id", sessionId)

    companion object {
        val EMPTY = CaptureMetadata()
    }
}
