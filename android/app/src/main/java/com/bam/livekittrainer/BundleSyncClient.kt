package com.bam.livekittrainer

import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

class BundleSyncClient(private val serverUrl: String) {
    fun upload(bundleZip: File): String {
        val endpoint = URL(serverUrl.trimEnd('/') + "/sync")
        val connection = endpoint.openConnection() as HttpURLConnection
        connection.requestMethod = "POST"
        connection.connectTimeout = 10_000
        connection.readTimeout = 120_000
        connection.doOutput = true
        connection.setRequestProperty("Content-Type", "application/zip")
        connection.setRequestProperty("X-Bundle-Name", bundleZip.name)
        connection.setFixedLengthStreamingMode(bundleZip.length())

        bundleZip.inputStream().use { input ->
            connection.outputStream.use { output ->
                input.copyTo(output)
            }
        }

        val code = connection.responseCode
        val response = if (code in 200..299) {
            connection.inputStream.bufferedReader().use { it.readText() }
        } else {
            connection.errorStream?.bufferedReader()?.use { it.readText() }.orEmpty()
        }
        connection.disconnect()

        if (code !in 200..299) {
            throw IOException("Sync failed HTTP $code: $response")
        }
        return response
    }
}
