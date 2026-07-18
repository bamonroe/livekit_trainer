package com.bam.livekittrainer

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioDeviceInfo
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.File
import java.io.RandomAccessFile
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.concurrent.thread

class WavRecorder(private val context: Context) {
    private var active: ActiveRecording? = null

    val isRecording: Boolean
        get() = active != null

    fun startBulk(project: WakeWordProject, script: String): File {
        check(active == null) { "recording already active" }
        check(
            context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED,
        ) { "record audio permission is required" }

        val output = bulkFile(project)
        val prompt = RecordingPrompt(
            label = ClipLabel.NEGATIVE,
            spokenPhrase = script,
            instruction = script,
        )
        startRecording(output, prompt)
        return output
    }

    fun startBackground(project: WakeWordProject): File {
        check(active == null) { "recording already active" }
        check(
            context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED,
        ) { "record audio permission is required" }

        val output = backgroundFile(project)
        val prompt = RecordingPrompt(
            label = ClipLabel.BACKGROUND,
            spokenPhrase = "",
            instruction = "background noise",
        )
        startRecording(output, prompt)
        return output
    }

    private fun startRecording(output: File, prompt: RecordingPrompt) {
        output.parentFile?.mkdirs()

        val minBuffer = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        val bufferSize = maxOf(minBuffer, SAMPLE_RATE / 2)

        // Open and start the microphone on the caller's thread so any failure
        // propagates synchronously (leaving `active` null and the recorder
        // unwedged) instead of crashing the worker thread.
        val recorder = openMicrophone(bufferSize)

        // The mic is live now, so the OS has resolved the actual input route and
        // its native hardware format; capture them before spawning the writer.
        val route = describeInput(recorder)

        val running = AtomicBoolean(true)
        val pcmBytes = AtomicInteger(0)
        val worker = thread(name = "wav-recorder") {
            writeRecording(recorder, bufferSize, output, running, pcmBytes)
        }

        active = ActiveRecording(
            output = output,
            running = running,
            worker = worker,
            prompt = prompt,
            startedAtMillis = System.currentTimeMillis(),
            pcmBytes = pcmBytes,
            route = route,
        )
    }

    /**
     * Describe the resolved input device: a human-readable route label plus the
     * microphone's native sample rate and channel count, which sit upstream of
     * the fixed 16 kHz mono PCM we actually write. Falls back to the capture
     * format when the platform does not report a routed device or its formats.
     */
    private fun describeInput(recorder: AudioRecord): AudioRoute {
        val device = recorder.routedDevice
        val route = if (device != null) {
            val type = audioDeviceTypeLabel(device.type)
            val name = device.productName?.toString()?.trim().orEmpty()
            if (name.isEmpty()) type else "$type: $name"
        } else {
            "unknown"
        }
        val sourceSampleRate = device?.sampleRates
            ?.filter { it > 0 }
            ?.maxOrNull()
            ?: SAMPLE_RATE
        val sourceChannels = device?.channelCounts
            ?.filter { it > 0 }
            ?.minOrNull()
            ?: CHANNELS
        return AudioRoute(route, sourceSampleRate, sourceChannels)
    }

    private fun audioDeviceTypeLabel(type: Int): String = when (type) {
        AudioDeviceInfo.TYPE_BUILTIN_MIC -> "builtin_mic"
        AudioDeviceInfo.TYPE_BLUETOOTH_SCO -> "bluetooth_sco"
        AudioDeviceInfo.TYPE_WIRED_HEADSET -> "wired_headset"
        AudioDeviceInfo.TYPE_USB_DEVICE -> "usb_device"
        AudioDeviceInfo.TYPE_USB_HEADSET -> "usb_headset"
        AudioDeviceInfo.TYPE_TELEPHONY -> "telephony"
        AudioDeviceInfo.TYPE_BUS -> "bus"
        else -> "type_$type"
    }

    @SuppressLint("MissingPermission")
    private fun openMicrophone(bufferSize: Int): AudioRecord {
        val recorder = try {
            AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize,
            )
        } catch (error: IllegalArgumentException) {
            throw IllegalStateException("could not open microphone", error)
        }
        if (recorder.state != AudioRecord.STATE_INITIALIZED) {
            recorder.release()
            throw IllegalStateException("microphone is unavailable")
        }
        try {
            recorder.startRecording()
        } catch (error: IllegalStateException) {
            recorder.release()
            throw IllegalStateException("could not start microphone", error)
        }
        if (recorder.recordingState != AudioRecord.RECORDSTATE_RECORDING) {
            recorder.release()
            throw IllegalStateException("microphone did not start")
        }
        return recorder
    }

    fun stop(): RecordingResult? {
        val recording = active ?: return null
        recording.running.set(false)
        recording.worker.join()
        active = null
        val byteRate = SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE
        val durationMs = recording.pcmBytes.get().toLong() * 1000L / byteRate
        return RecordingResult(
            output = recording.output,
            prompt = recording.prompt,
            recordedAtMillis = recording.startedAtMillis,
            durationMs = durationMs,
            sampleRateHz = SAMPLE_RATE,
            channels = CHANNELS,
            encoding = ENCODING,
            inputRoute = recording.route.inputRoute,
            sourceSampleRateHz = recording.route.sourceSampleRateHz,
            sourceChannels = recording.route.sourceChannels,
        )
    }

    private fun bulkFile(project: WakeWordProject): File {
        val id = "bulk_${System.currentTimeMillis()}_${UUID.randomUUID()}"
        return File(context.filesDir, "bulk/${project.slug}/$id.wav")
    }

    private fun backgroundFile(project: WakeWordProject): File {
        val id = "background_${System.currentTimeMillis()}_${UUID.randomUUID()}"
        return File(context.filesDir, "background/${project.slug}/$id.wav")
    }

    private fun writeRecording(
        recorder: AudioRecord,
        bufferSize: Int,
        output: File,
        running: AtomicBoolean,
        pcmBytes: AtomicInteger,
    ) {
        val buffer = ByteArray(bufferSize)
        try {
            RandomAccessFile(output, "rw").use { file ->
                file.setLength(0)
                writeHeader(file, 0)
                var total = 0
                try {
                    while (running.get()) {
                        val read = recorder.read(buffer, 0, buffer.size)
                        if (read > 0) {
                            file.write(buffer, 0, read)
                            total += read
                        } else if (read < 0) {
                            // Persistent read error (e.g. dead object); stop cleanly.
                            break
                        }
                    }
                } finally {
                    pcmBytes.set(total)
                    file.seek(0)
                    writeHeader(file, total)
                }
            }
        } finally {
            try {
                recorder.stop()
            } catch (_: IllegalStateException) {
            }
            recorder.release()
        }
    }

    private fun writeHeader(file: RandomAccessFile, pcmBytes: Int) {
        val byteRate = SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE
        val blockAlign = CHANNELS * BYTES_PER_SAMPLE
        file.writeBytes("RIFF")
        file.writeIntLe(36 + pcmBytes)
        file.writeBytes("WAVE")
        file.writeBytes("fmt ")
        file.writeIntLe(16)
        file.writeShortLe(1)
        file.writeShortLe(CHANNELS)
        file.writeIntLe(SAMPLE_RATE)
        file.writeIntLe(byteRate)
        file.writeShortLe(blockAlign)
        file.writeShortLe(16)
        file.writeBytes("data")
        file.writeIntLe(pcmBytes)
    }

    private fun RandomAccessFile.writeIntLe(value: Int) {
        write(value and 0xff)
        write((value shr 8) and 0xff)
        write((value shr 16) and 0xff)
        write((value shr 24) and 0xff)
    }

    private fun RandomAccessFile.writeShortLe(value: Int) {
        write(value and 0xff)
        write((value shr 8) and 0xff)
    }

    private data class ActiveRecording(
        val output: File,
        val running: AtomicBoolean,
        val worker: Thread,
        val prompt: RecordingPrompt,
        val startedAtMillis: Long,
        val pcmBytes: AtomicInteger,
        val route: AudioRoute,
    )

    /** Resolved input route and the mic's native format before conversion. */
    data class AudioRoute(
        val inputRoute: String,
        val sourceSampleRateHz: Int,
        val sourceChannels: Int,
    )

    data class RecordingResult(
        val output: File,
        val prompt: RecordingPrompt,
        val recordedAtMillis: Long,
        val durationMs: Long,
        val sampleRateHz: Int,
        val channels: Int,
        val encoding: String,
        val inputRoute: String = "",
        val sourceSampleRateHz: Int = 0,
        val sourceChannels: Int = 0,
    )

    private companion object {
        const val SAMPLE_RATE = 16_000
        const val CHANNELS = 1
        const val BYTES_PER_SAMPLE = 2
        const val ENCODING = "pcm_s16le"
    }
}
