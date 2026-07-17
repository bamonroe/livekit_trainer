package com.bam.livekittrainer

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.File
import java.io.RandomAccessFile
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

class WavRecorder(private val context: Context) {
    private var active: ActiveRecording? = null

    val isRecording: Boolean
        get() = active != null

    fun start(project: WakeWordProject, prompt: RecordingPrompt): File {
        check(active == null) { "recording already active" }
        check(
            context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED,
        ) { "record audio permission is required" }

        val output = clipFile(project)
        startRecording(output, prompt)
        return output
    }

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

    private fun startRecording(output: File, prompt: RecordingPrompt) {
        output.parentFile?.mkdirs()

        val running = AtomicBoolean(true)
        val worker = thread(name = "wav-recorder") {
            writeRecording(output, running)
        }

        active = ActiveRecording(
            output = output,
            running = running,
            worker = worker,
            prompt = prompt,
            startedAtMillis = System.currentTimeMillis(),
        )
    }

    fun stop(): RecordingResult? {
        val recording = active ?: return null
        recording.running.set(false)
        recording.worker.join()
        active = null
        return RecordingResult(
            output = recording.output,
            prompt = recording.prompt,
            recordedAtMillis = recording.startedAtMillis,
            durationMs = System.currentTimeMillis() - recording.startedAtMillis,
            sampleRateHz = SAMPLE_RATE,
            channels = CHANNELS,
            encoding = ENCODING,
        )
    }

    private fun clipFile(project: WakeWordProject): File {
        val id = "clip_${System.currentTimeMillis()}_${UUID.randomUUID()}"
        return File(context.filesDir, "clips/${project.slug}/$id.wav")
    }

    private fun bulkFile(project: WakeWordProject): File {
        val id = "bulk_${System.currentTimeMillis()}_${UUID.randomUUID()}"
        return File(context.filesDir, "bulk/${project.slug}/$id.wav")
    }

    @SuppressLint("MissingPermission")
    private fun writeRecording(output: File, running: AtomicBoolean) {
        val minBuffer = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        val bufferSize = maxOf(minBuffer, SAMPLE_RATE / 2)
        val buffer = ByteArray(bufferSize)
        var pcmBytes = 0

        val recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize,
        )

        RandomAccessFile(output, "rw").use { file ->
            file.setLength(0)
            writeHeader(file, 0)
            recorder.startRecording()
            try {
                while (running.get()) {
                    val read = recorder.read(buffer, 0, buffer.size)
                    if (read > 0) {
                        file.write(buffer, 0, read)
                        pcmBytes += read
                    }
                }
            } finally {
                recorder.stop()
                recorder.release()
                file.seek(0)
                writeHeader(file, pcmBytes)
            }
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
    )

    data class RecordingResult(
        val output: File,
        val prompt: RecordingPrompt,
        val recordedAtMillis: Long,
        val durationMs: Long,
        val sampleRateHz: Int,
        val channels: Int,
        val encoding: String,
    )

    private companion object {
        const val SAMPLE_RATE = 16_000
        const val CHANNELS = 1
        const val BYTES_PER_SAMPLE = 2
        const val ENCODING = "pcm_s16le"
    }
}
