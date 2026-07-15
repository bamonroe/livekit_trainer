package com.bam.livekittrainer

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.media.MediaPlayer
import android.os.Bundle
import android.view.inputmethod.EditorInfo
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.io.File
import java.util.UUID

class MainActivity : Activity() {
    private lateinit var store: ProjectStore
    private lateinit var recorder: WavRecorder
    private lateinit var exporter: BundleExporter
    private lateinit var projectList: LinearLayout
    private lateinit var phraseInput: EditText
    private var activeProject: WakeWordProject? = null
    private var player: MediaPlayer? = null
    private var statusMessage: String = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = ProjectStore(this)
        recorder = WavRecorder(this)
        exporter = BundleExporter(this)

        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_RECORD_AUDIO)
        }

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 48, 48, 48)
        }

        val title = TextView(this).apply {
            text = "Wake Word Trainer"
            textSize = 28f
        }

        val subtitle = TextView(this).apply {
            text = "Create a phrase to start collecting training clips."
            textSize = 18f
            setPadding(0, 12, 0, 24)
        }

        phraseInput = EditText(this).apply {
            hint = "Wake phrase, for example hey buddy"
            setSingleLine()
            imeOptions = EditorInfo.IME_ACTION_DONE
            textSize = 18f
        }

        val createButton = Button(this).apply {
            text = "Create project"
            setOnClickListener { createProject() }
        }

        val listTitle = TextView(this).apply {
            text = "Projects"
            textSize = 22f
            setPadding(0, 36, 0, 12)
        }

        projectList = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }

        root.addView(title)
        root.addView(subtitle)
        root.addView(phraseInput)
        root.addView(createButton)
        root.addView(listTitle)
        root.addView(projectList)

        setContentView(ScrollView(this).apply { addView(root) })
        renderProjects()
    }

    private fun createProject() {
        val phrase = phraseInput.text.toString().trim()
        if (phrase.isBlank()) {
            phraseInput.error = "Phrase required"
            return
        }

        store.addProject(
            WakeWordProject(
                id = UUID.randomUUID().toString(),
                phrase = phrase,
                slug = slugifyPhrase(phrase),
                createdAtMillis = System.currentTimeMillis(),
            ),
        )
        phraseInput.text.clear()
        renderProjects()
    }

    private fun renderProjects() {
        projectList.removeAllViews()
        val projects = store.loadProjects()
        if (projects.isEmpty()) {
            projectList.addView(
                TextView(this).apply {
                    text = "No wake-word projects yet."
                    textSize = 16f
                },
            )
            return
        }

        projects.forEach { project ->
            val prompts = PromptGenerator.initialBatch(project)
            val promptIndex = store.promptIndex(project.id, prompts.size)
            val currentPrompt = prompts[promptIndex]
            val clips = store.loadClips(project.id)
            val promptPreview = prompts
                .take(4)
                .joinToString(separator = "\n") { prompt ->
                    "${prompt.label.name.lowercase()}: ${prompt.instruction}"
                }

            projectList.addView(
                LinearLayout(this).apply {
                    orientation = LinearLayout.VERTICAL
                    setPadding(0, 16, 0, 24)
                    addView(
                        TextView(this@MainActivity).apply {
                            text = "${project.phrase}\n${project.slug}\n${clips.size} clips\n\n$promptPreview"
                            textSize = 18f
                        },
                    )
                    addView(
                        TextView(this@MainActivity).apply {
                            text = "Current prompt ${promptIndex + 1}/${prompts.size}: ${currentPrompt.instruction}"
                            textSize = 16f
                            setPadding(0, 8, 0, 8)
                        },
                    )
                    addView(
                        Button(this@MainActivity).apply {
                            text = if (recorder.isRecording && activeProject?.id == project.id) {
                                "Stop recording"
                            } else {
                                "Record current prompt"
                            }
                            setOnClickListener { toggleRecording(project, currentPrompt, prompts.size) }
                        },
                    )
                    addView(
                        Button(this@MainActivity).apply {
                            text = "Export bundle"
                            isEnabled = clips.isNotEmpty()
                            setOnClickListener { exportBundle(project, clips) }
                        },
                    )
                    addClipViews(project, clips)
                },
            )
        }
    }

    private fun LinearLayout.addClipViews(project: WakeWordProject, clips: List<ClipRecord>) {
        if (clips.isEmpty()) {
            addView(
                TextView(this@MainActivity).apply {
                    text = "No clips recorded yet."
                    textSize = 16f
                    setPadding(0, 8, 0, 8)
                },
            )
            return
        }

        clips.take(5).forEach { clip ->
            addView(
                TextView(this@MainActivity).apply {
                    text = "${clip.label.name.lowercase()} ${clip.durationMs} ms\n${clip.prompt}"
                    textSize = 16f
                    setPadding(0, 12, 0, 4)
                },
            )
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    addView(
                        Button(this@MainActivity).apply {
                            text = "Play"
                            setOnClickListener { playClip(clip) }
                        },
                    )
                    addView(
                        Button(this@MainActivity).apply {
                            text = "Delete"
                            setOnClickListener { deleteClip(project, clip) }
                        },
                    )
                },
            )
        }
    }

    private fun toggleRecording(project: WakeWordProject, prompt: RecordingPrompt, promptCount: Int) {
        if (recorder.isRecording) {
            val result = recorder.stop()
            activeProject = null
            if (result != null) {
                val clip = ClipRecord(
                    id = result.output.nameWithoutExtension,
                    projectId = project.id,
                    projectSlug = project.slug,
                    filePath = result.output.absolutePath,
                    label = result.prompt.label,
                    prompt = result.prompt.instruction,
                    spokenPhrase = result.prompt.spokenPhrase,
                    recordedAtMillis = result.recordedAtMillis,
                    durationMs = result.durationMs,
                    sampleRateHz = result.sampleRateHz,
                    channels = result.channels,
                    encoding = result.encoding,
                )
                store.addClip(clip)
                store.advancePrompt(project.id, promptCount)
                statusMessage = "Saved ${result.output.name}"
            }
            renderProjects()
            showStatus()
            return
        }

        try {
            recorder.start(project, prompt)
            activeProject = project
            statusMessage = "Recording ${prompt.label.name.lowercase()}: ${prompt.instruction}"
            renderProjects()
            showStatus()
        } catch (error: IllegalStateException) {
            statusMessage = error.message ?: "Could not start recording"
            showStatus()
        }
    }

    private fun playClip(clip: ClipRecord) {
        player?.release()
        player = MediaPlayer().apply {
            setDataSource(clip.filePath)
            setOnCompletionListener {
                it.release()
                if (player === it) {
                    player = null
                }
            }
            prepare()
            start()
        }
        statusMessage = "Playing ${File(clip.filePath).name}"
        showStatus()
    }

    private fun deleteClip(project: WakeWordProject, clip: ClipRecord) {
        player?.release()
        player = null
        File(clip.filePath).delete()
        store.deleteClip(clip)
        statusMessage = "Deleted clip from ${project.slug}"
        renderProjects()
        showStatus()
    }

    private fun exportBundle(project: WakeWordProject, clips: List<ClipRecord>) {
        val exportRoot = exporter.exportProject(project, clips)
        statusMessage = "Exported bundle to ${exportRoot.absolutePath}"
        renderProjects()
        showStatus()
    }

    private fun showStatus() {
        if (statusMessage.isBlank()) return
        projectList.addView(
            TextView(this).apply {
                text = statusMessage
                textSize = 16f
                setPadding(0, 8, 0, 8)
            },
            0,
        )
    }

    override fun onDestroy() {
        player?.release()
        player = null
        if (recorder.isRecording) {
            recorder.stop()
        }
        super.onDestroy()
    }

    private companion object {
        const val REQUEST_RECORD_AUDIO = 100
    }
}
