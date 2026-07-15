package com.bam.livekittrainer

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.inputmethod.EditorInfo
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.util.UUID

class MainActivity : Activity() {
    private lateinit var store: ProjectStore
    private lateinit var recorder: WavRecorder
    private lateinit var projectList: LinearLayout
    private lateinit var phraseInput: EditText
    private var activeProject: WakeWordProject? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = ProjectStore(this)
        recorder = WavRecorder(this)

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
            val firstPrompt = prompts.first()
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
                            text = "${project.phrase}\n${project.slug}\n\n$promptPreview"
                            textSize = 18f
                        },
                    )
                    addView(
                        Button(this@MainActivity).apply {
                            text = if (recorder.isRecording && activeProject?.id == project.id) {
                                "Stop recording"
                            } else {
                                "Record first prompt"
                            }
                            setOnClickListener { toggleRecording(project, firstPrompt) }
                        },
                    )
                },
            )
        }
    }

    private fun toggleRecording(project: WakeWordProject, prompt: RecordingPrompt) {
        if (recorder.isRecording) {
            val output = recorder.stop()
            activeProject = null
            renderProjects()
            if (output != null) {
                projectList.addView(
                    TextView(this).apply {
                        text = "Saved ${output.name}"
                        textSize = 16f
                        setPadding(0, 8, 0, 8)
                    },
                    0,
                )
            }
            return
        }

        try {
            recorder.start(project, prompt)
            activeProject = project
            renderProjects()
        } catch (error: IllegalStateException) {
            projectList.addView(
                TextView(this).apply {
                    text = error.message ?: "Could not start recording"
                    textSize = 18f
                    setPadding(0, 16, 0, 16)
                },
                0,
            )
        }
    }

    private companion object {
        const val REQUEST_RECORD_AUDIO = 100
    }
}
