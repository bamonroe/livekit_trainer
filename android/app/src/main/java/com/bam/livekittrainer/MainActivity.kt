package com.bam.livekittrainer

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.media.MediaPlayer
import android.os.Bundle
import android.view.Gravity
import android.view.View
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
    private lateinit var sidebar: LinearLayout
    private lateinit var workspace: LinearLayout
    private lateinit var phraseInput: EditText
    private lateinit var serverUrlInput: EditText
    private var selectedProjectId: String? = null
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
            orientation = LinearLayout.HORIZONTAL
            setBackgroundColor(SURFACE)
        }

        sidebar = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(14), dp(18), dp(14), dp(18))
            setBackgroundColor(SIDEBAR)
            layoutParams = LinearLayout.LayoutParams(dp(156), LinearLayout.LayoutParams.MATCH_PARENT)
        }

        workspace = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(18), dp(18), dp(18), dp(18))
        }

        root.addView(sidebar)
        root.addView(ScrollView(this).apply {
            addView(workspace)
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, 1f)
        })
        setContentView(root)
        render()
    }

    private fun render() {
        val projects = store.loadProjects()
        if (selectedProjectId == null || projects.none { it.id == selectedProjectId }) {
            selectedProjectId = projects.firstOrNull()?.id
        }
        renderSidebar(projects)
        renderWorkspace(projects.firstOrNull { it.id == selectedProjectId })
    }

    private fun renderSidebar(projects: List<WakeWordProject>) {
        sidebar.removeAllViews()
        sidebar.addView(text("Projects", 22f, TEXT, Typeface.BOLD))
        sidebar.addView(text("${projects.size} wake words", 13f, MUTED).withBottom(dp(14)))

        projects.forEach { project ->
            val clips = store.loadClips(project.id)
            sidebar.addView(
                Button(this).apply {
                    text = "${project.phrase}\n${clips.size} clips"
                    textSize = 13f
                    gravity = Gravity.CENTER_VERTICAL
                    isAllCaps = false
                    setTextColor(if (project.id == selectedProjectId) Color.WHITE else TEXT)
                    background = rounded(if (project.id == selectedProjectId) ACCENT else Color.WHITE, dp(8), 0)
                    setOnClickListener {
                        selectedProjectId = project.id
                        render()
                    }
                }.withBottom(dp(8)),
            )
        }
    }

    private fun renderWorkspace(project: WakeWordProject?) {
        workspace.removeAllViews()
        workspace.addView(text("Wake Word Trainer", 28f, TEXT, Typeface.BOLD))
        workspace.addView(text("Collect, review, and sync training clips.", 16f, MUTED).withBottom(dp(18)))
        workspace.addView(createProjectCard())

        if (project == null) {
            workspace.addView(emptyCard("Create a wake word project to start recording."))
            return
        }

        val prompts = PromptGenerator.initialBatch(project)
        val selectedIndex = store.promptIndex(project.id, prompts.size)
        val selectedPrompt = prompts[selectedIndex]
        val clips = store.loadClips(project.id)

        workspace.addView(projectHeader(project, clips).withTop(dp(14)))
        workspace.addView(promptCard(project, prompts, selectedIndex, selectedPrompt).withTop(dp(14)))
        workspace.addView(syncCard(project, clips).withTop(dp(14)))
        workspace.addView(clipsCard(project, clips).withTop(dp(14)))

        if (statusMessage.isNotBlank()) {
            workspace.addView(emptyCard(statusMessage).withTop(dp(14)))
        }
    }

    private fun createProjectCard(): View {
        val syncPrefs = getSharedPreferences(SYNC_PREFS, Context.MODE_PRIVATE)
        phraseInput = EditText(this).apply {
            hint = "New wake phrase"
            setSingleLine()
            imeOptions = EditorInfo.IME_ACTION_DONE
            textSize = 16f
        }
        serverUrlInput = EditText(this).apply {
            hint = "Sync server URL"
            setSingleLine()
            imeOptions = EditorInfo.IME_ACTION_DONE
            textSize = 14f
            setText(syncPrefs.getString(KEY_SYNC_SERVER_URL, DEFAULT_SYNC_SERVER_URL))
        }

        return card().apply {
            addView(text("New project", 18f, TEXT, Typeface.BOLD))
            addView(phraseInput)
            addView(
                Button(this@MainActivity).apply {
                    text = "Create project"
                    isAllCaps = false
                    setOnClickListener { createProject() }
                },
            )
            addView(serverUrlInput.withTop(dp(8)))
        }
    }

    private fun projectHeader(project: WakeWordProject, clips: List<ClipRecord>): View {
        val counts = clips.groupingBy { it.label }.eachCount()
        return card().apply {
            addView(text(project.phrase, 24f, TEXT, Typeface.BOLD))
            addView(text(project.slug, 14f, MUTED).withBottom(dp(10)))
            addView(
                text(
                    "Positive ${counts[ClipLabel.POSITIVE] ?: 0}    Negative ${negativeCount(counts)}    Background ${counts[ClipLabel.BACKGROUND] ?: 0}",
                    15f,
                    TEXT,
                ),
            )
        }
    }

    private fun promptCard(
        project: WakeWordProject,
        prompts: List<RecordingPrompt>,
        selectedIndex: Int,
        selectedPrompt: RecordingPrompt,
    ): View {
        return card().apply {
            addView(text("Recording prompt", 20f, TEXT, Typeface.BOLD))
            addView(labelText(selectedPrompt.label).withTop(dp(8)))
            addView(text(selectedPrompt.instruction, 22f, TEXT, Typeface.BOLD).withTop(dp(6)))

            val controls = LinearLayout(this@MainActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                setPadding(0, dp(12), 0, dp(12))
            }
            controls.addView(actionButton("Previous") {
                selectPrompt(project, prompts, selectedIndex - 1)
            })
            controls.addView(actionButton("Skip") {
                selectPrompt(project, prompts, selectedIndex + 1)
            }.withLeft(dp(8)))
            controls.addView(actionButton(if (recorder.isRecording && activeProject?.id == project.id) "Stop" else "Record") {
                toggleRecording(project, selectedPrompt, prompts.size)
            }.withLeft(dp(8)))
            addView(controls)

            addView(text("Pick any prompt", 16f, TEXT, Typeface.BOLD))
            prompts.forEachIndexed { index, prompt ->
                addView(
                    Button(this@MainActivity).apply {
                        text = "${index + 1}. ${prompt.label.name.lowercase().replace('_', ' ')}\n${prompt.instruction}"
                        textSize = 14f
                        isAllCaps = false
                        gravity = Gravity.CENTER_VERTICAL
                        setTextColor(if (index == selectedIndex) Color.WHITE else TEXT)
                        background = rounded(if (index == selectedIndex) ACCENT else PROMPT, dp(8), 0)
                        setOnClickListener { selectPrompt(project, prompts, index) }
                    }.withTop(dp(6)),
                )
            }
        }
    }

    private fun syncCard(project: WakeWordProject, clips: List<ClipRecord>): View {
        return card().apply {
            addView(text("Export and sync", 20f, TEXT, Typeface.BOLD))
            val row = LinearLayout(this@MainActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                setPadding(0, dp(10), 0, 0)
            }
            row.addView(actionButton("Export") { exportBundle(project, clips) }.apply { isEnabled = clips.isNotEmpty() })
            row.addView(actionButton("Sync with server") { syncBundle(project, clips) }.apply { isEnabled = clips.isNotEmpty() }.withLeft(dp(8)))
            addView(row)
        }
    }

    private fun clipsCard(project: WakeWordProject, clips: List<ClipRecord>): View {
        return card().apply {
            addView(text("Recent clips", 20f, TEXT, Typeface.BOLD))
            if (clips.isEmpty()) {
                addView(text("No clips recorded yet.", 15f, MUTED).withTop(dp(8)))
            } else {
                clips.take(8).forEach { clip ->
                    addView(text("${clip.label.name.lowercase()}  ${clip.durationMs} ms", 15f, TEXT, Typeface.BOLD).withTop(dp(10)))
                    addView(text(clip.prompt, 14f, MUTED))
                    val row = LinearLayout(this@MainActivity).apply {
                        orientation = LinearLayout.HORIZONTAL
                    }
                    row.addView(actionButton("Play") { playClip(clip) })
                    row.addView(actionButton("Delete") { deleteClip(project, clip) }.withLeft(dp(8)))
                    addView(row.withTop(dp(6)))
                }
            }
        }
    }

    private fun createProject() {
        val phrase = phraseInput.text.toString().trim()
        if (phrase.isBlank()) {
            phraseInput.error = "Phrase required"
            return
        }

        val project = WakeWordProject(
            id = UUID.randomUUID().toString(),
            phrase = phrase,
            slug = slugifyPhrase(phrase),
            createdAtMillis = System.currentTimeMillis(),
        )
        store.addProject(project)
        selectedProjectId = project.id
        phraseInput.text.clear()
        render()
    }

    private fun selectPrompt(project: WakeWordProject, prompts: List<RecordingPrompt>, index: Int) {
        val next = Math.floorMod(index, prompts.size)
        store.setPromptIndex(project.id, next)
        render()
    }

    private fun toggleRecording(project: WakeWordProject, prompt: RecordingPrompt, promptCount: Int) {
        if (recorder.isRecording) {
            val result = recorder.stop()
            activeProject = null
            if (result != null) {
                store.addClip(
                    ClipRecord(
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
                    ),
                )
                store.advancePrompt(project.id, promptCount)
                statusMessage = "Saved ${result.output.name}"
            }
            render()
            return
        }

        try {
            recorder.start(project, prompt)
            activeProject = project
            statusMessage = "Recording ${prompt.label.name.lowercase().replace('_', ' ')}"
            render()
        } catch (error: IllegalStateException) {
            statusMessage = error.message ?: "Could not start recording"
            render()
        }
    }

    private fun playClip(clip: ClipRecord) {
        player?.release()
        player = MediaPlayer().apply {
            setDataSource(clip.filePath)
            setOnCompletionListener {
                it.release()
                if (player === it) player = null
            }
            prepare()
            start()
        }
        statusMessage = "Playing ${File(clip.filePath).name}"
        render()
    }

    private fun deleteClip(project: WakeWordProject, clip: ClipRecord) {
        player?.release()
        player = null
        File(clip.filePath).delete()
        store.deleteClip(clip)
        statusMessage = "Deleted clip from ${project.slug}"
        render()
    }

    private fun exportBundle(project: WakeWordProject, clips: List<ClipRecord>) {
        val exportRoot = exporter.exportProject(project, clips)
        statusMessage = "Exported bundle to ${exportRoot.absolutePath}"
        render()
    }

    private fun syncBundle(project: WakeWordProject, clips: List<ClipRecord>) {
        val serverUrl = serverUrlInput.text.toString().trim()
        if (serverUrl.isBlank()) {
            serverUrlInput.error = "Server URL required"
            return
        }
        getSharedPreferences(SYNC_PREFS, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_SYNC_SERVER_URL, serverUrl)
            .apply()

        statusMessage = "Syncing ${clips.size} clips for ${project.slug}"
        render()

        Thread {
            try {
                val zip = exporter.exportProjectZip(project, clips)
                val response = BundleSyncClient(serverUrl).upload(zip)
                runOnUiThread {
                    statusMessage = "Synced ${project.slug}: $response"
                    render()
                }
            } catch (error: Exception) {
                runOnUiThread {
                    statusMessage = error.message ?: "Sync failed"
                    render()
                }
            }
        }.start()
    }

    override fun onDestroy() {
        player?.release()
        player = null
        if (recorder.isRecording) recorder.stop()
        super.onDestroy()
    }

    private fun card(): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(16), dp(16), dp(16))
            background = rounded(Color.WHITE, dp(10), STROKE)
        }
    }

    private fun emptyCard(message: String): View {
        return card().apply {
            addView(text(message, 15f, TEXT))
        }
    }

    private fun actionButton(label: String, onClick: () -> Unit): Button {
        return Button(this).apply {
            text = label
            textSize = 14f
            isAllCaps = false
            setOnClickListener { onClick() }
        }
    }

    private fun labelText(label: ClipLabel): TextView {
        return text(label.name.lowercase().replace('_', ' '), 13f, Color.WHITE, Typeface.BOLD).apply {
            gravity = Gravity.CENTER
            setPadding(dp(10), dp(4), dp(10), dp(4))
            background = rounded(labelColor(label), dp(20), 0)
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT)
        }
    }

    private fun text(value: String, size: Float, color: Int, style: Int = Typeface.NORMAL): TextView {
        return TextView(this).apply {
            text = value
            textSize = size
            setTextColor(color)
            typeface = Typeface.create(Typeface.DEFAULT, style)
            includeFontPadding = true
        }
    }

    private fun rounded(fill: Int, radius: Int, stroke: Int): GradientDrawable {
        return GradientDrawable().apply {
            setColor(fill)
            cornerRadius = radius.toFloat()
            if (stroke != 0) setStroke(dp(1), stroke)
        }
    }

    private fun negativeCount(counts: Map<ClipLabel, Int>): Int {
        return (counts[ClipLabel.NEGATIVE] ?: 0) +
            (counts[ClipLabel.HARD_NEGATIVE] ?: 0) +
            (counts[ClipLabel.FALSE_POSITIVE] ?: 0)
    }

    private fun labelColor(label: ClipLabel): Int {
        return when (label) {
            ClipLabel.POSITIVE, ClipLabel.FALSE_NEGATIVE -> Color.rgb(47, 111, 109)
            ClipLabel.NEGATIVE, ClipLabel.HARD_NEGATIVE, ClipLabel.FALSE_POSITIVE -> Color.rgb(130, 86, 49)
            ClipLabel.BACKGROUND -> Color.rgb(83, 91, 112)
        }
    }

    private fun View.withTop(top: Int): View {
        val params = (layoutParams as? LinearLayout.LayoutParams)
            ?: LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
        params.topMargin = top
        layoutParams = params
        return this
    }

    private fun View.withBottom(bottom: Int): View {
        val params = (layoutParams as? LinearLayout.LayoutParams)
            ?: LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
        params.bottomMargin = bottom
        layoutParams = params
        return this
    }

    private fun View.withLeft(left: Int): View {
        val params = (layoutParams as? LinearLayout.LayoutParams)
            ?: LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT)
        params.leftMargin = left
        layoutParams = params
        return this
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    private companion object {
        const val REQUEST_RECORD_AUDIO = 100
        const val SYNC_PREFS = "sync"
        const val KEY_SYNC_SERVER_URL = "server_url"
        const val DEFAULT_SYNC_SERVER_URL = "http://100.64.0.2:8765"
        val SURFACE: Int = Color.rgb(243, 245, 247)
        val SIDEBAR: Int = Color.rgb(230, 235, 232)
        val PROMPT: Int = Color.rgb(238, 241, 244)
        val TEXT: Int = Color.rgb(31, 36, 38)
        val MUTED: Int = Color.rgb(91, 101, 107)
        val ACCENT: Int = Color.rgb(47, 111, 109)
        val STROKE: Int = Color.rgb(216, 222, 226)
    }
}
