package com.bam.livekittrainer

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.media.MediaPlayer
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.text.InputType
import android.text.SpannableString
import android.text.Spanned
import android.text.style.BackgroundColorSpan
import android.text.style.ForegroundColorSpan
import android.text.style.StyleSpan
import android.view.Gravity
import android.view.View
import android.view.WindowInsets
import android.view.inputmethod.EditorInfo
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.io.File
import java.util.UUID
import org.json.JSONObject

class MainActivity : Activity() {
    private lateinit var store: ProjectStore
    private lateinit var recorder: WavRecorder
    private lateinit var exporter: BundleExporter
    private lateinit var lexicon: PromptLexicon
    private lateinit var root: LinearLayout
    private lateinit var sidebar: LinearLayout
    private lateinit var workspaceScroll: ScrollView
    private lateinit var workspace: LinearLayout
    private lateinit var phraseInput: EditText
    private lateinit var serverUrlInput: EditText
    private lateinit var whisperServerUrlInput: EditText
    private lateinit var bulkWakePlacementsInput: EditText
    private var drawerOpen: Boolean = false
    private var currentPage: AppPage = AppPage.Project
    private var recordingMode: RecordingMode = RecordingMode.ShortPrompts
    private var bulkScriptRevision: Int = 0
    private var bulkReviewFilter: BulkReviewFilter = BulkReviewFilter.All
    private var darkMode: Boolean = false
    private var selectedProjectId: String? = null
    private var selectedBulkRecordingId: String? = null
    private var activeProject: WakeWordProject? = null
    private var bulkReviewProjectSlug: String? = null
    private var bulkReviewClips: List<BulkReviewClip> = emptyList()
    private var bulkAlignmentProjectSlug: String? = null
    private var bulkAlignment: BulkAlignment? = null
    private var loadingBulkReview: Boolean = false
    private var loadingBulkAlignment: Boolean = false
    private var processingBulkSplit: Boolean = false
    private var alignmentPlaybackMs: Int = 0
    private var alignmentPlaybackStartUptimeMs: Long = 0L
    private var lastScheduledAlignmentBoundaryMs: Int = -1
    private val selectedConditions = mutableSetOf<ClipCondition>()
    private var player: MediaPlayer? = null
    private var activePlaybackKey: String? = null
    private val playbackHandler = Handler(Looper.getMainLooper())
    private var alignmentTicker: Runnable? = null
    private var statusMessage: String = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = ProjectStore(this)
        recorder = WavRecorder(this)
        exporter = BundleExporter(this)
        lexicon = PromptLexicon(this)
        darkMode = getSharedPreferences(SYNC_PREFS, Context.MODE_PRIVATE)
            .getBoolean(KEY_DARK_MODE, false)

        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_RECORD_AUDIO)
        }

        window.statusBarColor = surfaceColor()
        window.navigationBarColor = surfaceColor()

        root = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setBackgroundColor(surfaceColor())
        }
        root.setOnApplyWindowInsetsListener { view, insets ->
            val bars = insets.getInsets(WindowInsets.Type.systemBars())
            view.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }

        sidebar = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(14), dp(18), dp(14), dp(18))
            setBackgroundColor(sidebarColor())
            layoutParams = LinearLayout.LayoutParams(dp(184), LinearLayout.LayoutParams.MATCH_PARENT)
        }

        workspace = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(14), dp(16), dp(18))
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
        }

        workspaceScroll = ScrollView(this).apply {
            addView(workspace)
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, 1f)
            isFillViewport = false
            setBackgroundColor(surfaceColor())
        }

        root.addView(sidebar)
        root.addView(workspaceScroll)
        setContentView(root)
        render()
    }

    private fun render() {
        val projects = store.loadProjects()
        if (selectedProjectId == null || projects.none { it.id == selectedProjectId }) {
            selectedProjectId = projects.firstOrNull()?.id
        }

        window.statusBarColor = surfaceColor()
        window.navigationBarColor = surfaceColor()
        root.setBackgroundColor(surfaceColor())
        renderSidebar(projects)
        sidebar.visibility = if (drawerOpen && currentPage == AppPage.Project) View.VISIBLE else View.GONE
        sidebar.layoutParams = LinearLayout.LayoutParams(
            if (drawerOpen) LinearLayout.LayoutParams.MATCH_PARENT else dp(184),
            LinearLayout.LayoutParams.MATCH_PARENT,
        )
        workspaceScroll.visibility = if (drawerOpen && currentPage == AppPage.Project) View.GONE else View.VISIBLE
        sidebar.setBackgroundColor(sidebarColor())
        when (currentPage) {
            AppPage.Project -> renderWorkspace(projects.firstOrNull { it.id == selectedProjectId })
            AppPage.BulkRecord -> renderBulkRecordPage(projects.firstOrNull { it.id == selectedProjectId })
            AppPage.BulkDetail -> renderBulkDetailPage(projects.firstOrNull { it.id == selectedProjectId })
            AppPage.Settings -> renderSettingsPage()
        }
    }

    private fun renderSidebar(projects: List<WakeWordProject>) {
        sidebar.removeAllViews()
        sidebar.addView(text("Projects", 21f, textColor(), Typeface.BOLD))
        sidebar.addView(text("${projects.size} wake words", 13f, mutedColor()).withBottom(dp(16)))

        projects.forEach { project ->
            val bulkCount = store.loadBulkRecordings(project.id).size
            sidebar.addView(
                menuRow(project.phrase, "$bulkCount bulk recordings", project.id == selectedProjectId).apply {
                    setOnClickListener {
                        selectedProjectId = project.id
                        drawerOpen = false
                        render()
                    }
                }.withBottom(dp(8)),
            )
        }
    }

    private fun renderWorkspace(project: WakeWordProject?) {
        workspace.removeAllViews()
        workspace.addView(topBar())

        if (project == null) {
            workspace.addView(text("Collect, review, and sync wake-word clips.", 15f, mutedColor()).withBottom(dp(14)))
            workspace.addView(createProjectCard())
            workspace.addView(emptyCard("Create a wake word project to start recording.").withTop(dp(14)))
            return
        }

        val bulkRecordings = store.loadBulkRecordings(project.id)

        workspace.addView(projectHeader(project, bulkRecordings).withTop(dp(4)))
        workspace.addView(bulkOverviewCard(project, bulkRecordings).withTop(dp(12)))
        workspace.addView(createProjectCard().withTop(dp(14)))

        if (statusMessage.isNotBlank()) {
            workspace.addView(emptyCard(statusMessage).withTop(dp(14)))
        }
    }

    private fun topBar(): View {
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, 0, 0, dp(12))
            addView(
                iconButton(if (currentPage == AppPage.Settings) "‹" else "☰").apply {
                    setOnClickListener {
                        if (currentPage != AppPage.Project) {
                            currentPage = AppPage.Project
                            drawerOpen = false
                        } else {
                            drawerOpen = !drawerOpen
                        }
                        render()
                    }
                },
            )
            addView(
                text(pageTitle(), 24f, textColor(), Typeface.BOLD).apply {
                    gravity = Gravity.CENTER_VERTICAL
                    layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                        leftMargin = dp(10)
                    }
                },
            )
            addView(
                iconButton("⚙").apply {
                    visibility = if (currentPage == AppPage.Project) View.VISIBLE else View.GONE
                    setOnClickListener {
                        currentPage = AppPage.Settings
                        drawerOpen = false
                        render()
                    }
                },
            )
        }
    }

    private fun pageTitle(): String {
        return when (currentPage) {
            AppPage.Project -> "Wake Word Trainer"
            AppPage.BulkRecord -> "New bulk recording"
            AppPage.BulkDetail -> "Bulk recording"
            AppPage.Settings -> "Settings"
        }
    }

    private fun createProjectCard(): View {
        phraseInput = EditText(this).apply {
            hint = "New wake phrase"
            setSingleLine()
            imeOptions = EditorInfo.IME_ACTION_DONE
            textSize = 16f
            styleInput()
        }

        return card().apply {
            addView(text("New project", 17f, textColor(), Typeface.BOLD))
            addView(phraseInput.withTop(dp(8)))
            addView(actionButton("Create project", ButtonStyle.Secondary) { createProject() }.withTop(dp(8)))
        }
    }

    private fun renderSettingsPage() {
        workspace.removeAllViews()
        workspace.addView(topBar())
        workspace.addView(settingsCard().withTop(dp(4)))
        if (statusMessage.isNotBlank()) {
            workspace.addView(emptyCard(statusMessage).withTop(dp(14)))
        }
    }

    private fun renderBulkRecordPage(project: WakeWordProject?) {
        workspace.removeAllViews()
        workspace.addView(topBar())
        if (project == null) {
            workspace.addView(emptyCard("Create a wake word project before recording bulk audio.").withTop(dp(8)))
            return
        }
        val bulkRecordings = store.loadBulkRecordings(project.id)
        workspace.addView(bulkScriptCard(project, bulkRecordings).withTop(dp(4)))
        if (statusMessage.isNotBlank()) {
            workspace.addView(emptyCard(statusMessage).withTop(dp(14)))
        }
    }

    private fun renderBulkDetailPage(project: WakeWordProject?) {
        workspace.removeAllViews()
        workspace.addView(topBar())
        if (project == null) {
            workspace.addView(emptyCard("Create a wake word project before reviewing bulk audio.").withTop(dp(8)))
            return
        }
        val recording = store.loadBulkRecordings(project.id)
            .firstOrNull { it.id == selectedBulkRecordingId }
        if (recording == null) {
            workspace.addView(emptyCard("Bulk recording not found.").withTop(dp(8)))
            return
        }
        workspace.addView(bulkRecordingDetailCard(project, recording).withTop(dp(4)))
        if (statusMessage.isNotBlank()) {
            workspace.addView(emptyCard(statusMessage).withTop(dp(14)))
        }
    }

    private fun settingsCard(): View {
        val syncPrefs = getSharedPreferences(SYNC_PREFS, Context.MODE_PRIVATE)
        serverUrlInput = EditText(this).apply {
            hint = "Sync server URL"
            isSaveEnabled = false
            setSingleLine()
            imeOptions = EditorInfo.IME_ACTION_DONE
            textSize = 14f
            setText(syncPrefs.getString(KEY_SYNC_SERVER_URL, DEFAULT_SYNC_SERVER_URL))
            styleInput()
        }
        whisperServerUrlInput = EditText(this).apply {
            hint = "Whisper server URL"
            isSaveEnabled = false
            setSingleLine()
            imeOptions = EditorInfo.IME_ACTION_DONE
            textSize = 14f
            setText(syncPrefs.getString(KEY_WHISPER_SERVER_URL, DEFAULT_WHISPER_SERVER_URL))
            styleInput()
        }
        bulkWakePlacementsInput = EditText(this).apply {
            hint = "Wake placements per bulk script"
            isSaveEnabled = false
            setSingleLine()
            inputType = InputType.TYPE_CLASS_NUMBER
            imeOptions = EditorInfo.IME_ACTION_DONE
            textSize = 14f
            setText(savedBulkWakePlacements().toString())
            styleInput()
        }

        return card().apply {
            addView(text("Settings", 20f, textColor(), Typeface.BOLD))
            addView(text("Sync server", 15f, mutedColor()).withTop(dp(8)))
            addView(serverUrlInput.withTop(dp(6)))
            addView(text("Whisper server", 15f, mutedColor()).withTop(dp(14)))
            addView(whisperServerUrlInput.withTop(dp(6)))
            addView(text("Bulk wake placements", 15f, mutedColor()).withTop(dp(14)))
            addView(bulkWakePlacementsInput.withTop(dp(6)))
            addView(actionButton("Save settings", ButtonStyle.Primary) {
                saveSettings(
                    serverUrlInput.text.toString().trim(),
                    whisperServerUrlInput.text.toString().trim(),
                    bulkWakePlacementsInput.text.toString().trim(),
                )
            }.withTop(dp(10)))
            addView(actionButton("Load server projects", ButtonStyle.Secondary) {
                loadServerProjects(serverUrlInput.text.toString().trim())
            }.withTop(dp(8)))
            addView(text("Appearance", 15f, mutedColor()).withTop(dp(14)))
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    addView(actionButton("Light", if (!darkMode) ButtonStyle.Primary else ButtonStyle.Secondary) { setDarkMode(false) })
                    addView(actionButton("Dark", if (darkMode) ButtonStyle.Primary else ButtonStyle.Secondary) { setDarkMode(true) }.withLeft(dp(8)))
                }.withTop(dp(8)),
            )
            addView(text("Training data", 15f, mutedColor()).withTop(dp(18)))
            addView(actionButton("Delete all training data", ButtonStyle.Danger) {
                confirmResetTrainingData()
            }.withTop(dp(8)))
        }
    }

    private fun recordingModeCard(): View {
        return card().apply {
            addView(text("Recording mode", 20f, textColor(), Typeface.BOLD))
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    addView(actionButton("Short prompts", if (recordingMode == RecordingMode.ShortPrompts) ButtonStyle.Primary else ButtonStyle.Secondary) {
                        recordingMode = RecordingMode.ShortPrompts
                        render()
                    })
                    addView(actionButton("Bulk script", if (recordingMode == RecordingMode.BulkScript) ButtonStyle.Primary else ButtonStyle.Secondary) {
                        recordingMode = RecordingMode.BulkScript
                        render()
                    }.withLeft(dp(8)))
                }.withTop(dp(10)),
            )
        }
    }

    private fun bulkOverviewCard(project: WakeWordProject, bulkRecordings: List<BulkRecording>): View {
        val reviewClips = if (bulkReviewProjectSlug == project.slug) bulkReviewClips else emptyList()
        val reviewCounts = reviewClips.groupingBy { it.label }.eachCount()
        return card().apply {
            addView(text("Bulk recordings", 20f, textColor(), Typeface.BOLD))
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    addView(actionButton("New recording", ButtonStyle.Primary) {
                        currentPage = AppPage.BulkRecord
                        drawerOpen = false
                        render()
                    })
                    addView(actionButton(if (processingBulkSplit) "Splitting" else "Split batch", ButtonStyle.Secondary) {
                        splitBulkBatch(project, bulkRecordings)
                    }.apply { isEnabled = !processingBulkSplit }.withLeft(dp(8)))
                    addView(actionButton(if (loadingBulkReview) "Loading" else "Load review", ButtonStyle.Secondary) {
                        loadBulkReview(project)
                    }.apply { isEnabled = !loadingBulkReview }.withLeft(dp(8)))
                }.withTop(dp(8)),
            )
            if (reviewClips.isNotEmpty()) {
                addView(text("${reviewCounts["positive"] ?: 0} positive, ${reviewCounts["negative"] ?: 0} negative", 14f, mutedColor()).withTop(dp(8)))
            }
            if (bulkRecordings.isEmpty()) {
                addView(text("No bulk recordings yet.", 14f, mutedColor()).withTop(dp(10)))
            } else {
                bulkRecordings.forEach { recording ->
                    addView(bulkRecordingRow(recording).withTop(dp(8)))
                }
            }
        }
    }

    private fun bulkRecordingDetailCard(project: WakeWordProject, recording: BulkRecording): View {
        val reviewClips = if (bulkReviewProjectSlug == project.slug) {
            bulkReviewClips.filter { it.sourceRecording == recording.id }
        } else {
            emptyList()
        }
        val alignment = bulkAlignment
            ?.takeIf { bulkAlignmentProjectSlug == project.slug && it.sourceRecording == recording.id }
        return card().apply {
            addView(text("Bulk recording  ${recording.durationMs / 1000}s", 20f, textColor(), Typeface.BOLD))
            addView(text(File(recording.filePath).name, 12f, mutedColor()).withTop(dp(2)))
            addView(text("Original script", 15f, mutedColor()).withTop(dp(12)))
            addView(text(recording.script, 15f, textColor()).withTop(dp(4)))
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    addView(actionButton(if (loadingBulkAlignment) "Loading timing" else "Load source timing", ButtonStyle.Secondary) {
                        loadBulkAlignment(project, recording.id)
                    }.apply { isEnabled = !loadingBulkAlignment })
                    addView(actionButton(if (loadingBulkReview) "Loading slices" else "Load slices", ButtonStyle.Secondary) {
                        loadBulkReview(project)
                    }.apply { isEnabled = !loadingBulkReview }.withLeft(dp(8)))
                    addView(actionButton("Delete recording", ButtonStyle.Ghost) { deleteBulkRecording(recording) }.withLeft(dp(8)))
                }.withTop(dp(12)),
            )
            if (alignment != null || loadingBulkAlignment) {
                addView(bulkAlignmentCard(project, alignment).withTop(dp(10)))
            }
            addView(text("Slices", 18f, textColor(), Typeface.BOLD).withTop(dp(14)))
            if (reviewClips.isEmpty()) {
                addView(text("No slices loaded for this recording.", 14f, mutedColor()).withTop(dp(6)))
            } else {
                reviewClips.forEach { clip ->
                    addView(bulkReviewRow(project, clip).withTop(dp(8)))
                }
            }
        }
    }

    private fun bulkScriptCard(project: WakeWordProject, bulkRecordings: List<BulkRecording>): View {
        val wakePlacements = savedBulkWakePlacements()
        val scriptContent = PromptGenerator.bulkScriptContent(
            project,
            lexicon,
            store.promptBatch(project.id),
            bulkScriptRevision,
            wakePlacements,
        )
        val script = scriptContent.text
        return card().apply {
            addView(text("Bulk collection", 20f, textColor(), Typeface.BOLD))
            addView(bulkScriptText(scriptContent, project.phrase).withTop(dp(10)))
            addView(text("$wakePlacements wake placements, ${bulkRecordings.size} saved bulk recordings", 14f, mutedColor()).withTop(dp(8)))
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    addView(actionButton("Refresh", ButtonStyle.Secondary) {
                        bulkScriptRevision += 1
                        statusMessage = "Generated a new bulk script"
                        render()
                    })
                    val recordingThisProject = recorder.isRecording && activeProject?.id == project.id
                    addView(actionButton(if (recordingThisProject) "Stop" else "Record script", if (recordingThisProject) ButtonStyle.Danger else ButtonStyle.Primary) {
                        toggleBulkRecording(project, script)
                    }.withLeft(dp(8)))
                }.withTop(dp(12)),
            )
        }
    }

    private fun projectHeader(project: WakeWordProject, bulkRecordings: List<BulkRecording>): View {
        val reviewClips = if (bulkReviewProjectSlug == project.slug) bulkReviewClips else emptyList()
        return card().apply {
            addView(text(project.phrase, 26f, textColor(), Typeface.BOLD))
            addView(text(project.slug, 14f, mutedColor()).withBottom(dp(12)))
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    addView(statChip("Bulk WAVs", bulkRecordings.size))
                    addView(statChip("Review slices", reviewClips.size).withLeft(dp(8)))
                    addView(statChip("Wake placements", savedBulkWakePlacements()).withLeft(dp(8)))
                },
            )
        }
    }

    private fun bulkReviewFilterControls(reviewCounts: Map<String, Int>): View {
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(actionButton("All ${reviewCounts.values.sum()}", if (bulkReviewFilter == BulkReviewFilter.All) ButtonStyle.Primary else ButtonStyle.Secondary) {
                bulkReviewFilter = BulkReviewFilter.All
                render()
            })
            addView(actionButton("Pos ${reviewCounts["positive"] ?: 0}", if (bulkReviewFilter == BulkReviewFilter.Positive) ButtonStyle.Primary else ButtonStyle.Secondary) {
                bulkReviewFilter = BulkReviewFilter.Positive
                render()
            }.withLeft(dp(8)))
            addView(actionButton("Neg ${reviewCounts["negative"] ?: 0}", if (bulkReviewFilter == BulkReviewFilter.Negative) ButtonStyle.Primary else ButtonStyle.Secondary) {
                bulkReviewFilter = BulkReviewFilter.Negative
                render()
            }.withLeft(dp(8)))
        }
    }

    private fun promptCard(
        project: WakeWordProject,
        prompts: List<RecordingPrompt>,
        selectedIndex: Int,
        selectedPrompt: RecordingPrompt,
    ): View {
        return card().apply {
            addView(text("Recording prompt", 20f, textColor(), Typeface.BOLD))
            addView(labelText(selectedPrompt.label).withTop(dp(8)))
            addView(text(selectedPrompt.instruction, 24f, textColor(), Typeface.BOLD).withTop(dp(6)))
            addView(conditionSelector().withTop(dp(12)))

            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    setPadding(0, dp(14), 0, dp(14))
                    addView(actionButton("Previous", ButtonStyle.Secondary) {
                        selectPrompt(project, prompts, selectedIndex - 1)
                    })
                    addView(actionButton("Skip", ButtonStyle.Secondary) {
                        selectPrompt(project, prompts, selectedIndex + 1)
                    }.withLeft(dp(8)))
                    val recordingThisProject = recorder.isRecording && activeProject?.id == project.id
                    addView(
                        actionButton(if (recordingThisProject) "Stop" else "Record", if (recordingThisProject) ButtonStyle.Danger else ButtonStyle.Primary) {
                            toggleRecording(project, selectedPrompt, prompts.size)
                        }.withLeft(dp(8)),
                    )
                },
            )

            addView(text("Prompt queue", 16f, textColor(), Typeface.BOLD))
            prompts.forEachIndexed { index, prompt ->
                addView(
                    promptRow(index, prompt, index == selectedIndex).apply {
                        setOnClickListener { selectPrompt(project, prompts, index) }
                    }.withTop(dp(6)),
                )
            }
        }
    }

    private fun syncCard(project: WakeWordProject, clips: List<ClipRecord>, bulkRecordings: List<BulkRecording>): View {
        val hasRecordings = clips.isNotEmpty() || bulkRecordings.isNotEmpty()
        return card().apply {
            addView(text("Export and sync", 20f, textColor(), Typeface.BOLD))
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    setPadding(0, dp(10), 0, 0)
                    addView(actionButton("Export", ButtonStyle.Secondary) { exportBundle(project, clips, bulkRecordings) }.apply { isEnabled = hasRecordings })
                    addView(actionButton("Sync", ButtonStyle.Primary) { syncBundle(project, clips, bulkRecordings) }.apply { isEnabled = hasRecordings }.withLeft(dp(8)))
                },
            )
        }
    }

    private fun clipsCard(project: WakeWordProject, clips: List<ClipRecord>): View {
        return card().apply {
            addView(text("Recent clips", 20f, textColor(), Typeface.BOLD))
            if (clips.isEmpty()) {
                addView(text("No clips recorded yet.", 15f, mutedColor()).withTop(dp(8)))
            } else {
                clips.take(8).forEach { clip ->
                    addView(clipRow(project, clip).withTop(dp(8)))
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
                        conditions = selectedConditions.toList(),
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
            val conditionText = selectedConditions
                .takeIf { it.isNotEmpty() }
                ?.joinToString(", ") { it.displayName.lowercase() }
                ?.let { " with $it" }
                ?: ""
            statusMessage = "Recording ${prompt.label.name.lowercase().replace('_', ' ')}$conditionText"
            render()
        } catch (error: IllegalStateException) {
            statusMessage = error.message ?: "Could not start recording"
            render()
        }
    }

    private fun toggleBulkRecording(project: WakeWordProject, script: String) {
        if (recorder.isRecording) {
            val result = recorder.stop()
            activeProject = null
            if (result != null) {
                store.addBulkRecording(
                    BulkRecording(
                        id = result.output.nameWithoutExtension,
                        projectId = project.id,
                        projectSlug = project.slug,
                        filePath = result.output.absolutePath,
                        script = result.prompt.instruction,
                        recordedAtMillis = result.recordedAtMillis,
                        durationMs = result.durationMs,
                        sampleRateHz = result.sampleRateHz,
                        channels = result.channels,
                        encoding = result.encoding,
                        conditions = selectedConditions.toList(),
                    ),
                )
                bulkScriptRevision += 1
                statusMessage = "Saved bulk recording ${result.output.name}"
            }
            render()
            return
        }

        try {
            recorder.startBulk(project, script)
            activeProject = project
            statusMessage = "Recording bulk script"
            render()
        } catch (error: IllegalStateException) {
            statusMessage = error.message ?: "Could not start bulk recording"
            render()
        }
    }

    private fun playClip(clip: ClipRecord) {
        val playbackKey = "clip:${clip.id}"
        player?.let { current ->
            if (activePlaybackKey == playbackKey) {
                if (current.isPlaying) {
                    current.pause()
                } else {
                    current.start()
                }
                render()
                return
            }
        }
        player?.release()
        player = MediaPlayer().apply {
            setDataSource(clip.filePath)
            setOnCompletionListener {
                it.release()
                if (player === it) player = null
                activePlaybackKey = null
                render()
            }
            prepare()
            start()
        }
        activePlaybackKey = playbackKey
        statusMessage = "Playing ${File(clip.filePath).name}"
        render()
    }

    private fun deleteClip(project: WakeWordProject, clip: ClipRecord) {
        player?.release()
        player = null
        activePlaybackKey = null
        File(clip.filePath).delete()
        store.deleteClip(clip)
        statusMessage = "Deleted clip from ${project.slug}"
        render()
    }

    private fun deleteBulkRecording(recording: BulkRecording) {
        player?.release()
        player = null
        activePlaybackKey = null
        File(recording.filePath).delete()
        store.deleteBulkRecording(recording)
        statusMessage = "Deleted bulk recording"
        render()
    }

    private fun exportBundle(project: WakeWordProject, clips: List<ClipRecord>, bulkRecordings: List<BulkRecording>) {
        val exportRoot = exporter.exportProject(project, clips, bulkRecordings)
        statusMessage = "Exported bundle to ${exportRoot.absolutePath}"
        render()
    }

    private fun syncBundle(project: WakeWordProject, clips: List<ClipRecord>, bulkRecordings: List<BulkRecording>) {
        val serverUrl = savedServerUrl()
        if (serverUrl.isBlank()) {
            statusMessage = "Server URL required"
            currentPage = AppPage.Settings
            render()
            return
        }

        statusMessage = "Syncing ${clips.size} clips and ${bulkRecordings.size} bulk recordings for ${project.slug}"
        render()

        Thread {
            try {
                val client = BundleSyncClient(serverUrl, savedWhisperServerUrl())
                val syncedBulkIds = client.syncedBulkRecordingIds(project.slug)
                val unsyncedBulkRecordings = bulkRecordings.filter { it.id !in syncedBulkIds }
                val zip = exporter.exportProjectZip(project, clips, unsyncedBulkRecordings)
                val response = client.upload(zip)
                val skipped = bulkRecordings.size - unsyncedBulkRecordings.size
                runOnUiThread {
                    statusMessage = "Synced ${project.slug}; skipped $skipped existing bulk recordings: $response"
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

    private fun splitBulkBatch(project: WakeWordProject, bulkRecordings: List<BulkRecording>) {
        val serverUrl = savedServerUrl()
        if (serverUrl.isBlank()) {
            statusMessage = "Server URL required"
            currentPage = AppPage.Settings
            render()
            return
        }
        if (bulkRecordings.isEmpty()) {
            statusMessage = "Record a bulk script first"
            render()
            return
        }

        processingBulkSplit = true
        statusMessage = "Splitting ${bulkRecordings.size} bulk recordings"
        render()

        Thread {
            try {
                val client = BundleSyncClient(serverUrl, savedWhisperServerUrl())
                val zip = exporter.exportProjectZip(project, emptyList(), bulkRecordings)
                val response = client.upload(zip)
                val clips = client.loadBulkReview(project.slug)
                runOnUiThread {
                    bulkReviewProjectSlug = project.slug
                    bulkReviewClips = clips
                    bulkAlignmentProjectSlug = null
                    bulkAlignment = null
                    processingBulkSplit = false
                    val alignmentMessage = syncAlignmentMessage(response)
                    statusMessage = if (clips.isEmpty()) {
                        "No generated slices. $alignmentMessage"
                    } else {
                        "Split batch; loaded ${clips.size} slices"
                    }
                    render()
                }
            } catch (error: Exception) {
                runOnUiThread {
                    processingBulkSplit = false
                    statusMessage = error.message ?: "Split batch failed"
                    render()
                }
            }
        }.start()
    }

    private fun loadBulkReview(project: WakeWordProject) {
        val serverUrl = savedServerUrl()
        if (serverUrl.isBlank()) {
            statusMessage = "Server URL required"
            currentPage = AppPage.Settings
            render()
            return
        }
        loadingBulkReview = true
        statusMessage = "Loading generated slices for ${project.slug}"
        render()

        Thread {
            try {
                val clips = BundleSyncClient(serverUrl, savedWhisperServerUrl()).loadBulkReview(project.slug)
                runOnUiThread {
                    bulkReviewProjectSlug = project.slug
                    bulkReviewClips = clips
                    loadingBulkReview = false
                    statusMessage = if (clips.isEmpty()) {
                        "No generated slices. Press Split batch, and check the Whisper server URL."
                    } else {
                        "Loaded ${clips.size} generated slices"
                    }
                    render()
                }
            } catch (error: Exception) {
                runOnUiThread {
                    loadingBulkReview = false
                    statusMessage = error.message ?: "Load review failed"
                    render()
                }
            }
        }.start()
    }

    private fun syncAlignmentMessage(response: String): String {
        return try {
            JSONObject(response).optString("alignment_output", response)
        } catch (_: Exception) {
            response
        }
    }

    private fun playBulkReviewClip(clip: BulkReviewClip) {
        val playbackKey = "bulk:${clip.id}"
        player?.let { current ->
            if (activePlaybackKey == playbackKey) {
                if (current.isPlaying) {
                    current.pause()
                } else {
                    current.start()
                }
                render()
                return
            }
        }
        stopAlignmentTicker()
        player?.release()
        player = null
        activePlaybackKey = playbackKey
        try {
            player = MediaPlayer().apply {
                setDataSource(clip.audioUrl)
                setOnPreparedListener {
                    it.start()
                    render()
                }
                setOnCompletionListener {
                    it.release()
                    if (player === it) player = null
                    activePlaybackKey = null
                    render()
                }
                prepareAsync()
            }
            statusMessage = "Loading ${clip.label} slice"
        } catch (error: Exception) {
            activePlaybackKey = null
            player = null
            statusMessage = error.message ?: "Could not play generated slice"
        }
        render()
    }

    private fun loadBulkAlignment(project: WakeWordProject, sourceRecording: String) {
        val serverUrl = savedServerUrl()
        if (serverUrl.isBlank()) {
            statusMessage = "Server URL required"
            currentPage = AppPage.Settings
            render()
            return
        }
        loadingBulkAlignment = true
        statusMessage = "Loading alignment"
        render()

        Thread {
            try {
                val alignment = BundleSyncClient(serverUrl, savedWhisperServerUrl())
                    .loadBulkAlignment(project.slug, sourceRecording)
                runOnUiThread {
                    bulkAlignmentProjectSlug = project.slug
                    bulkAlignment = alignment
                    alignmentPlaybackMs = 0
                    loadingBulkAlignment = false
                    statusMessage = "Loaded alignment ${sourceRecording.takeLast(8)}"
                    render()
                }
            } catch (error: Exception) {
                runOnUiThread {
                    loadingBulkAlignment = false
                    statusMessage = error.message ?: "Load alignment failed"
                    render()
                }
            }
        }.start()
    }

    private fun playBulkAlignment(alignment: BulkAlignment) {
        val playbackKey = "alignment:${alignment.sourceRecording}"
        player?.let { current ->
            if (activePlaybackKey == playbackKey) {
                if (current.isPlaying) {
                    current.pause()
                    stopAlignmentTicker()
                } else {
                    current.start()
                    startAlignmentSchedule(alignment)
                }
                render()
                return
            }
        }
        stopAlignmentTicker()
        player?.release()
        player = null
        activePlaybackKey = playbackKey
        alignmentPlaybackMs = 0
        try {
            player = MediaPlayer().apply {
                setDataSource(alignment.audioUrl)
                setOnPreparedListener {
                    it.start()
                    startAlignmentSchedule(alignment)
                }
                setOnCompletionListener {
                    stopAlignmentTicker()
                    it.release()
                    if (player === it) player = null
                    activePlaybackKey = null
                    render()
                }
                prepareAsync()
            }
            statusMessage = "Loading source recording"
        } catch (error: Exception) {
            activePlaybackKey = null
            player = null
            statusMessage = error.message ?: "Could not play source recording"
        }
        render()
    }

    private fun startAlignmentSchedule(alignment: BulkAlignment) {
        stopAlignmentTicker()
        alignmentPlaybackStartUptimeMs = SystemClock.uptimeMillis()
        lastScheduledAlignmentBoundaryMs = -1
        alignmentPlaybackMs = player?.currentPosition ?: 0
        render()
        scheduleNextAlignmentBoundary(alignment)
    }

    private fun scheduleNextAlignmentBoundary(alignment: BulkAlignment) {
        val current = player ?: return
        if (!current.isPlaying) return
        val playerMs = current.currentPosition
        val elapsedMs = (SystemClock.uptimeMillis() - alignmentPlaybackStartUptimeMs).toInt().coerceAtLeast(0)
        alignmentPlaybackMs = maxOf(playerMs, elapsedMs)
        val nextBoundaryMs = alignmentBoundariesMs(alignment)
            .firstOrNull { boundary -> boundary > alignmentPlaybackMs + 1 && boundary > lastScheduledAlignmentBoundaryMs }
            ?: return
        lastScheduledAlignmentBoundaryMs = nextBoundaryMs
        val ticker = Runnable {
            val active = player ?: return@Runnable
            alignmentPlaybackMs = active.currentPosition
            render()
            if (active.isPlaying) {
                scheduleNextAlignmentBoundary(alignment)
            }
        }
        alignmentTicker = ticker
        playbackHandler.postAtTime(ticker, alignmentPlaybackStartUptimeMs + nextBoundaryMs)
    }

    private fun stopAlignmentTicker() {
        alignmentTicker?.let { playbackHandler.removeCallbacks(it) }
        alignmentTicker = null
        lastScheduledAlignmentBoundaryMs = -1
    }

    private fun alignmentBoundariesMs(alignment: BulkAlignment): List<Int> {
        return alignment.words
            .flatMap { word -> listOf(word.startSec, word.endSec) }
            .map { seconds -> (seconds * 1000.0).toInt() }
            .distinct()
            .sorted()
    }

    private fun deleteBulkReviewClip(project: WakeWordProject, clip: BulkReviewClip) {
        val serverUrl = savedServerUrl()
        stopAlignmentTicker()
        player?.release()
        player = null
        activePlaybackKey = null
        statusMessage = "Deleting generated slice"
        render()

        Thread {
            try {
                val response = BundleSyncClient(serverUrl, savedWhisperServerUrl()).deleteBulkReviewClip(project.slug, clip)
                runOnUiThread {
                    bulkReviewClips = bulkReviewClips.filterNot {
                        it.category == clip.category && it.fileName == clip.fileName
                    }
                    statusMessage = "Deleted generated slice: $response"
                    render()
                }
            } catch (error: Exception) {
                runOnUiThread {
                    statusMessage = error.message ?: "Delete generated slice failed"
                    render()
                }
            }
        }.start()
    }

    private fun saveSettings(serverUrl: String, whisperServerUrl: String, bulkWakePlacementsText: String) {
        if (serverUrl.isBlank()) {
            serverUrlInput.error = "Server URL required"
            return
        }
        val bulkWakePlacements = bulkWakePlacementsText.toIntOrNull()?.coerceIn(1, 48)
        if (bulkWakePlacements == null) {
            bulkWakePlacementsInput.error = "Enter 1 to 48"
            return
        }
        getSharedPreferences(SYNC_PREFS, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_SYNC_SERVER_URL, serverUrl)
            .putString(KEY_WHISPER_SERVER_URL, whisperServerUrl)
            .putInt(KEY_BULK_WAKE_PLACEMENTS, bulkWakePlacements)
            .apply()
        statusMessage = "Saving settings to server"
        render()

        Thread {
            try {
                val response = BundleSyncClient(serverUrl, whisperServerUrl).saveSettings()
                runOnUiThread {
                    statusMessage = "Saved settings: $response"
                    render()
                }
            } catch (error: Exception) {
                runOnUiThread {
                    statusMessage = "Saved locally, server update failed: ${error.message ?: "unknown error"}"
                    render()
                }
            }
        }.start()
    }

    private fun loadServerProjects(serverUrl: String) {
        if (serverUrl.isBlank()) {
            serverUrlInput.error = "Server URL required"
            return
        }
        statusMessage = "Loading server projects"
        render()

        Thread {
            try {
                val projects = BundleSyncClient(serverUrl, savedWhisperServerUrl()).loadProjects()
                projects.forEach { project -> store.addProject(project.normalizedForLocalImport()) }
                runOnUiThread {
                    if (selectedProjectId == null) {
                        selectedProjectId = projects.firstOrNull()?.id
                    }
                    statusMessage = "Loaded ${projects.size} server projects"
                    currentPage = AppPage.Project
                    render()
                }
            } catch (error: Exception) {
                runOnUiThread {
                    statusMessage = error.message ?: "Load server projects failed"
                    render()
                }
            }
        }.start()
    }

    private fun WakeWordProject.normalizedForLocalImport(): WakeWordProject {
        return if (createdAtMillis > 0L) {
            this
        } else {
            copy(createdAtMillis = System.currentTimeMillis())
        }
    }

    private fun savedServerUrl(): String {
        return getSharedPreferences(SYNC_PREFS, Context.MODE_PRIVATE)
            .getString(KEY_SYNC_SERVER_URL, DEFAULT_SYNC_SERVER_URL)
            ?: DEFAULT_SYNC_SERVER_URL
    }

    private fun savedWhisperServerUrl(): String {
        return getSharedPreferences(SYNC_PREFS, Context.MODE_PRIVATE)
            .getString(KEY_WHISPER_SERVER_URL, DEFAULT_WHISPER_SERVER_URL)
            .orEmpty()
            .ifBlank { DEFAULT_WHISPER_SERVER_URL }
    }

    private fun savedBulkWakePlacements(): Int {
        return getSharedPreferences(SYNC_PREFS, Context.MODE_PRIVATE)
            .getInt(KEY_BULK_WAKE_PLACEMENTS, PromptGenerator.DEFAULT_BULK_WAKE_PLACEMENTS)
            .coerceIn(1, 48)
    }

    private fun setDarkMode(enabled: Boolean) {
        darkMode = enabled
        getSharedPreferences(SYNC_PREFS, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(KEY_DARK_MODE, enabled)
            .apply()
        render()
    }

    private fun confirmResetTrainingData() {
        AlertDialog.Builder(this)
            .setTitle("Delete all training data?")
            .setMessage("This removes wake-word projects, recorded clips, prompt progress, and exported bundles from this device.")
            .setNegativeButton("Cancel", null)
            .setPositiveButton("Delete") { _, _ ->
                resetTrainingData()
            }
            .show()
    }

    private fun resetTrainingData() {
        player?.release()
        player = null
        activePlaybackKey = null
        if (recorder.isRecording) recorder.stop()
        activeProject = null
        selectedProjectId = null
        selectedConditions.clear()
        val clipCount = store.resetAllData()
        statusMessage = "Deleted all training data and $clipCount clips"
        currentPage = AppPage.Project
        drawerOpen = false
        render()
    }

    override fun onDestroy() {
        player?.release()
        player = null
        activePlaybackKey = null
        if (recorder.isRecording) recorder.stop()
        super.onDestroy()
    }

    private fun card(): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(16), dp(16), dp(16))
            background = rounded(cardColor(), dp(12), strokeColor())
        }
    }

    private fun emptyCard(message: String): View {
        return card().apply {
            addView(text(message, 15f, textColor()))
        }
    }

    private fun actionButton(label: String, style: ButtonStyle = ButtonStyle.Secondary, onClick: () -> Unit): Button {
        return Button(this).apply {
            text = label
            textSize = 14f
            isAllCaps = false
            minHeight = dp(44)
            minWidth = dp(44)
            elevation = 0f
            stateListAnimator = null
            setPadding(dp(14), 0, dp(14), 0)
            setTextColor(buttonTextColor(style))
            background = rounded(buttonFillColor(style), dp(12), buttonStrokeColor(style))
            setOnClickListener { onClick() }
        }
    }

    private fun iconButton(label: String): Button {
        return actionButton(label, ButtonStyle.Ghost) {}.apply {
            textSize = 20f
            layoutParams = LinearLayout.LayoutParams(dp(48), dp(48))
            setPadding(0, 0, 0, 0)
        }
    }

    private fun EditText.styleInput() {
        setTextColor(textColor())
        setHintTextColor(mutedColor())
        minHeight = dp(48)
        setPadding(dp(12), 0, dp(12), 0)
        background = rounded(inputColor(), dp(12), strokeColor())
    }

    private fun labelText(label: ClipLabel): TextView {
        return text(label.name.lowercase().replace('_', ' '), 13f, Color.WHITE, Typeface.BOLD).apply {
            gravity = Gravity.CENTER
            setPadding(dp(10), dp(4), dp(10), dp(4))
            background = rounded(labelColor(label), dp(20), 0)
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT)
        }
    }

    private fun menuRow(title: String, subtitle: String, selected: Boolean): View {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(12), dp(10), dp(12), dp(10))
            background = rounded(if (selected) ACCENT else Color.TRANSPARENT, dp(12), 0)
            addView(text(title, 14f, if (selected) Color.WHITE else textColor(), Typeface.BOLD))
            addView(text(subtitle, 12f, if (selected) Color.rgb(221, 245, 242) else mutedColor()).withTop(dp(2)))
        }
    }

    private fun statChip(label: String, count: Int): View {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(10), dp(8), dp(10), dp(8))
            background = rounded(promptColor(), dp(12), 0)
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            addView(text(count.toString(), 20f, textColor(), Typeface.BOLD).apply { gravity = Gravity.CENTER })
            addView(text(label, 12f, mutedColor()).apply { gravity = Gravity.CENTER })
        }
    }

    private fun promptRow(index: Int, prompt: RecordingPrompt, selected: Boolean): View {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(12), dp(10), dp(12), dp(10))
            background = rounded(if (selected) ACCENT else promptColor(), dp(12), if (selected) 0 else strokeColor())
            addView(
                text(
                    "${index + 1}. ${prompt.label.name.lowercase().replace('_', ' ')}",
                    12f,
                    if (selected) Color.rgb(221, 245, 242) else mutedColor(),
                    Typeface.BOLD,
                ),
            )
            addView(text(prompt.instruction, 15f, if (selected) Color.WHITE else textColor()).withTop(dp(2)))
        }
    }

    private fun clipRow(project: WakeWordProject, clip: ClipRecord): View {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(12), dp(10), dp(12), dp(10))
            background = rounded(promptColor(), dp(12), 0)
            addView(text("${clip.label.name.lowercase().replace('_', ' ')}  ${clip.durationMs} ms", 14f, textColor(), Typeface.BOLD))
            addView(text(clip.prompt, 13f, mutedColor()).withTop(dp(2)))
            if (clip.conditions.isNotEmpty()) {
                addView(text(clip.conditions.joinToString(", ") { it.displayName }, 12f, mutedColor(), Typeface.BOLD).withTop(dp(2)))
            }
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    val playbackKey = "clip:${clip.id}"
                    val playLabel = if (activePlaybackKey == playbackKey && player?.isPlaying == true) "Pause" else "Play"
                    addView(actionButton(playLabel, ButtonStyle.Ghost) { playClip(clip) })
                    addView(actionButton("Delete", ButtonStyle.Ghost) { deleteClip(project, clip) }.withLeft(dp(8)))
                }.withTop(dp(6)),
            )
        }
    }

    private fun bulkRecordingRow(recording: BulkRecording): View {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(12), dp(10), dp(12), dp(10))
            background = rounded(promptColor(), dp(12), 0)
            addView(text("Bulk recording  ${recording.durationMs / 1000}s", 14f, textColor(), Typeface.BOLD))
            addView(text(File(recording.filePath).name, 12f, mutedColor()).withTop(dp(2)))
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    addView(actionButton("Open", ButtonStyle.Secondary) {
                        selectedBulkRecordingId = recording.id
                        currentPage = AppPage.BulkDetail
                        drawerOpen = false
                        render()
                    })
                    addView(actionButton("Delete", ButtonStyle.Ghost) { deleteBulkRecording(recording) }.withLeft(dp(8)))
                }.withTop(dp(6)),
            )
        }
    }

    private fun bulkReviewRow(project: WakeWordProject, clip: BulkReviewClip): View {
        val confidence = (clip.averageProbability * 100.0).toInt().coerceIn(0, 100)
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(12), dp(10), dp(12), dp(10))
            background = rounded(promptColor(), dp(12), 0)
            addView(text("${clip.label}  ${clip.durationMs} ms  ${confidence}%", 14f, textColor(), Typeface.BOLD))
            addView(text(reviewSliceDisplayId(clip), 12f, mutedColor(), Typeface.BOLD).withTop(dp(2)))
            addView(reviewTranscriptText(project, clip).withTop(dp(2)))
            addView(
                text(
                    "${clip.sourceRecording.takeLast(8)}  ${"%.2f".format(clip.sourceStartSec)}-${"%.2f".format(clip.sourceEndSec)}s  ${clip.wordCount} words",
                    12f,
                    mutedColor(),
                ).withTop(dp(2)),
            )
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    val playbackKey = "bulk:${clip.id}"
                    val playLabel = if (activePlaybackKey == playbackKey && player?.isPlaying == true) "Pause" else "Play"
                    addView(actionButton(playLabel, ButtonStyle.Ghost) { playBulkReviewClip(clip) })
                    addView(actionButton("Source timing", ButtonStyle.Ghost) { loadBulkAlignment(project, clip.sourceRecording) }.withLeft(dp(8)))
                    addView(actionButton("Delete", ButtonStyle.Ghost) { deleteBulkReviewClip(project, clip) }.withLeft(dp(8)))
                }.withTop(dp(6)),
            )
        }
    }

    private fun bulkAlignmentCard(project: WakeWordProject, alignment: BulkAlignment?): View {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(12), dp(10), dp(12), dp(10))
            background = rounded(promptColor(), dp(12), strokeColor())
            addView(text("Alignment replay", 16f, textColor(), Typeface.BOLD))
            if (alignment == null) {
                addView(text("Loading alignment", 14f, mutedColor()).withTop(dp(4)))
                return@apply
            }
            addView(text("${alignment.sourceRecording.takeLast(8)}  ${alignment.words.size} words  ${alignment.cuts.size} cuts", 12f, mutedColor()).withTop(dp(2)))
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    val playbackKey = "alignment:${alignment.sourceRecording}"
                    val playLabel = if (activePlaybackKey == playbackKey && player?.isPlaying == true) "Pause source" else "Play source"
                    addView(actionButton(playLabel, ButtonStyle.Secondary) { playBulkAlignment(alignment) })
                    addView(
                        text("%.2fs".format(alignmentPlaybackMs / 1000.0), 13f, mutedColor()).apply {
                            gravity = Gravity.CENTER_VERTICAL
                            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, 1f).apply {
                                leftMargin = dp(10)
                            }
                        },
                    )
                }.withTop(dp(8)),
            )
            addView(alignmentTranscriptText(project, alignment).withTop(dp(8)))
        }
    }

    private fun alignmentTranscriptText(project: WakeWordProject, alignment: BulkAlignment): TextView {
        val currentSec = alignmentPlaybackMs / 1000.0
        val activeWord = activeAlignmentWord(alignment.words, currentSec)
        val builder = StringBuilder()
        val wordRanges = mutableListOf<Pair<IntRange, BulkAlignmentWord>>()
        val cutRanges = mutableListOf<IntRange>()
        var cutIndex = 0
        val cuts = alignment.cuts.sortedBy { it.startSec }
        alignment.words.forEach { word ->
            while (cutIndex < cuts.size && cuts[cutIndex].startSec <= word.startSec) {
                val label = "|${shortCutId(cuts[cutIndex].id)}|"
                val start = builder.length
                builder.append(label).append(' ')
                cutRanges.add(start until start + label.length)
                cutIndex += 1
            }
            val start = builder.length
            builder.append(word.word).append(' ')
            wordRanges.add((start until start + word.word.length) to word)
        }
        val styled = SpannableString(builder.toString().trimEnd())
        wordRanges.forEach { (range, word) ->
            if (word === activeWord) {
                styled.setSpan(BackgroundColorSpan(if (darkMode) Color.rgb(95, 80, 32) else Color.rgb(255, 239, 153)), range.first, range.last + 1, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
                styled.setSpan(StyleSpan(Typeface.BOLD), range.first, range.last + 1, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            }
        }
        cutRanges.forEach { range ->
            styled.setSpan(ForegroundColorSpan(Color.rgb(190, 45, 45)), range.first, range.last + 1, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            styled.setSpan(StyleSpan(Typeface.BOLD), range.first, range.last + 1, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        }
        applyPhraseStyle(styled, styled.toString(), project.phrase, Color.rgb(30, 132, 73))
        return text("", 14f, textColor()).apply {
            text = styled
        }
    }

    private fun activeAlignmentWord(words: List<BulkAlignmentWord>, currentSec: Double): BulkAlignmentWord? {
        words.firstOrNull { word -> currentSec >= word.startSec && currentSec <= word.endSec }?.let {
            return it
        }
        return words.minByOrNull { word ->
            when {
                currentSec < word.startSec -> word.startSec - currentSec
                else -> currentSec - word.endSec
            }
        }?.takeIf { word ->
            val distance = when {
                currentSec < word.startSec -> word.startSec - currentSec
                else -> currentSec - word.endSec
            }
            distance <= 0.08
        }
    }

    private fun shortCutId(id: String): String {
        return id.take(6)
    }

    private fun reviewSliceDisplayId(clip: BulkReviewClip): String {
        return clip.id.take(6)
    }

    private fun reviewTranscriptText(project: WakeWordProject, clip: BulkReviewClip): TextView {
        val transcript = clip.spokenPhrase.ifBlank { "(no transcript)" }
        val styled = SpannableString(transcript)
        if (clip.spokenPhrase.isNotBlank()) {
            val highlightColor = if (clip.label == "positive") {
                Color.rgb(30, 132, 73)
            } else {
                Color.rgb(190, 45, 45)
            }
            applyPhraseStyle(styled, transcript, project.phrase, highlightColor)
        }
        return text("", 14f, textColor()).apply {
            text = styled
        }
    }

    private fun conditionSelector(): View {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(text("Conditions", 16f, textColor(), Typeface.BOLD))
            ClipCondition.entries.chunked(2).forEach { rowConditions ->
                addView(
                    LinearLayout(this@MainActivity).apply {
                        orientation = LinearLayout.HORIZONTAL
                        rowConditions.forEachIndexed { index, condition ->
                            addView(
                                conditionButton(condition).apply {
                                    if (index > 0) withLeft(dp(8))
                                },
                            )
                        }
                        if (rowConditions.size == 1) {
                            addView(View(this@MainActivity).apply {
                                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                                    leftMargin = dp(8)
                                }
                            })
                        }
                    }.withTop(dp(6)),
                )
            }
        }
    }

    private fun conditionButton(condition: ClipCondition): Button {
        val selected = condition in selectedConditions
        return actionButton(condition.displayName, if (selected) ButtonStyle.Primary else ButtonStyle.Secondary) {
            if (condition in selectedConditions) {
                selectedConditions.remove(condition)
            } else {
                selectedConditions.add(condition)
            }
            render()
        }.apply {
            textSize = 12f
            layoutParams = LinearLayout.LayoutParams(0, dp(44), 1f)
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

    private fun bulkScriptText(script: BulkScriptContent, phrase: String): TextView {
        val styled = SpannableString(script.text)
        applyPhraseStyle(styled, script.text, phrase, Color.rgb(30, 132, 73))
        script.hardNegatives.forEach { nearMiss ->
            applyPhraseStyle(styled, script.text, nearMiss, Color.rgb(190, 45, 45))
        }
        return text("", 17f, textColor()).apply {
            text = styled
            setLineSpacing(dp(2).toFloat(), 1.0f)
        }
    }

    private fun applyPhraseStyle(styled: SpannableString, source: String, phrase: String, color: Int) {
        val trimmed = phrase.trim()
        if (trimmed.isEmpty()) return
        val pattern = Regex(
            "(?i)(?<![A-Za-z0-9])${Regex.escape(trimmed)}(?![A-Za-z0-9])",
        )
        pattern.findAll(source).forEach { match ->
            styled.setSpan(
                ForegroundColorSpan(color),
                match.range.first,
                match.range.last + 1,
                Spanned.SPAN_EXCLUSIVE_EXCLUSIVE,
            )
            styled.setSpan(
                StyleSpan(Typeface.BOLD),
                match.range.first,
                match.range.last + 1,
                Spanned.SPAN_EXCLUSIVE_EXCLUSIVE,
            )
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
            ClipLabel.POSITIVE, ClipLabel.FALSE_NEGATIVE -> ACCENT
            ClipLabel.NEGATIVE, ClipLabel.HARD_NEGATIVE, ClipLabel.FALSE_POSITIVE -> Color.rgb(149, 90, 49)
            ClipLabel.BACKGROUND -> Color.rgb(83, 91, 112)
        }
    }

    private fun surfaceColor(): Int = if (darkMode) Color.rgb(24, 28, 30) else Color.rgb(247, 248, 250)

    private fun sidebarColor(): Int = if (darkMode) Color.rgb(31, 37, 40) else Color.rgb(235, 240, 240)

    private fun cardColor(): Int = if (darkMode) Color.rgb(36, 42, 45) else Color.WHITE

    private fun promptColor(): Int = if (darkMode) Color.rgb(48, 55, 59) else Color.rgb(241, 244, 246)

    private fun inputColor(): Int = if (darkMode) Color.rgb(30, 35, 38) else Color.rgb(250, 251, 252)

    private fun textColor(): Int = if (darkMode) Color.rgb(238, 242, 244) else Color.rgb(30, 36, 38)

    private fun mutedColor(): Int = if (darkMode) Color.rgb(177, 188, 193) else Color.rgb(91, 101, 107)

    private fun strokeColor(): Int = if (darkMode) Color.rgb(70, 81, 86) else Color.rgb(219, 225, 229)

    private fun buttonFillColor(style: ButtonStyle): Int {
        return when (style) {
            ButtonStyle.Primary -> ACCENT
            ButtonStyle.Secondary -> promptColor()
            ButtonStyle.Ghost -> Color.TRANSPARENT
            ButtonStyle.Danger -> if (darkMode) Color.rgb(125, 54, 54) else Color.rgb(161, 62, 62)
        }
    }

    private fun buttonTextColor(style: ButtonStyle): Int {
        return when (style) {
            ButtonStyle.Primary, ButtonStyle.Danger -> Color.WHITE
            ButtonStyle.Secondary, ButtonStyle.Ghost -> textColor()
        }
    }

    private fun buttonStrokeColor(style: ButtonStyle): Int {
        return if (style == ButtonStyle.Ghost) strokeColor() else 0
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
        const val KEY_WHISPER_SERVER_URL = "whisper_server_url"
        const val KEY_BULK_WAKE_PLACEMENTS = "bulk_wake_placements"
        const val KEY_DARK_MODE = "dark_mode"
        const val DEFAULT_SYNC_SERVER_URL = "http://100.64.0.2:8765"
        const val DEFAULT_WHISPER_SERVER_URL = "http://pickle.bam.net:8571"
        val ACCENT: Int = Color.rgb(37, 110, 112)
    }

    private enum class AppPage {
        Project,
        BulkRecord,
        BulkDetail,
        Settings,
    }

    private enum class RecordingMode {
        ShortPrompts,
        BulkScript,
    }

    private enum class BulkReviewFilter {
        All,
        Positive,
        Negative,
    }

    private enum class ButtonStyle {
        Primary,
        Secondary,
        Ghost,
        Danger,
    }
}
