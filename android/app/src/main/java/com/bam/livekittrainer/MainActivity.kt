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
    private lateinit var bottomNav: LinearLayout
    private lateinit var workspaceScroll: ScrollView
    private lateinit var workspace: LinearLayout
    private lateinit var serverUrlInput: EditText
    private lateinit var whisperServerUrlInput: EditText
    private lateinit var bulkWakePlacementsInput: EditText
    private var currentPage: AppPage = AppPage.Record
    private var lastRenderedPage: AppPage? = null
    private var lastRenderedDetailId: String? = null
    private var bulkScriptRevision: Int = 0
    private var reprocessing: Boolean = false
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
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(surfaceColor())
        }
        root.setOnApplyWindowInsetsListener { view, insets ->
            val bars = insets.getInsets(WindowInsets.Type.systemBars())
            view.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }

        workspace = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(18), dp(14), dp(18), dp(24))
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
        }

        workspaceScroll = ScrollView(this).apply {
            addView(workspace)
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f)
            isFillViewport = false
            setBackgroundColor(surfaceColor())
        }

        bottomNav = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
        }

        root.addView(workspaceScroll)
        root.addView(bottomNav)
        setContentView(root)
        render()
    }

    private fun render() {
        val projects = store.loadProjects()
        if (selectedProjectId == null || projects.none { it.id == selectedProjectId }) {
            selectedProjectId = projects.firstOrNull()?.id
        }

        window.statusBarColor = surfaceColor()
        window.navigationBarColor = navColor()
        root.setBackgroundColor(surfaceColor())
        workspaceScroll.setBackgroundColor(surfaceColor())

        // Keep the scroll position across in-place redraws (e.g. tapping play on a
        // slice) and only snap to the top when the user actually changes pages.
        val keepScroll = currentPage == lastRenderedPage &&
            (currentPage != AppPage.Detail || selectedBulkRecordingId == lastRenderedDetailId)
        val priorScroll = workspaceScroll.scrollY

        val project = projects.firstOrNull { it.id == selectedProjectId }
        when (currentPage) {
            AppPage.Record -> renderRecordPage(project)
            AppPage.Review -> renderReviewPage(project)
            AppPage.Detail -> renderDetailPage(project)
            AppPage.Settings -> renderSettingsPage()
        }
        renderBottomNav()

        lastRenderedPage = currentPage
        lastRenderedDetailId = selectedBulkRecordingId
        workspaceScroll.post {
            workspaceScroll.scrollTo(0, if (keepScroll) priorScroll else 0)
        }
    }

    private fun renderBottomNav() {
        bottomNav.removeAllViews()
        bottomNav.background = topBorder(navColor(), strokeColor())
        bottomNav.setPadding(dp(6), dp(6), dp(6), dp(6))
        bottomNav.addView(navTab("Record", "●", AppPage.Record))
        bottomNav.addView(navTab("Review", "≡", AppPage.Review))
        bottomNav.addView(navTab("Settings", "⚙", AppPage.Settings))
    }

    private fun navTab(label: String, glyph: String, page: AppPage): View {
        val active = currentPage == page || (page == AppPage.Review && currentPage == AppPage.Detail)
        val tint = if (active) ACCENT else mutedColor()
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            setPadding(dp(6), dp(8), dp(6), dp(8))
            background = rounded(if (active) navActiveColor() else Color.TRANSPARENT, dp(14), 0)
            addView(text(glyph, 19f, tint).apply { gravity = Gravity.CENTER })
            addView(
                text(label, 12f, tint, if (active) Typeface.BOLD else Typeface.NORMAL)
                    .apply { gravity = Gravity.CENTER }.withTop(dp(2)),
            )
            setOnClickListener {
                if (currentPage != page) {
                    statusMessage = ""
                    currentPage = page
                    render()
                }
            }
        }
    }

    private fun topBar(title: String, showBack: Boolean = false): View {
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dp(2), 0, dp(16))
            if (showBack) {
                addView(
                    iconButton("‹").apply {
                        setOnClickListener {
                            currentPage = AppPage.Review
                            render()
                        }
                        (layoutParams as LinearLayout.LayoutParams).rightMargin = dp(6)
                    },
                )
            }
            addView(
                text(title, 27f, textColor(), Typeface.BOLD).apply {
                    layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                },
            )
            addView(projectChip())
        }
    }

    private fun projectChip(): View {
        val label = activeProjectOrNull()?.phrase ?: "No project"
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(13), dp(9), dp(13), dp(9))
            background = rounded(promptColor(), dp(20), strokeColor())
            addView(text(label, 13f, textColor(), Typeface.BOLD))
            addView(text("  ▾", 12f, mutedColor()))
            setOnClickListener { showProjectPicker() }
        }
    }

    private fun activeProjectOrNull(): WakeWordProject? {
        return store.loadProjects().firstOrNull { it.id == selectedProjectId }
    }

    private fun showProjectPicker() {
        val projects = store.loadProjects()
        val labels = projects.map { it.phrase }.toMutableList()
        labels.add("+  New project…")
        AlertDialog.Builder(this)
            .setTitle("Projects")
            .setItems(labels.toTypedArray()) { _, which ->
                if (which == projects.size) {
                    promptNewProject()
                } else {
                    selectedProjectId = projects[which].id
                    statusMessage = ""
                    render()
                }
            }
            .show()
    }

    private fun promptNewProject() {
        val input = EditText(this).apply {
            hint = "Wake phrase"
            setSingleLine()
            imeOptions = EditorInfo.IME_ACTION_DONE
            styleInput()
        }
        val wrap = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(8), dp(20), 0)
            addView(input)
        }
        AlertDialog.Builder(this)
            .setTitle("New wake word project")
            .setMessage("The phrase this model should wake on.")
            .setView(wrap)
            .setNegativeButton("Cancel", null)
            .setPositiveButton("Create") { _, _ ->
                val phrase = input.text.toString().trim()
                if (phrase.isNotBlank()) {
                    val project = WakeWordProject(
                        id = UUID.randomUUID().toString(),
                        phrase = phrase,
                        slug = slugifyPhrase(phrase),
                        createdAtMillis = System.currentTimeMillis(),
                    )
                    store.addProject(project)
                    selectedProjectId = project.id
                    currentPage = AppPage.Record
                    statusMessage = "Created project “$phrase”"
                    render()
                }
            }
            .show()
    }

    private fun maybeStatus() {
        if (statusMessage.isNotBlank()) {
            workspace.addView(statusStrip(statusMessage).withTop(dp(12)))
        }
    }

    private fun renderRecordPage(project: WakeWordProject?) {
        workspace.removeAllViews()
        workspace.addView(topBar("Record"))
        if (project == null) {
            workspace.addView(emptyCard("Create a wake word project to start recording. Tap the project chip above."))
            maybeStatus()
            return
        }
        val bulkRecordings = store.loadBulkRecordings(project.id)
        workspace.addView(bulkScriptCard(project, bulkRecordings))
        maybeStatus()
    }

    private fun renderReviewPage(project: WakeWordProject?) {
        workspace.removeAllViews()
        workspace.addView(topBar("Review"))
        if (project == null) {
            workspace.addView(emptyCard("Create a project and record a bulk take first."))
            maybeStatus()
            return
        }
        val bulkRecordings = store.loadBulkRecordings(project.id)
        workspace.addView(syncCard(project, bulkRecordings))
        workspace.addView(recordingsCard(project, bulkRecordings).withTop(dp(12)))
        maybeStatus()
    }

    private fun renderDetailPage(project: WakeWordProject?) {
        workspace.removeAllViews()
        workspace.addView(topBar("Recording", showBack = true))
        if (project == null) {
            workspace.addView(emptyCard("Project not found."))
            maybeStatus()
            return
        }
        val recording = store.loadBulkRecordings(project.id)
            .firstOrNull { it.id == selectedBulkRecordingId }
        if (recording == null) {
            workspace.addView(emptyCard("Bulk recording not found."))
            maybeStatus()
            return
        }
        workspace.addView(bulkRecordingDetailCard(project, recording))
        maybeStatus()
    }

    private fun renderSettingsPage() {
        workspace.removeAllViews()
        workspace.addView(topBar("Settings"))
        workspace.addView(settingsCard())
        maybeStatus()
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

    private fun syncCard(project: WakeWordProject, bulkRecordings: List<BulkRecording>): View {
        val reviewClips = if (bulkReviewProjectSlug == project.slug) bulkReviewClips else emptyList()
        val reviewCounts = reviewClips.groupingBy { it.label }.eachCount()
        val busy = processingBulkSplit || reprocessing
        return card().apply {
            addView(text("Sync & process", 20f, textColor(), Typeface.BOLD))
            addView(
                text(
                    "Upload your bulk takes; the server transcribes and slices them into training clips.",
                    14f,
                    mutedColor(),
                ).withTop(dp(4)),
            )
            addView(
                actionButton(if (processingBulkSplit) "Syncing…" else "Sync & process", ButtonStyle.Primary) {
                    syncAndProcess(project, bulkRecordings)
                }.apply { isEnabled = !busy }.tall().withTop(dp(12)),
            )
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    addView(
                        actionButton(if (reprocessing) "Reprocessing…" else "Reprocess", ButtonStyle.Secondary) {
                            reprocessProject(project)
                        }.apply { isEnabled = !busy }.weight1(),
                    )
                    addView(
                        actionButton(if (loadingBulkReview) "Loading…" else "Refresh slices", ButtonStyle.Secondary) {
                            loadBulkReview(project)
                        }.apply { isEnabled = !loadingBulkReview }.weight1().withLeft(dp(8)),
                    )
                }.withTop(dp(8)),
            )
            addView(
                text(
                    if (reviewClips.isNotEmpty()) {
                        "${reviewCounts["positive"] ?: 0} positive · ${reviewCounts["negative"] ?: 0} negative clips"
                    } else {
                        "Reprocess re-slices audio already on the server without re-uploading."
                    },
                    13f,
                    mutedColor(),
                ).withTop(dp(10)),
            )
        }
    }

    private fun recordingsCard(project: WakeWordProject, bulkRecordings: List<BulkRecording>): View {
        return card().apply {
            addView(text("Bulk recordings  ${bulkRecordings.size}", 18f, textColor(), Typeface.BOLD))
            if (bulkRecordings.isEmpty()) {
                addView(text("No bulk recordings yet. Record one on the Record tab.", 14f, mutedColor()).withTop(dp(8)))
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
        val busy = processingBulkSplit || reprocessing
        return card().apply {
            addView(text("Bulk recording  ${recording.durationMs / 1000}s", 20f, textColor(), Typeface.BOLD))
            addView(text(File(recording.filePath).name, 12f, mutedColor()).withTop(dp(2)))
            addView(text("Original script", 15f, mutedColor()).withTop(dp(12)))
            addView(text(recording.script, 15f, textColor()).withTop(dp(4)))
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    addView(
                        actionButton(if (reprocessing) "Reprocessing…" else "Reprocess", ButtonStyle.Primary) {
                            reprocessRecording(project, recording)
                        }.apply { isEnabled = !busy }.weight1(),
                    )
                    addView(
                        actionButton(if (loadingBulkReview) "Loading…" else "Refresh slices", ButtonStyle.Secondary) {
                            loadBulkReview(project)
                        }.apply { isEnabled = !loadingBulkReview }.weight1().withLeft(dp(8)),
                    )
                }.withTop(dp(12)),
            )
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    addView(
                        actionButton(if (loadingBulkAlignment) "Loading timing" else "Source timing", ButtonStyle.Secondary) {
                            loadBulkAlignment(project, recording.id)
                        }.apply { isEnabled = !loadingBulkAlignment }.weight1(),
                    )
                    addView(
                        actionButton("Delete", ButtonStyle.Danger) { confirmDeleteRecording(recording) }
                            .weight1().withLeft(dp(8)),
                    )
                }.withTop(dp(8)),
            )
            if (alignment != null || loadingBulkAlignment) {
                addView(bulkAlignmentCard(project, alignment).withTop(dp(12)))
            }
            addView(text("Slices  ${reviewClips.size}", 18f, textColor(), Typeface.BOLD).withTop(dp(16)))
            if (reviewClips.isEmpty()) {
                addView(text("No slices for this recording yet. Reprocess or refresh slices.", 14f, mutedColor()).withTop(dp(6)))
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
        val recordingThisProject = recorder.isRecording && activeProject?.id == project.id
        return card().apply {
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    addView(
                        text("Bulk script", 20f, textColor(), Typeface.BOLD).apply {
                            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                        },
                    )
                    addView(actionButton("Shuffle", ButtonStyle.Secondary) {
                        bulkScriptRevision += 1
                        statusMessage = "Generated a new bulk script"
                        render()
                    })
                },
            )
            addView(text("Read the whole script in one take. Bold green is the wake phrase; red are near-misses.", 13f, mutedColor()).withTop(dp(4)))
            addView(bulkScriptText(scriptContent, project.phrase).withTop(dp(12)))
            addView(text("$wakePlacements wake placements · ${bulkRecordings.size} saved takes", 13f, mutedColor()).withTop(dp(10)))
            addView(
                actionButton(
                    if (recordingThisProject) "◼  Stop recording" else "●  Record script",
                    if (recordingThisProject) ButtonStyle.Danger else ButtonStyle.Primary,
                ) {
                    toggleBulkRecording(project, script)
                }.tall().withTop(dp(14)),
            )
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
                        conditions = emptyList(),
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

    private fun confirmDeleteRecording(recording: BulkRecording) {
        AlertDialog.Builder(this)
            .setTitle("Delete this recording?")
            .setMessage("Removes the local take and its slices from the server. This cannot be undone.")
            .setNegativeButton("Cancel", null)
            .setPositiveButton("Delete") { _, _ -> deleteBulkRecording(recording) }
            .show()
    }

    private fun deleteBulkRecording(recording: BulkRecording) {
        player?.release()
        player = null
        activePlaybackKey = null
        File(recording.filePath).delete()
        store.deleteBulkRecording(recording)
        // Drop any loaded slices for this recording so the UI updates at once.
        bulkReviewClips = bulkReviewClips.filterNot { it.sourceRecording == recording.id }
        val serverUrl = savedServerUrl()
        currentPage = AppPage.Review
        statusMessage = "Deleted bulk recording"
        render()

        if (serverUrl.isNotBlank()) {
            Thread {
                try {
                    BundleSyncClient(serverUrl, savedWhisperServerUrl())
                        .deleteRecording(recording.projectSlug, recording.id)
                } catch (_: Exception) {
                    // Local delete already happened; a stale server copy is harmless.
                }
            }.start()
        }
    }

    private fun syncAndProcess(project: WakeWordProject, bulkRecordings: List<BulkRecording>) {
        val serverUrl = savedServerUrl()
        if (serverUrl.isBlank()) {
            statusMessage = "Set a sync server URL in Settings first"
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
        statusMessage = "Syncing ${bulkRecordings.size} bulk recordings…"
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
                        "No clips generated. $alignmentMessage"
                    } else {
                        "Synced and sliced ${clips.size} clips"
                    }
                    render()
                }
            } catch (error: Exception) {
                runOnUiThread {
                    processingBulkSplit = false
                    statusMessage = error.message ?: "Sync failed"
                    render()
                }
            }
        }.start()
    }

    private fun reprocessProject(project: WakeWordProject) {
        runReprocess(project, "Reprocessing all recordings…") { client ->
            client.reprocessProject(project.slug)
        }
    }

    private fun reprocessRecording(project: WakeWordProject, recording: BulkRecording) {
        runReprocess(project, "Reprocessing recording…") { client ->
            client.reprocessRecording(project.slug, recording.id)
        }
    }

    private fun runReprocess(
        project: WakeWordProject,
        pending: String,
        call: (BundleSyncClient) -> String,
    ) {
        val serverUrl = savedServerUrl()
        if (serverUrl.isBlank()) {
            statusMessage = "Set a sync server URL in Settings first"
            currentPage = AppPage.Settings
            render()
            return
        }
        reprocessing = true
        statusMessage = pending
        render()

        Thread {
            try {
                val client = BundleSyncClient(serverUrl, savedWhisperServerUrl())
                val message = call(client)
                val clips = client.loadBulkReview(project.slug)
                runOnUiThread {
                    bulkReviewProjectSlug = project.slug
                    bulkReviewClips = clips
                    bulkAlignmentProjectSlug = null
                    bulkAlignment = null
                    reprocessing = false
                    statusMessage = message.ifBlank { "Reprocessed ${clips.size} clips" }
                    render()
                }
            } catch (error: Exception) {
                runOnUiThread {
                    reprocessing = false
                    statusMessage = error.message ?: "Reprocess failed"
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
                        "No clips yet. Tap Sync & process, and check the Whisper server URL."
                    } else {
                        "Loaded ${clips.size} clips"
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
                    currentPage = AppPage.Record
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
        val clipCount = store.resetAllData()
        statusMessage = "Deleted all training data and $clipCount clips"
        currentPage = AppPage.Record
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
            setPadding(dp(18), dp(18), dp(18), dp(18))
            background = rounded(cardColor(), dp(18), strokeColor())
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
            background = rounded(buttonFillColor(style), dp(14), buttonStrokeColor(style))
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

    private fun bulkRecordingRow(recording: BulkRecording): View {
        val counts = if (bulkReviewProjectSlug == activeProjectOrNull()?.slug) {
            bulkReviewClips.filter { it.sourceRecording == recording.id }.groupingBy { it.label }.eachCount()
        } else {
            emptyMap()
        }
        val open = {
            selectedBulkRecordingId = recording.id
            statusMessage = ""
            currentPage = AppPage.Detail
            render()
        }
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(14), dp(12), dp(14), dp(12))
            background = rounded(promptColor(), dp(14), 0)
            setOnClickListener { open() }
            addView(text("Take  ${recording.durationMs / 1000}s", 15f, textColor(), Typeface.BOLD))
            addView(
                text(
                    if (counts.isEmpty()) {
                        "Not synced yet"
                    } else {
                        "${counts["positive"] ?: 0} positive · ${counts["negative"] ?: 0} negative"
                    },
                    12f,
                    mutedColor(),
                ).withTop(dp(2)),
            )
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    addView(actionButton("Open", ButtonStyle.Secondary) { open() }.weight1())
                    addView(actionButton("Delete", ButtonStyle.Ghost) { confirmDeleteRecording(recording) }.weight1().withLeft(dp(8)))
                }.withTop(dp(8)),
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

    /** A flat fill with only a top hairline, for the bottom navigation bar. */
    private fun topBorder(fill: Int, stroke: Int): android.graphics.drawable.Drawable {
        val line = GradientDrawable().apply { setColor(stroke) }
        val body = GradientDrawable().apply { setColor(fill) }
        return android.graphics.drawable.LayerDrawable(arrayOf(line, body)).apply {
            // Inset the fill 1px from the top so only the stroke layer shows there.
            setLayerInset(1, 0, dp(1), 0, 0)
        }
    }

    private fun statusStrip(message: String): View {
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(dp(14), dp(12), dp(14), dp(12))
            background = rounded(promptColor(), dp(14), strokeColor())
            addView(text(message, 14f, textColor()))
        }
    }

    private fun surfaceColor(): Int = if (darkMode) Color.rgb(24, 28, 30) else Color.rgb(247, 248, 250)

    private fun navColor(): Int = if (darkMode) Color.rgb(30, 35, 38) else Color.WHITE

    private fun navActiveColor(): Int = if (darkMode) Color.rgb(38, 52, 52) else Color.rgb(224, 238, 237)

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

    /** Full-width, taller emphasis for a primary button. */
    private fun Button.tall(): Button {
        minHeight = dp(56)
        textSize = 16f
        val params = (layoutParams as? LinearLayout.LayoutParams)
            ?: LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
        params.width = LinearLayout.LayoutParams.MATCH_PARENT
        layoutParams = params
        return this
    }

    /** Equal-share width inside a horizontal row. */
    private fun View.weight1(): View {
        val params = (layoutParams as? LinearLayout.LayoutParams)
            ?: LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT)
        params.width = 0
        params.weight = 1f
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
        const val DEFAULT_WHISPER_SERVER_URL = "http://pickle.bam.net:8572"
        val ACCENT: Int = Color.rgb(37, 110, 112)
    }

    private enum class AppPage {
        Record,
        Review,
        Detail,
        Settings,
    }

    private enum class ButtonStyle {
        Primary,
        Secondary,
        Ghost,
        Danger,
    }
}
