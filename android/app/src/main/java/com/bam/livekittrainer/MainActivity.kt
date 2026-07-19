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
import android.content.res.Configuration
import android.os.Build
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
import android.view.WindowInsetsController
import android.view.inputmethod.EditorInfo
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import java.io.File
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
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
    private lateinit var bulkWakePlacementsInput: EditText
    private var currentPage: AppPage = AppPage.Record
    private var lastRenderedPage: AppPage? = null
    private var lastRenderedDetailId: String? = null
    private var bulkScriptRevision: Int = 0
    private var reprocessing: Boolean = false
    private var darkMode: Boolean = false
    private var appearanceMode: String = APPEARANCE_SYSTEM
    private var selectedProjectId: String? = null
    private var selectedBulkRecordingId: String? = null
    private var activeProject: WakeWordProject? = null
    // True while the shared recorder is capturing a background-noise take rather
    // than a bulk script, so the two record buttons show independent state.
    private var backgroundRecordingActive: Boolean = false
    // True while the shared recorder is capturing a test take (Test page) rather
    // than a training take, so its record button tracks state independently.
    private var testRecordingActive: Boolean = false
    // Separate shuffle counter for the test script so reshuffling it never
    // disturbs the training bulk script shown on the Record page.
    private var testScriptRevision: Int = 0
    private var uploadingTestTakes: Boolean = false
    private var bulkReviewProjectSlug: String? = null
    private var bulkReviewClips: List<BulkReviewClip> = emptyList()
    // The server's authoritative recording list for the active wake word,
    // including takes captured on other devices. Null slug means not loaded yet.
    private var serverRecordingsSlug: String? = null
    private var serverRecordings: List<ServerRecording> = emptyList()
    private var loadingServerRecordings: Boolean = false
    private var projectCounts: Map<String, ProjectCounts> = emptyMap()
    private var bulkAlignmentProjectSlug: String? = null
    private var bulkAlignment: BulkAlignment? = null
    private var loadingBulkReview: Boolean = false
    private var loadingBulkAlignment: Boolean = false
    private var processingBulkSplit: Boolean = false
    private var alignmentPlaybackMs: Int = 0
    private var alignmentPlaybackStartUptimeMs: Long = 0L
    private var lastScheduledAlignmentBoundaryMs: Int = -1
    // Model-test scoring state for the Test page. The score curve is fixed once
    // loaded; the threshold slider only re-interprets it, so slider drags update
    // the curve view and counts label in place rather than re-rendering the page.
    private var scoreProjectSlug: String? = null
    private var scoreRecordingId: String? = null
    private var scoreResult: ScoreResult? = null
    private var loadingScore: Boolean = false
    private var scoreThreshold: Double = 0.5
    // Minimum plateau width (ms) an above-threshold run must span to count as a
    // detection. A real wake word paints a wide plateau; a spurious blip is a
    // tick or two. Filters those spikes without touching the curve. See
    // [ScoreEvents].
    private var scoreWindowMs: Double = 150.0
    // Scoring mode: "full" = continuous rolling window (honest streaming test),
    // "reset" = silence-padded per step (matches isolated-clip training).
    private var scoreMode: String = "full"
    private var scoreCurveView: ScoreCurveView? = null
    private var scoreCountsText: TextView? = null
    // Test-page take-list collapse state. Test takes are the ones you usually
    // score, so they start open; the larger training pool starts collapsed. The
    // list is also paged so a big library doesn't bury the score card.
    private var testTakesExpanded: Boolean = true
    private var trainingTakesExpanded: Boolean = false
    private var testTakesShown: Int = TAKE_PAGE_SIZE
    private var trainingTakesShown: Int = TAKE_PAGE_SIZE
    private var scorePlaybackTicker: Runnable? = null
    private var player: MediaPlayer? = null
    private var activePlaybackKey: String? = null
    private val playbackHandler = Handler(Looper.getMainLooper())
    private var alignmentTicker: Runnable? = null
    private var statusMessage: String = ""
    // Groups every take recorded in this app sitting under one id so the trainer
    // can tell which clips were captured together. New process launch, new id.
    private val sessionId: String = UUID.randomUUID().toString()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = ProjectStore(this)
        recorder = WavRecorder(this)
        exporter = BundleExporter(this)
        lexicon = PromptLexicon(this)
        val prefs = getSharedPreferences(SYNC_PREFS, Context.MODE_PRIVATE)
        appearanceMode = prefs.getString(KEY_APPEARANCE, null)
            ?: if (prefs.getBoolean(KEY_DARK_MODE, false)) APPEARANCE_DARK else APPEARANCE_SYSTEM
        darkMode = resolveDarkMode()

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
        applyBarAppearance()
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
            AppPage.Test -> renderTestPage(project)
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
        bottomNav.addView(navTab("Test", "◇", AppPage.Test))
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
        val backgroundRecordings = store.loadBackgroundRecordings(project.id)
        workspace.addView(backgroundNoiseCard(project, backgroundRecordings).withTop(dp(12)))
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
        ensureServerRecordings(project)
        val bulkRecordings = store.loadBulkRecordings(project.id)
        val backgroundRecordings = store.loadBackgroundRecordings(project.id)
        val serverForSlug =
            if (serverRecordingsSlug == project.slug) serverRecordings else emptyList()
        workspace.addView(syncCard(project, bulkRecordings))
        workspace.addView(
            recordingsCard(
                project,
                bulkRecordings,
                // Test takes are managed on the Test page, not the training pool.
                serverForSlug.filter { !it.isBackground && !it.isTest },
            ).withTop(dp(12)),
        )
        val serverBackground = serverForSlug.filter { it.isBackground }
        if (backgroundRecordings.isNotEmpty() || serverBackground.isNotEmpty()) {
            workspace.addView(
                backgroundRecordingsCard(backgroundRecordings, serverBackground).withTop(dp(12)),
            )
        }
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
            ?: serverRecordings
                .firstOrNull { it.id == selectedBulkRecordingId && it.isBackground.not() }
                ?.let { serverRecordingAsBulk(project, it) }
        if (recording == null) {
            workspace.addView(emptyCard("Bulk recording not found."))
            maybeStatus()
            return
        }
        workspace.addView(bulkRecordingDetailCard(project, recording))
        maybeStatus()
    }

    private fun renderTestPage(project: WakeWordProject?) {
        // Rebuilding the page drops the old curve view; forget the stale refs so
        // in-place slider/playhead updates never touch a detached view.
        scoreCurveView = null
        scoreCountsText = null
        workspace.removeAllViews()
        workspace.addView(topBar("Test"))
        if (project == null) {
            workspace.addView(emptyCard("Create a project and record a take first, then test the model against it."))
            maybeStatus()
            return
        }
        ensureServerRecordings(project)
        workspace.addView(testRecorderCard(project))
        // The score card sits directly under the recorder, above the take list,
        // so tapping Score on a take doesn't leave the result buried far below a
        // long list. The list is collapsible/paged beneath it.
        val result = scoreResult?.takeIf { scoreProjectSlug == project.slug }
        if (result != null || loadingScore) {
            workspace.addView(scoreResultCard(project, result).withTop(dp(12)))
        }
        workspace.addView(testPickerCard(project).withTop(dp(12)))
        maybeStatus()
    }

    private fun testRecorderCard(project: WakeWordProject): View {
        val wakePlacements = savedBulkWakePlacements()
        val scriptContent = PromptGenerator.bulkScriptContent(
            project,
            lexicon,
            store.promptBatch(project.id),
            // Offset the batch seed so the test script differs from the training
            // script even at the same shuffle revision.
            testScriptRevision + 7919,
            wakePlacements,
            positiveDense = savedPositiveDense(),
        )
        val script = scriptContent.text
        val recordingThisProject =
            recorder.isRecording && testRecordingActive && activeProject?.id == project.id
        val localTests = store.loadTestRecordings(project.id)
        val serverIds = if (serverRecordingsSlug == project.slug) {
            serverRecordings.map { it.id }.toSet()
        } else {
            emptySet()
        }
        val pending = localTests.count { it.id !in serverIds }
        return card().apply {
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    addView(
                        text("Record a test take", 20f, textColor(), Typeface.BOLD).apply {
                            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                        },
                    )
                    addView(actionButton("Shuffle", ButtonStyle.Secondary) {
                        testScriptRevision += 1
                        statusMessage = "Generated a new test script"
                        render()
                    }.apply { isEnabled = !recordingThisProject })
                },
            )
            addView(
                text(
                    "Read this in one take, then upload it to score the model. Test takes are transcribed for scoring only — never sliced into training data.",
                    13f,
                    mutedColor(),
                ).withTop(dp(4)),
            )
            addView(bulkScriptText(scriptContent, project.phrase).withTop(dp(12)))
            addView(
                actionButton(
                    if (recordingThisProject) "◼  Stop recording" else "●  Record test take",
                    if (recordingThisProject) ButtonStyle.Danger else ButtonStyle.Primary,
                ) {
                    toggleTestRecording(project, script)
                }.tall().withTop(dp(14)),
            )
            if (pending > 0 && !recordingThisProject) {
                addView(
                    actionButton(
                        if (uploadingTestTakes) "Uploading…" else "⇧  Upload $pending test take${if (pending == 1) "" else "s"} & score",
                        ButtonStyle.Secondary,
                    ) {
                        uploadTestTakes(project)
                    }.withTop(dp(10)).apply { isEnabled = !uploadingTestTakes },
                )
            }
        }
    }

    private fun toggleTestRecording(project: WakeWordProject, script: String) {
        if (recorder.isRecording) {
            // Don't let a Test-page tap stop a training take capturing elsewhere.
            if (!testRecordingActive) {
                statusMessage = "Finish the current recording first"
                render()
                return
            }
            val result = recorder.stop()
            activeProject = null
            testRecordingActive = false
            if (result != null) {
                store.addTestRecording(
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
                        capture = captureMetadataFor(result),
                    ),
                )
                testScriptRevision += 1
                statusMessage = "Saved test take — upload it to score the model"
            }
            render()
            return
        }

        try {
            recorder.startTest(project, script)
            activeProject = project
            testRecordingActive = true
            statusMessage = "Recording test take"
            render()
        } catch (error: IllegalStateException) {
            statusMessage = error.message ?: "Could not start test recording"
            render()
        }
    }

    /**
     * Upload every local test take the server does not yet hold, then score the
     * newest one. Test takes travel in their own manifest array and never enter
     * the training pool, so this is safe to run against a live wake word.
     */
    private fun uploadTestTakes(project: WakeWordProject) {
        val serverUrl = savedServerUrl()
        if (serverUrl.isBlank()) {
            statusMessage = "Set a sync server URL in Settings first"
            currentPage = AppPage.Settings
            render()
            return
        }
        val localTests = store.loadTestRecordings(project.id)
        if (localTests.isEmpty()) {
            statusMessage = "Record a test take first"
            render()
            return
        }
        uploadingTestTakes = true
        statusMessage = "Uploading test takes…"
        render()

        Thread {
            try {
                val client = BundleSyncClient(serverUrl)
                val serverChecksums = try {
                    client.loadServerChecksums(project.slug)
                } catch (_: Exception) {
                    emptyMap()
                }
                val toUpload = localTests.filter {
                    recordingNeedsUpload(it.id, it.filePath, serverChecksums)
                }
                if (toUpload.isNotEmpty()) {
                    val zip = exporter.exportProjectZip(
                        project,
                        emptyList(),
                        emptyList(),
                        emptyList(),
                        toUpload,
                    )
                    client.upload(zip)
                }
                val recordings = try {
                    client.loadServerRecordings(project.slug)
                } catch (_: Exception) {
                    serverRecordings
                }
                // Score the newest test take now that it is transcribed server-side.
                val newest = localTests.maxByOrNull { it.recordedAtMillis }?.id
                runOnUiThread {
                    serverRecordingsSlug = project.slug
                    serverRecordings = recordings
                    uploadingTestTakes = false
                    statusMessage = if (toUpload.isEmpty()) {
                        "Test takes already uploaded"
                    } else {
                        "Uploaded ${toUpload.size} test take${if (toUpload.size == 1) "" else "s"}"
                    }
                    render()
                    if (newest != null && recordings.any { it.id == newest }) {
                        loadScore(project, newest)
                    }
                }
            } catch (error: Exception) {
                runOnUiThread {
                    uploadingTestTakes = false
                    statusMessage = error.message ?: "Upload failed"
                    render()
                }
            }
        }.start()
    }

    private fun testPickerCard(project: WakeWordProject): View {
        val recordings = if (serverRecordingsSlug == project.slug) {
            serverRecordings.filter { !it.isBackground }
        } else {
            emptyList()
        }
        val testTakes = recordings.filter { it.isTest }
        val trainingTakes = recordings.filter { !it.isTest }
        return card().apply {
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    addView(
                        text("Model test", 20f, textColor(), Typeface.BOLD).apply {
                            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                        },
                    )
                    addView(
                        actionButton(
                            if (loadingServerRecordings) "Refreshing…" else "Refresh",
                            ButtonStyle.Secondary,
                        ) {
                            reloadServerRecordings(project)
                        }.apply { isEnabled = !loadingServerRecordings },
                    )
                },
            )
            addView(
                text(
                    "Score a recorded take against the trained model in continuous mode — the honest streaming test. " +
                        "Green markers are wake words the model caught; red are misses. Refresh pulls in takes synced " +
                        "from other devices.",
                    14f,
                    mutedColor(),
                ).withTop(dp(4)),
            )
            when {
                loadingServerRecordings && recordings.isEmpty() ->
                    addView(text("Loading recordings…", 14f, mutedColor()).withTop(dp(12)))
                recordings.isEmpty() ->
                    addView(text("No recordings on the server yet for this wake word.", 14f, mutedColor()).withTop(dp(12)))
                else -> {
                    addTakeGroup(
                        project,
                        title = "Test takes",
                        takes = testTakes,
                        expanded = testTakesExpanded,
                        shown = testTakesShown,
                        onToggle = { testTakesExpanded = !testTakesExpanded; render() },
                        onShowMore = { testTakesShown += TAKE_PAGE_SIZE; render() },
                    )
                    addTakeGroup(
                        project,
                        title = "Training takes",
                        takes = trainingTakes,
                        expanded = trainingTakesExpanded,
                        shown = trainingTakesShown,
                        onToggle = { trainingTakesExpanded = !trainingTakesExpanded; render() },
                        onShowMore = { trainingTakesShown += TAKE_PAGE_SIZE; render() },
                    )
                }
            }
        }
    }

    /**
     * A collapsible, paged group of takes. The header shows the count and a
     * chevron; tapping it toggles the body. When open, only the first [shown]
     * rows render, with a "Show more" footer so a big library never buries the
     * score card above it.
     */
    private fun LinearLayout.addTakeGroup(
        project: WakeWordProject,
        title: String,
        takes: List<ServerRecording>,
        expanded: Boolean,
        shown: Int,
        onToggle: () -> Unit,
        onShowMore: () -> Unit,
    ) {
        if (takes.isEmpty()) return
        addView(
            LinearLayout(this@MainActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(dp(4), dp(10), dp(4), dp(4))
                setOnClickListener { onToggle() }
                addView(
                    text(if (expanded) "▾" else "▸", 15f, mutedColor(), Typeface.BOLD),
                )
                addView(
                    text("  $title", 15f, textColor(), Typeface.BOLD).apply {
                        layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                    },
                )
                addView(text("${takes.size}", 14f, mutedColor(), Typeface.BOLD))
            }.withTop(dp(8)),
        )
        if (!expanded) return
        takes.take(shown).forEach { recording ->
            addView(testRecordingRow(project, recording).withTop(dp(8)))
        }
        if (takes.size > shown) {
            addView(
                actionButton("Show ${takes.size - shown} more", ButtonStyle.Secondary) { onShowMore() }
                    .withTop(dp(8)),
            )
        }
    }

    private fun testRecordingRow(project: WakeWordProject, recording: ServerRecording): View {
        val selected = scoreRecordingId == recording.id
        val subtitle = buildString {
            append("${recording.durationMs / 1000}s · ${formatRecordedAt(recording.recordedAtMillis)}")
            recording.deviceLabel?.let { append(" · $it") }
        }
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(12), dp(10), dp(12), dp(10))
            background = rounded(if (selected) navActiveColor() else promptColor(), dp(12), strokeColor())
            // Long-press a test take to delete it from the server (and this
            // device). Training takes are managed on the Review page instead.
            if (recording.isTest) {
                setOnLongClickListener {
                    confirmDeleteTestTake(project, recording)
                    true
                }
            }
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.VERTICAL
                    layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                    addView(
                        text(
                            if (recording.isTest) "${recording.id.takeLast(8)}  · TEST" else recording.id.takeLast(8),
                            15f,
                            if (recording.isTest) ACCENT else textColor(),
                            Typeface.BOLD,
                        ),
                    )
                    addView(text(subtitle, 12f, mutedColor()).withTop(dp(2)))
                },
            )
            val busy = loadingScore && selected
            addView(
                actionButton(if (busy) "Scoring…" else "Score", ButtonStyle.Primary) {
                    loadScore(project, recording.id)
                }.apply { isEnabled = !loadingScore },
            )
        }
    }

    private fun confirmDeleteTestTake(project: WakeWordProject, recording: ServerRecording) {
        AlertDialog.Builder(this)
            .setTitle("Delete this test take?")
            .setMessage("Removes it from the server and this device. This cannot be undone.")
            .setNegativeButton("Cancel", null)
            .setPositiveButton("Delete") { _, _ -> deleteTestTake(project, recording) }
            .show()
    }

    private fun deleteTestTake(project: WakeWordProject, recording: ServerRecording) {
        stopScorePlayback()
        // Drop the local copy this device holds, if any, so the two stay in step.
        store.loadTestRecordings(project.id)
            .firstOrNull { it.id == recording.id }
            ?.let {
                File(it.filePath).delete()
                store.deleteTestRecording(it)
            }
        serverRecordings = serverRecordings.filterNot { it.id == recording.id }
        if (scoreRecordingId == recording.id) {
            scoreResult = null
            scoreRecordingId = null
        }
        val serverUrl = savedServerUrl()
        statusMessage = "Deleting test take…"
        render()
        if (serverUrl.isBlank()) {
            statusMessage = "Set a sync server URL in Settings first"
            render()
            return
        }
        Thread {
            try {
                val client = BundleSyncClient(serverUrl)
                client.deleteRecording(project.slug, recording.id)
                val recordings = try {
                    client.loadServerRecordings(project.slug)
                } catch (_: Exception) {
                    serverRecordings
                }
                runOnUiThread {
                    serverRecordingsSlug = project.slug
                    serverRecordings = recordings
                    statusMessage = "Deleted test take"
                    render()
                }
            } catch (error: Exception) {
                runOnUiThread {
                    statusMessage = error.message ?: "Delete failed"
                    render()
                }
            }
        }.start()
    }

    private fun scoreResultCard(project: WakeWordProject, result: ScoreResult?): View {
        return card().apply {
            if (result == null) {
                addView(text("Scoring…", 20f, textColor(), Typeface.BOLD))
                addView(
                    text(
                        "Replaying the take through the model. This can take a minute for a long recording.",
                        14f,
                        mutedColor(),
                    ).withTop(dp(6)),
                )
                return@apply
            }

            addView(text("Score  ${result.sourceRecording.takeLast(8)}", 20f, textColor(), Typeface.BOLD))
            val modeName = if (result.mode == "reset") "padded" else "continuous"
            addView(
                text(
                    "“${result.phrase}” · $modeName mode · ${"%.1f".format(result.durationMs / 1000.0)}s",
                    13f,
                    mutedColor(),
                ).withTop(dp(2)),
            )
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    addView(
                        actionButton(
                            "Continuous",
                            if (scoreMode == "full") ButtonStyle.Primary else ButtonStyle.Secondary,
                        ) {
                            if (scoreMode != "full") {
                                scoreMode = "full"
                                scoreRecordingId?.let { loadScore(project, it) }
                            }
                        }.apply {
                            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                            isEnabled = !loadingScore
                        },
                    )
                    addView(
                        actionButton(
                            "Padded",
                            if (scoreMode == "reset") ButtonStyle.Primary else ButtonStyle.Secondary,
                        ) {
                            if (scoreMode != "reset") {
                                scoreMode = "reset"
                                scoreRecordingId?.let { loadScore(project, it) }
                            }
                        }.apply {
                            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                            isEnabled = !loadingScore
                        }.withLeft(dp(8)),
                    )
                }.withTop(dp(10)),
            )

            val curve = ScoreCurveView(this@MainActivity).apply {
                setColors(
                    curve = ACCENT,
                    grid = strokeColor(),
                    hit = Color.rgb(30, 132, 73),
                    miss = Color.rgb(190, 45, 45),
                    threshold = mutedColor(),
                    label = mutedColor(),
                )
                setData(result.timesMs, result.scores, result.targets, result.durationMs)
                setThreshold(scoreThreshold)
                setWindow(scoreWindowMs)
                layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
            }
            scoreCurveView = curve
            addView(curve.withTop(dp(12)))

            val counts = text(scoreCountsSummary(result, scoreThreshold), 15f, textColor(), Typeface.BOLD)
            scoreCountsText = counts
            addView(counts.withTop(dp(10)))

            addView(
                text(
                    "Bands = where LiveKit fires; ▾ blue = where Whisper heard the phrase.  " +
                        "Orange = both agree.  Purple = model caught it, Whisper missed (model-only win).  " +
                        "Grey = low-confidence false alarm.",
                    12f,
                    mutedColor(),
                ).withTop(dp(6)),
            )

            addView(text("Detection threshold  ${"%.2f".format(scoreThreshold)}", 13f, mutedColor()).withTop(dp(10)))
            val thresholdLabel = getChildAt(childCount - 1) as TextView
            addView(
                SeekBar(this@MainActivity).apply {
                    max = 100
                    progress = (scoreThreshold * 100).toInt().coerceIn(0, 100)
                    setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                        override fun onProgressChanged(bar: SeekBar, value: Int, fromUser: Boolean) {
                            scoreThreshold = value / 100.0
                            thresholdLabel.text = "Detection threshold  ${"%.2f".format(scoreThreshold)}"
                            scoreCurveView?.setThreshold(scoreThreshold)
                            scoreCountsText?.text = scoreCountsSummary(result, scoreThreshold)
                        }

                        override fun onStartTrackingTouch(bar: SeekBar) {}
                        override fun onStopTrackingTouch(bar: SeekBar) {}
                    })
                }.withTop(dp(4)),
            )

            addView(
                text("Detection window  ${scoreWindowMs.toInt()} ms", 13f, mutedColor())
                    .withTop(dp(10)),
            )
            val widthLabel = getChildAt(childCount - 1) as TextView
            addView(
                SeekBar(this@MainActivity).apply {
                    max = 1000
                    progress = scoreWindowMs.toInt().coerceIn(0, 1000)
                    setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                        override fun onProgressChanged(bar: SeekBar, value: Int, fromUser: Boolean) {
                            scoreWindowMs = value.toDouble()
                            widthLabel.text = "Detection window  $value ms"
                            scoreCurveView?.setWindow(scoreWindowMs)
                            scoreCountsText?.text = scoreCountsSummary(result, scoreThreshold)
                        }

                        override fun onStartTrackingTouch(bar: SeekBar) {}
                        override fun onStopTrackingTouch(bar: SeekBar) {}
                    })
                }.withTop(dp(4)),
            )

            val playbackKey = "score:${result.sourceRecording}"
            val playing = activePlaybackKey == playbackKey && player?.isPlaying == true
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    addView(
                        actionButton(if (playing) "◼  Stop playback" else "▶  Play take", ButtonStyle.Secondary) {
                            playScoreSource(project, result)
                        }.apply {
                            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                        },
                    )
                    addView(
                        actionButton(if (loadingScore) "…" else "↻  Re-score fresh", ButtonStyle.Secondary) {
                            scoreRecordingId?.let { loadScore(project, it, forceFresh = true) }
                        }.apply {
                            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                            isEnabled = !loadingScore
                        }.withLeft(dp(8)),
                    )
                }.withTop(dp(12)),
            )
            addView(
                text(
                    "Slide the threshold to see how many wake words the model keeps and how many false alarms it adds. " +
                        "Detection window is the decision width: slide a window this wide, and wherever the model's peak " +
                        "inside it clears the threshold, that whole window counts as a fire (a band). Widen it to pool " +
                        "context and merge nearby fires; narrow it to tighten to the raw crossings. Re-score fresh bypasses " +
                        "the cached curve so a changed backend is re-run. Nothing here is added to your training data.",
                    12f,
                    mutedColor(),
                ).withTop(dp(10)),
            )
        }
    }

    /**
     * True positives / misses / false alarms recomputed live at [threshold] and
     * the current plateau-width gate. Both sliders stay live client-side without
     * re-hitting the server — the curve is fixed, only its reading changes.
     */
    private fun scoreCountsSummary(result: ScoreResult, threshold: Double): String {
        val events = ScoreEvents.events(result.timesMs, result.scores, threshold, scoreWindowMs)
        val t = ScoreEvents.tally(result.targets, events)
        val base = "Detected ${t.detected}/${result.targets.size} · missed ${t.missed} · false alarms ${t.falseAlarms}"
        // Only mention the model-only wins when there are any, so the common case
        // stays uncluttered.
        return if (t.modelOnly > 0) "$base · model-only ${t.modelOnly}" else base
    }

    private fun loadScore(project: WakeWordProject, recordingId: String, forceFresh: Boolean = false) {
        val serverUrl = savedServerUrl()
        if (serverUrl.isBlank()) {
            statusMessage = "Set a sync server URL in Settings first"
            currentPage = AppPage.Settings
            render()
            return
        }
        stopScorePlayback()
        loadingScore = true
        scoreProjectSlug = project.slug
        scoreRecordingId = recordingId
        scoreResult = null
        val mode = scoreMode
        val modeLabel = if (mode == "reset") "padded" else "continuous"
        statusMessage = if (forceFresh) {
            "Re-scoring ${recordingId.takeLast(8)} fresh in $modeLabel mode…"
        } else {
            "Scoring ${recordingId.takeLast(8)} in $modeLabel mode…"
        }
        render()

        Thread {
            try {
                val result = BundleSyncClient(serverUrl)
                    .loadScore(project.slug, recordingId, mode, scoreThreshold, noCache = forceFresh)
                runOnUiThread {
                    scoreProjectSlug = project.slug
                    scoreResult = result
                    loadingScore = false
                    val events = ScoreEvents.events(result.timesMs, result.scores, scoreThreshold, scoreWindowMs)
                    val detected = ScoreEvents.detectedFlags(result.targets, events).count { it }
                    statusMessage =
                        "Detected $detected/${result.targets.size} wake words in $modeLabel mode"
                    render()
                }
            } catch (error: Exception) {
                runOnUiThread {
                    loadingScore = false
                    statusMessage = error.message ?: "Scoring failed"
                    render()
                }
            }
        }.start()
    }

    private fun playScoreSource(project: WakeWordProject, result: ScoreResult) {
        val playbackKey = "score:${result.sourceRecording}"
        player?.let { current ->
            if (activePlaybackKey == playbackKey) {
                if (current.isPlaying) {
                    current.pause()
                    stopScoreTicker()
                } else {
                    current.start()
                    startScoreTicker()
                }
                render()
                return
            }
        }
        stopScorePlayback()
        activePlaybackKey = playbackKey
        try {
            val url = BundleSyncClient(savedServerUrl())
                .sourceAudioUrl(project.slug, result.sourceRecording)
            player = MediaPlayer().apply {
                setDataSource(url)
                setOnPreparedListener {
                    it.start()
                    startScoreTicker()
                    render()
                }
                setOnCompletionListener {
                    stopScoreTicker()
                    it.release()
                    if (player === it) player = null
                    activePlaybackKey = null
                    scoreCurveView?.setPlayhead(-1.0)
                    render()
                }
                prepareAsync()
            }
            statusMessage = "Loading take audio"
        } catch (error: Exception) {
            activePlaybackKey = null
            player = null
            statusMessage = error.message ?: "Could not play take"
        }
        render()
    }

    private fun startScoreTicker() {
        stopScoreTicker()
        val ticker = object : Runnable {
            override fun run() {
                val active = player ?: return
                scoreCurveView?.setPlayhead(active.currentPosition.toDouble())
                if (active.isPlaying) {
                    playbackHandler.postDelayed(this, 60)
                }
            }
        }
        scorePlaybackTicker = ticker
        playbackHandler.post(ticker)
    }

    private fun stopScoreTicker() {
        scorePlaybackTicker?.let { playbackHandler.removeCallbacks(it) }
        scorePlaybackTicker = null
    }

    private fun stopScorePlayback() {
        stopScoreTicker()
        player?.release()
        player = null
        activePlaybackKey = null
        scoreCurveView?.setPlayhead(-1.0)
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
            addView(text("Bulk wake placements", 15f, mutedColor()).withTop(dp(14)))
            addView(bulkWakePlacementsInput.withTop(dp(6)))
            addView(actionButton("Save settings", ButtonStyle.Primary) {
                saveSettings(
                    serverUrlInput.text.toString().trim(),
                    bulkWakePlacementsInput.text.toString().trim(),
                )
            }.withTop(dp(10)))
            addView(actionButton("Load server projects", ButtonStyle.Secondary) {
                loadServerProjects(serverUrlInput.text.toString().trim())
            }.withTop(dp(8)))
            val dense = savedPositiveDense()
            addView(text("Bulk script style", 15f, mutedColor()).withTop(dp(14)))
            addView(
                text(
                    if (dense) {
                        "Dense: short, varied ways to say the wake phrase back to back, with frequent near misses. Many positives per minute."
                    } else {
                        "Prose: wake phrase woven into full sentences. More natural negatives, but slower to gather positives."
                    },
                    12f,
                    mutedColor(),
                ).withTop(dp(4)),
            )
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    addView(actionButton("Prose", if (!dense) ButtonStyle.Primary else ButtonStyle.Secondary) { setPositiveDense(false) }.weight1())
                    addView(actionButton("Dense", if (dense) ButtonStyle.Primary else ButtonStyle.Secondary) { setPositiveDense(true) }.weight1().withLeft(dp(8)))
                }.withTop(dp(8)),
            )
            addView(text("Appearance", 15f, mutedColor()).withTop(dp(14)))
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    addView(actionButton("System", if (appearanceMode == APPEARANCE_SYSTEM) ButtonStyle.Primary else ButtonStyle.Secondary) { setAppearance(APPEARANCE_SYSTEM) })
                    addView(actionButton("Light", if (appearanceMode == APPEARANCE_LIGHT) ButtonStyle.Primary else ButtonStyle.Secondary) { setAppearance(APPEARANCE_LIGHT) }.withLeft(dp(8)))
                    addView(actionButton("Dark", if (appearanceMode == APPEARANCE_DARK) ButtonStyle.Primary else ButtonStyle.Secondary) { setAppearance(APPEARANCE_DARK) }.withLeft(dp(8)))
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
                    "Upload your bulk and background takes; the server transcribes and slices them into training clips.",
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
            projectCounts[project.slug]?.let { counts ->
                addView(trainingPoolCard(counts).withTop(dp(12)))
            }
        }
    }

    /**
     * Shows this wake word's own clip tallies alongside the negatives pooled in
     * from every other project. Negatives are shared and pile up fast, so this
     * surfaces how many real positives still need collecting for a balanced set.
     */
    private fun trainingPoolCard(counts: ProjectCounts): View {
        val ownNegativeTotal = counts.negative + counts.pooledNegative
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(14), dp(12), dp(14), dp(12))
            background = rounded(promptColor(), dp(14), 0)
            addView(text("Training pool", 15f, textColor(), Typeface.BOLD))
            addView(
                text(
                    "${counts.positive} positive · ${counts.negative} negative · ${counts.background} background (this word)",
                    13f,
                    mutedColor(),
                ).withTop(dp(4)),
            )
            addView(
                text(
                    "+${counts.pooledNegative} negatives reused from other words → $ownNegativeTotal total negatives for training",
                    13f,
                    mutedColor(),
                ).withTop(dp(2)),
            )
            if (counts.positive > 0 && ownNegativeTotal >= counts.positive * 5) {
                val ratio = ownNegativeTotal / counts.positive
                addView(
                    text(
                        "Positives are heavily outnumbered (~1:$ratio). Record more wake takes; training can also overweight positives via batch_n_per_class.",
                        13f,
                        Color.parseColor("#C2410C"),
                    ).withTop(dp(6)),
                )
            } else if (counts.positive == 0) {
                addView(
                    text(
                        "No positives yet for this word. Record a bulk take with the wake phrase.",
                        13f,
                        Color.parseColor("#C2410C"),
                    ).withTop(dp(6)),
                )
            }
        }
    }

    private fun recordingsCard(
        project: WakeWordProject,
        localBulk: List<BulkRecording>,
        serverBulk: List<ServerRecording>,
    ): View {
        // The server is the master list. Local takes not yet on the server are
        // shown as pending so nothing captured here is invisible either.
        val serverIds = serverBulk.map { it.id }.toSet()
        val pending = localBulk.filter { it.id !in serverIds }
        val total = serverBulk.size + pending.size
        return card().apply {
            addView(text("Bulk recordings  $total", 18f, textColor(), Typeface.BOLD))
            if (total == 0) {
                val message = if (loadingServerRecordings) {
                    "Loading recordings from the server…"
                } else {
                    "No bulk recordings yet. Record one on the Record tab."
                }
                addView(text(message, 14f, mutedColor()).withTop(dp(8)))
            } else {
                serverBulk.forEach { recording ->
                    val localMatch = localBulk.firstOrNull { it.id == recording.id }
                    addView(serverRecordingRow(project, recording, localMatch).withTop(dp(8)))
                }
                pending.forEach { recording ->
                    addView(bulkRecordingRow(recording).withTop(dp(8)))
                }
            }
        }
    }

    /**
     * A recording as the server sees it, so it renders the same whether it was
     * captured on this device or another one. Shows the capturing device, slice
     * tallies, and a Delete that removes it from the server (and the local copy
     * if this device happens to have one).
     */
    private fun serverRecordingRow(
        project: WakeWordProject,
        recording: ServerRecording,
        localMatch: BulkRecording?,
    ): View {
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
            addView(text(formatRecordedAt(recording.recordedAtMillis), 12f, mutedColor()).withTop(dp(2)))
            addView(text(deviceAttribution(recording), 12f, mutedColor()).withTop(dp(2)))
            addView(
                text(
                    "${recording.positiveCount} positive · ${recording.negativeCount} negative",
                    12f,
                    mutedColor(),
                ).withTop(dp(2)),
            )
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    addView(actionButton("Open", ButtonStyle.Secondary) { open() }.weight1())
                    addView(
                        actionButton("Delete", ButtonStyle.Ghost) {
                            confirmDeleteServerRecording(project, recording, localMatch)
                        }.weight1().withLeft(dp(8)),
                    )
                }.withTop(dp(8)),
            )
        }
    }

    /** "This phone" for takes from this device, else the capturing device name. */
    private fun deviceAttribution(recording: ServerRecording): String {
        val label = recording.deviceLabel ?: return "Unknown device"
        return if (recording.deviceModel.equals(android.os.Build.MODEL, ignoreCase = true)) {
            "This device · $label"
        } else {
            "From $label"
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
            addView(text(formatRecordedAt(recording.recordedAtMillis), 13f, textColor()).withTop(dp(2)))
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
            positiveDense = savedPositiveDense(),
        )
        val script = scriptContent.text
        val recordingThisProject =
            recorder.isRecording && !backgroundRecordingActive && !testRecordingActive &&
                activeProject?.id == project.id
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
                        capture = captureMetadataFor(result),
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

    /**
     * Combine the device/app/session context this activity knows with the audio
     * route the recorder resolved, into the provenance stored on each take.
     */
    private fun captureMetadataFor(result: WavRecorder.RecordingResult): CaptureMetadata {
        val appVersion = try {
            packageManager.getPackageInfo(packageName, 0).versionName ?: ""
        } catch (_: Exception) {
            ""
        }
        return CaptureMetadata(
            deviceManufacturer = Build.MANUFACTURER ?: "",
            deviceModel = Build.MODEL ?: "",
            osVersion = "Android ${Build.VERSION.RELEASE} (SDK ${Build.VERSION.SDK_INT})",
            appVersion = appVersion,
            inputRoute = result.inputRoute,
            sourceSampleRateHz = result.sourceSampleRateHz,
            sourceChannels = result.sourceChannels,
            sessionId = sessionId,
        )
    }

    private fun backgroundNoiseCard(
        project: WakeWordProject,
        backgroundRecordings: List<BackgroundRecording>,
    ): View {
        val recordingThisProject =
            recorder.isRecording && backgroundRecordingActive && activeProject?.id == project.id
        val totalSeconds = backgroundRecordings.sumOf { it.durationMs } / 1000
        return card().apply {
            addView(text("Background noise", 20f, textColor(), Typeface.BOLD))
            addView(
                text(
                    "Record the room, silence, appliances, typing, TV — anything that is not speech. The server slices it into short background clips the model trains against.",
                    13f,
                    mutedColor(),
                ).withTop(dp(4)),
            )
            addView(
                text(
                    "${backgroundRecordings.size} saved takes · ${totalSeconds}s captured",
                    13f,
                    mutedColor(),
                ).withTop(dp(10)),
            )
            addView(
                actionButton(
                    if (recordingThisProject) "◼  Stop noise capture" else "●  Record background",
                    if (recordingThisProject) ButtonStyle.Danger else ButtonStyle.Secondary,
                ) {
                    toggleBackgroundRecording(project)
                }.tall().withTop(dp(14)),
            )
        }
    }

    private fun toggleBackgroundRecording(project: WakeWordProject) {
        if (recorder.isRecording) {
            // Ignore taps on the noise button while a bulk take is capturing.
            if (!backgroundRecordingActive) {
                statusMessage = "Finish the bulk recording before capturing background noise"
                render()
                return
            }
            val result = recorder.stop()
            activeProject = null
            backgroundRecordingActive = false
            if (result != null) {
                store.addBackgroundRecording(
                    BackgroundRecording(
                        id = result.output.nameWithoutExtension,
                        projectId = project.id,
                        projectSlug = project.slug,
                        filePath = result.output.absolutePath,
                        recordedAtMillis = result.recordedAtMillis,
                        durationMs = result.durationMs,
                        sampleRateHz = result.sampleRateHz,
                        channels = result.channels,
                        encoding = result.encoding,
                        capture = captureMetadataFor(result),
                    ),
                )
                statusMessage = "Saved background take ${result.output.name}"
            }
            render()
            return
        }

        try {
            recorder.startBackground(project)
            activeProject = project
            backgroundRecordingActive = true
            statusMessage = "Recording background noise"
            render()
        } catch (error: IllegalStateException) {
            statusMessage = error.message ?: "Could not start background recording"
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
                    BundleSyncClient(serverUrl)
                        .deleteRecording(recording.projectSlug, recording.id)
                } catch (_: Exception) {
                    // Local delete already happened; a stale server copy is harmless.
                }
            }.start()
        }
    }

    private fun confirmDeleteServerRecording(
        project: WakeWordProject,
        recording: ServerRecording,
        localMatch: BulkRecording?,
    ) {
        val kind = if (recording.isBackground) "background take" else "recording"
        AlertDialog.Builder(this)
            .setTitle("Delete this $kind?")
            .setMessage("Removes it and its slices from the server for every device. This cannot be undone.")
            .setNegativeButton("Cancel", null)
            .setPositiveButton("Delete") { _, _ ->
                deleteServerRecording(project, recording, localMatch)
            }
            .show()
    }

    private fun deleteServerRecording(
        project: WakeWordProject,
        recording: ServerRecording,
        localMatch: BulkRecording?,
    ) {
        player?.release()
        player = null
        activePlaybackKey = null
        // Remove any local copy this device holds so the two stay in step.
        localMatch?.let {
            File(it.filePath).delete()
            store.deleteBulkRecording(it)
        }
        serverRecordings = serverRecordings.filterNot { it.id == recording.id }
        bulkReviewClips = bulkReviewClips.filterNot { it.sourceRecording == recording.id }
        val serverUrl = savedServerUrl()
        currentPage = AppPage.Review
        statusMessage = "Deleting…"
        render()

        if (serverUrl.isBlank()) {
            statusMessage = "Set a sync server URL in Settings first"
            render()
            return
        }
        Thread {
            try {
                val client = BundleSyncClient(serverUrl)
                client.deleteRecording(project.slug, recording.id)
                val recordings = client.loadServerRecordings(project.slug)
                val counts = try {
                    client.loadProjectCounts()
                } catch (_: Exception) {
                    projectCounts
                }
                runOnUiThread {
                    serverRecordingsSlug = project.slug
                    serverRecordings = recordings
                    projectCounts = counts
                    statusMessage = "Deleted"
                    render()
                }
            } catch (error: Exception) {
                runOnUiThread {
                    statusMessage = error.message ?: "Delete failed"
                    render()
                }
            }
        }.start()
    }

    private fun syncAndProcess(project: WakeWordProject, bulkRecordings: List<BulkRecording>) {
        val serverUrl = savedServerUrl()
        if (serverUrl.isBlank()) {
            statusMessage = "Set a sync server URL in Settings first"
            currentPage = AppPage.Settings
            render()
            return
        }
        val backgroundRecordings = store.loadBackgroundRecordings(project.id)
        if (bulkRecordings.isEmpty() && backgroundRecordings.isEmpty()) {
            statusMessage = "Record a bulk script or background noise first"
            render()
            return
        }

        processingBulkSplit = true
        val backgroundNote =
            if (backgroundRecordings.isEmpty()) "" else " + ${backgroundRecordings.size} background"
        statusMessage = "Syncing ${bulkRecordings.size} bulk recordings$backgroundNote…"
        render()

        Thread {
            try {
                val client = BundleSyncClient(serverUrl)
                // Ask the server which takes it already holds and skip re-uploading
                // any whose bytes are unchanged, so a sync only ships new audio.
                val serverChecksums = try {
                    client.loadServerChecksums(project.slug)
                } catch (_: Exception) {
                    emptyMap()
                }
                val bulkToUpload = bulkRecordings.filter {
                    recordingNeedsUpload(it.id, it.filePath, serverChecksums)
                }
                val backgroundToUpload = backgroundRecordings.filter {
                    recordingNeedsUpload(it.id, it.filePath, serverChecksums)
                }
                val skipped =
                    (bulkRecordings.size - bulkToUpload.size) +
                        (backgroundRecordings.size - backgroundToUpload.size)

                val response = if (bulkToUpload.isEmpty() && backgroundToUpload.isEmpty()) {
                    null
                } else {
                    val zip = exporter.exportProjectZip(
                        project,
                        emptyList(),
                        bulkToUpload,
                        backgroundToUpload,
                    )
                    client.upload(zip)
                }
                val clips = client.loadBulkReview(project.slug)
                val recordings = try {
                    client.loadServerRecordings(project.slug)
                } catch (_: Exception) {
                    serverRecordings
                }
                val counts = try {
                    client.loadProjectCounts()
                } catch (_: Exception) {
                    projectCounts
                }
                runOnUiThread {
                    bulkReviewProjectSlug = project.slug
                    bulkReviewClips = clips
                    serverRecordingsSlug = project.slug
                    serverRecordings = recordings
                    projectCounts = counts
                    bulkAlignmentProjectSlug = null
                    bulkAlignment = null
                    processingBulkSplit = false
                    val skippedNote =
                        if (skipped > 0) " ($skipped already on server, not re-uploaded)" else ""
                    statusMessage = if (response == null) {
                        "Already up to date — nothing new to upload$skippedNote"
                    } else {
                        val alignmentMessage = syncAlignmentMessage(response)
                        if (clips.isEmpty()) {
                            "No clips generated. $alignmentMessage"
                        } else {
                            "Synced and sliced ${clips.size} clips$skippedNote"
                        }
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

    /**
     * Whether a local recording still needs uploading. New to the server (id
     * absent) → yes. Present with a matching checksum → no. Present but the
     * server has no checksum (legacy row) → trust the id and skip. Present with a
     * different checksum (re-recorded under the same id) → yes.
     */
    private fun recordingNeedsUpload(
        id: String,
        filePath: String,
        serverChecksums: Map<String, String?>,
    ): Boolean {
        if (!serverChecksums.containsKey(id)) return true
        val serverSha = serverChecksums[id] ?: return false
        val localSha = sha256OfFile(filePath) ?: return true
        return !serverSha.equals(localSha, ignoreCase = true)
    }

    /** Hex SHA-256 of a file's raw bytes, or null if it cannot be read. */
    private fun sha256OfFile(filePath: String): String? {
        val file = File(filePath)
        if (!file.isFile) return null
        return try {
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { stream ->
                val buffer = ByteArray(64 * 1024)
                while (true) {
                    val read = stream.read(buffer)
                    if (read < 0) break
                    digest.update(buffer, 0, read)
                }
            }
            digest.digest().joinToString("") { "%02x".format(it) }
        } catch (_: Exception) {
            null
        }
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
                val client = BundleSyncClient(serverUrl)
                val message = call(client)
                val clips = client.loadBulkReview(project.slug)
                val recordings = try {
                    client.loadServerRecordings(project.slug)
                } catch (_: Exception) {
                    serverRecordings
                }
                runOnUiThread {
                    bulkReviewProjectSlug = project.slug
                    bulkReviewClips = clips
                    serverRecordingsSlug = project.slug
                    serverRecordings = recordings
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

        loadingServerRecordings = true
        Thread {
            try {
                val client = BundleSyncClient(serverUrl)
                val clips = client.loadBulkReview(project.slug)
                val recordings = try {
                    client.loadServerRecordings(project.slug)
                } catch (_: Exception) {
                    serverRecordings
                }
                val counts = try {
                    client.loadProjectCounts()
                } catch (_: Exception) {
                    projectCounts
                }
                runOnUiThread {
                    bulkReviewProjectSlug = project.slug
                    bulkReviewClips = clips
                    serverRecordingsSlug = project.slug
                    serverRecordings = recordings
                    projectCounts = counts
                    loadingBulkReview = false
                    loadingServerRecordings = false
                    statusMessage = if (clips.isEmpty()) {
                        "No clips yet. Tap Sync & process to transcribe and slice."
                    } else {
                        "Loaded ${clips.size} clips"
                    }
                    render()
                }
            } catch (error: Exception) {
                runOnUiThread {
                    loadingBulkReview = false
                    loadingServerRecordings = false
                    statusMessage = error.message ?: "Load review failed"
                    render()
                }
            }
        }.start()
    }

    /**
     * Fetch the server's recording list once when the Review page opens for a
     * wake word, so takes captured on other devices show up without the user
     * having to tap Refresh. Silent: failures leave the last known list.
     */
    /**
     * Force a re-fetch of the server's recording list, even if this project's
     * list is already loaded — so a device picks up test takes another device
     * synced without relaunching. The server is the master record.
     */
    private fun reloadServerRecordings(project: WakeWordProject) {
        val serverUrl = savedServerUrl()
        if (serverUrl.isBlank() || loadingServerRecordings) return
        loadingServerRecordings = true
        statusMessage = "Refreshing recordings…"
        render()
        Thread {
            try {
                val recordings = BundleSyncClient(serverUrl).loadServerRecordings(project.slug)
                runOnUiThread {
                    serverRecordingsSlug = project.slug
                    serverRecordings = recordings
                    loadingServerRecordings = false
                    statusMessage = "Recordings refreshed"
                    render()
                }
            } catch (error: Exception) {
                runOnUiThread {
                    loadingServerRecordings = false
                    statusMessage = error.message ?: "Refresh failed"
                    render()
                }
            }
        }.start()
    }

    private fun ensureServerRecordings(project: WakeWordProject) {
        val serverUrl = savedServerUrl()
        if (serverUrl.isBlank()) return
        if (loadingServerRecordings || serverRecordingsSlug == project.slug) return
        loadingServerRecordings = true
        Thread {
            try {
                val recordings = BundleSyncClient(serverUrl).loadServerRecordings(project.slug)
                runOnUiThread {
                    serverRecordingsSlug = project.slug
                    serverRecordings = recordings
                    loadingServerRecordings = false
                    render()
                }
            } catch (_: Exception) {
                runOnUiThread {
                    loadingServerRecordings = false
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
                val alignment = BundleSyncClient(serverUrl)
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
                val response = BundleSyncClient(serverUrl).deleteBulkReviewClip(project.slug, clip)
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

    private fun saveSettings(serverUrl: String, bulkWakePlacementsText: String) {
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
            .putInt(KEY_BULK_WAKE_PLACEMENTS, bulkWakePlacements)
            .apply()
        statusMessage = "Saving settings to server"
        render()

        Thread {
            try {
                val response = BundleSyncClient(serverUrl).saveSettings()
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
                val projects = BundleSyncClient(serverUrl).loadProjects()
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


    private fun savedBulkWakePlacements(): Int {
        return getSharedPreferences(SYNC_PREFS, Context.MODE_PRIVATE)
            .getInt(KEY_BULK_WAKE_PLACEMENTS, PromptGenerator.DEFAULT_BULK_WAKE_PLACEMENTS)
            .coerceIn(1, 48)
    }

    private fun savedPositiveDense(): Boolean {
        return getSharedPreferences(SYNC_PREFS, Context.MODE_PRIVATE)
            .getBoolean(KEY_BULK_POSITIVE_DENSE, false)
    }

    private fun setPositiveDense(enabled: Boolean) {
        getSharedPreferences(SYNC_PREFS, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(KEY_BULK_POSITIVE_DENSE, enabled)
            .apply()
        // Re-roll so the visible script switches style immediately.
        bulkScriptRevision += 1
        statusMessage = if (enabled) "Dense positive script" else "Prose script"
        render()
    }

    private fun setAppearance(mode: String) {
        appearanceMode = mode
        darkMode = resolveDarkMode()
        getSharedPreferences(SYNC_PREFS, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_APPEARANCE, mode)
            .remove(KEY_DARK_MODE)
            .apply()
        render()
    }

    private fun systemInDarkMode(): Boolean =
        (resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK) ==
            Configuration.UI_MODE_NIGHT_YES

    private fun resolveDarkMode(): Boolean = when (appearanceMode) {
        APPEARANCE_LIGHT -> false
        APPEARANCE_DARK -> true
        else -> systemInDarkMode()
    }

    // Keep the status- and navigation-bar icons legible: light (dark-on-light)
    // icons for the light theme, and light-colored icons for the dark theme.
    private fun applyBarAppearance() {
        val lightBars = !darkMode
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val mask = WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS or
                WindowInsetsController.APPEARANCE_LIGHT_NAVIGATION_BARS
            window.insetsController?.setSystemBarsAppearance(if (lightBars) mask else 0, mask)
        } else {
            @Suppress("DEPRECATION")
            run {
                var flags = window.decorView.systemUiVisibility
                val bits = View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR or
                    View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR
                flags = if (lightBars) flags or bits else flags and bits.inv()
                window.decorView.systemUiVisibility = flags
            }
        }
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
        stopAlignmentTicker()
        stopScoreTicker()
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

    /**
     * Adapt a server recording into the local model so the detail page can open
     * takes captured on another device. There is no local WAV, so whole-take
     * playback is unavailable, but slices and alignment come from the server.
     */
    private fun serverRecordingAsBulk(project: WakeWordProject, recording: ServerRecording): BulkRecording =
        BulkRecording(
            id = recording.id,
            projectId = project.id,
            projectSlug = project.slug,
            filePath = "",
            script = "",
            recordedAtMillis = recording.recordedAtMillis,
            durationMs = recording.durationMs,
            sampleRateHz = 16_000,
            channels = 1,
            encoding = "pcm_16bit",
        )

    private fun formatRecordedAt(millis: Long): String =
        SimpleDateFormat("MMM d, yyyy · h:mm a", Locale.getDefault()).format(Date(millis))

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
            addView(text(formatRecordedAt(recording.recordedAtMillis), 12f, mutedColor()).withTop(dp(2)))
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

    private fun backgroundRecordingsCard(
        localBackground: List<BackgroundRecording>,
        serverBackground: List<ServerRecording>,
    ): View {
        val serverIds = serverBackground.map { it.id }.toSet()
        val pending = localBackground.filter { it.id !in serverIds }
        val total = serverBackground.size + pending.size
        val totalSeconds =
            (serverBackground.sumOf { it.durationMs } + pending.sumOf { it.durationMs }) / 1000
        return card().apply {
            addView(text("Background takes  $total", 18f, textColor(), Typeface.BOLD))
            addView(
                text(
                    "${totalSeconds}s of ambient audio. The server chops each take into short background clips pooled across every wake word.",
                    13f,
                    mutedColor(),
                ).withTop(dp(4)),
            )
            serverBackground.forEach { recording ->
                val localMatch = localBackground.firstOrNull { it.id == recording.id }
                if (localMatch != null) {
                    addView(backgroundRecordingRow(localMatch).withTop(dp(8)))
                } else {
                    addView(serverBackgroundRow(recording).withTop(dp(8)))
                }
            }
            pending.forEach { recording ->
                addView(backgroundRecordingRow(recording).withTop(dp(8)))
            }
        }
    }

    /** A background take held on the server but not on this device. */
    private fun serverBackgroundRow(recording: ServerRecording): View {
        val project = activeProjectOrNull()
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(14), dp(12), dp(14), dp(12))
            background = rounded(promptColor(), dp(14), 0)
            addView(text("Take  ${recording.durationMs / 1000}s", 15f, textColor(), Typeface.BOLD))
            addView(text(formatRecordedAt(recording.recordedAtMillis), 12f, mutedColor()).withTop(dp(2)))
            addView(text(deviceAttribution(recording), 12f, mutedColor()).withTop(dp(2)))
            addView(text("${recording.backgroundCount} background clips", 12f, mutedColor()).withTop(dp(2)))
            addView(
                actionButton("Delete", ButtonStyle.Ghost) {
                    if (project != null) confirmDeleteServerRecording(project, recording, null)
                }.withTop(dp(8)),
            )
        }
    }

    private fun backgroundRecordingRow(recording: BackgroundRecording): View {
        val playbackKey = "background:${recording.id}"
        val playing = activePlaybackKey == playbackKey && player?.isPlaying == true
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(14), dp(12), dp(14), dp(12))
            background = rounded(promptColor(), dp(14), 0)
            addView(text("Take  ${recording.durationMs / 1000}s", 15f, textColor(), Typeface.BOLD))
            addView(text(formatRecordedAt(recording.recordedAtMillis), 12f, mutedColor()).withTop(dp(2)))
            addView(text(File(recording.filePath).name, 12f, mutedColor()).withTop(dp(2)))
            addView(
                LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    addView(
                        actionButton(if (playing) "Pause" else "Play", ButtonStyle.Secondary) {
                            playBackgroundRecording(recording)
                        }.weight1(),
                    )
                    addView(
                        actionButton("Delete", ButtonStyle.Ghost) {
                            confirmDeleteBackgroundRecording(recording)
                        }.weight1().withLeft(dp(8)),
                    )
                }.withTop(dp(8)),
            )
        }
    }

    private fun playBackgroundRecording(recording: BackgroundRecording) {
        val playbackKey = "background:${recording.id}"
        player?.let { current ->
            if (activePlaybackKey == playbackKey) {
                if (current.isPlaying) current.pause() else current.start()
                render()
                return
            }
        }
        stopAlignmentTicker()
        player?.release()
        player = null
        val source = File(recording.filePath)
        if (!source.isFile) {
            activePlaybackKey = null
            statusMessage = "Background take file is missing"
            render()
            return
        }
        activePlaybackKey = playbackKey
        try {
            player = MediaPlayer().apply {
                setDataSource(source.absolutePath)
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
            statusMessage = "Playing background take"
        } catch (error: Exception) {
            activePlaybackKey = null
            player = null
            statusMessage = error.message ?: "Could not play background take"
        }
        render()
    }

    private fun confirmDeleteBackgroundRecording(recording: BackgroundRecording) {
        AlertDialog.Builder(this)
            .setTitle("Delete this background take?")
            .setMessage("Removes the local take and its clips from the server. This cannot be undone.")
            .setNegativeButton("Cancel", null)
            .setPositiveButton("Delete") { _, _ -> deleteBackgroundRecording(recording) }
            .show()
    }

    private fun deleteBackgroundRecording(recording: BackgroundRecording) {
        if (activePlaybackKey == "background:${recording.id}") {
            player?.release()
            player = null
            activePlaybackKey = null
        }
        File(recording.filePath).delete()
        store.deleteBackgroundRecording(recording)
        bulkReviewClips = bulkReviewClips.filterNot { it.sourceRecording == recording.id }
        val serverUrl = savedServerUrl()
        statusMessage = "Deleted background take"
        render()

        if (serverUrl.isNotBlank()) {
            Thread {
                try {
                    BundleSyncClient(serverUrl)
                        .deleteRecording(recording.projectSlug, recording.id)
                } catch (_: Exception) {
                    // Local delete already happened; a stale server copy is harmless.
                }
            }.start()
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
        const val TAKE_PAGE_SIZE = 5
        const val SYNC_PREFS = "sync"
        const val KEY_SYNC_SERVER_URL = "server_url"
        const val KEY_BULK_WAKE_PLACEMENTS = "bulk_wake_placements"
        const val KEY_BULK_POSITIVE_DENSE = "bulk_positive_dense"
        const val KEY_DARK_MODE = "dark_mode"
        const val KEY_APPEARANCE = "appearance_mode"
        const val APPEARANCE_SYSTEM = "system"
        const val APPEARANCE_LIGHT = "light"
        const val APPEARANCE_DARK = "dark"
        const val DEFAULT_SYNC_SERVER_URL = "http://100.64.0.2:8765"
        val ACCENT: Int = Color.rgb(37, 110, 112)
    }

    private enum class AppPage {
        Record,
        Review,
        Detail,
        Test,
        Settings,
    }

    private enum class ButtonStyle {
        Primary,
        Secondary,
        Ghost,
        Danger,
    }
}
