use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, Path as AxumPath, RawQuery, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use chrono::Utc;
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    env,
    fs::{self, File},
    io::{self, Cursor},
    net::SocketAddr,
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex},
};
use zip::ZipArchive;

mod db;

const POSITIVE_MAX_SECONDS: f64 = 1.5;
// Hard ceiling on the final padded slice length. The context/target budgets
// above bound only the word span; lead/tail padding is added on top, so without
// this every positive ran ~0.3s over. Positives keep their tail (the wake
// phrase ends the clip) so the start is trimmed in; negatives keep their start.
const MAX_SLICE_SECONDS: f64 = 1.5;
// Whisper word timestamps drift from the true audio, so cutting exactly at
// word.start/word.end clips onsets and (worst of all) chops the tail-aligned
// wake phrase. Nudge each cut outward, bounded by the neighboring words, to keep
// slices honest to their transcript. The positive tail is padded hardest because
// a positive that lost its wake phrase is the most damaging error.
const CUT_LEAD_PADDING_SECONDS: f64 = 0.08;
const POSITIVE_TAIL_PADDING_SECONDS: f64 = 0.28;
const NEGATIVE_TAIL_PADDING_SECONDS: f64 = 0.10;
const NEGATIVE_TARGET_SECONDS: f64 = 1.5;
// Ambient background recordings carry no speech to align, so they are chopped
// into fixed windows sized to the trainer's clip_duration (2.0s). Each window
// becomes an independent background training example; a trailing remnant shorter
// than the minimum is dropped rather than padded into a misleadingly short clip.
const BACKGROUND_CHUNK_SECONDS: f64 = 2.0;
const BACKGROUND_MIN_CHUNK_SECONDS: f64 = 1.0;
// Sentinel stored in a recording's `script` column to mark it as a background
// noise take rather than a scripted bulk read. Reprocess branches on this so
// background sources are re-chunked deterministically instead of Whisper-aligned.
const BACKGROUND_SCRIPT_MARKER: &str = "__background_noise__";
// Sentinel stored in a positive take's `script` column when its wake word is a
// non-lexical sound. The app stamps it (from the project's energy-positives
// toggle) so this take is *always* energy-sliced, regardless of what Whisper
// transcribes — this is stronger than the empty-transcript auto-fallback, which
// misses a take where Whisper happens to catch one of many bursts. Must match
// `BulkRecording.ENERGY_POSITIVE_MARKER` in the Android app.
const ENERGY_POSITIVE_SCRIPT_MARKER: &str = "__energy_positive__";

// Energy/VAD fallback for non-lexical positive takes (sounds, not words — e.g. a
// fast "beep beep") where Whisper returns no words, so word-timestamp slicing
// finds nothing. Positives are recorded as repeated sound bursts with ~1s gaps,
// so we segment the take by sound-burst-vs-silence energy and cut each burst.
// Frames are short RMS windows; a burst opens above `OPEN` and closes below
// `CLOSE` (hysteresis) of the way from the noise floor to the loudest frame.
const ENERGY_FRAME_SECONDS: f64 = 0.02;
const ENERGY_OPEN_FRACTION: f64 = 0.22;
const ENERGY_CLOSE_FRACTION: f64 = 0.12;
// Bursts separated by a gap this short are merged into one clip, so the two
// quick sounds inside one "beep beep" stay together while the ~1s gap between
// repetitions still splits them into separate positives.
const ENERGY_MERGE_GAP_SECONDS: f64 = 0.35;
// A voiced run shorter than this is treated as noise, not a real sound burst.
const ENERGY_MIN_BURST_SECONDS: f64 = 0.08;
// Padding cut around each detected burst so the onset/tail is not clipped.
const ENERGY_LEAD_PADDING_SECONDS: f64 = 0.10;
const ENERGY_TAIL_PADDING_SECONDS: f64 = 0.18;

#[derive(Clone)]
struct AppState {
    data_root: Arc<PathBuf>,
    incoming_root: Arc<PathBuf>,
    settings_path: Arc<PathBuf>,
    settings: Arc<Mutex<ServerSettings>>,
    // Whisper server URL from the WHISPER_SERVER_URL environment variable. This
    // is the source of truth for where audio is transcribed; the app no longer
    // configures it.
    whisper_url: Arc<Option<String>>,
    // Wake-word scorer service URL from SCORER_SERVER_URL. Optional: only the
    // model-test scoring endpoint needs it; the rest of the server works without.
    scorer_url: Arc<Option<String>>,
    // Trained-model directory (MODELS_DIR), mounted read-only. Used only to
    // fingerprint the .onnx so cached score curves invalidate when a model is
    // retrained. May not exist; then the fingerprint is a sentinel.
    models_root: Arc<PathBuf>,
    // Absolute path of the repo on the *host*. Training launches a sibling
    // trainer container over the mounted docker socket, and its `-v` bind mount
    // source must be a host path, not a path inside this container. None until
    // HOST_REPO_ROOT is set; the training endpoints refuse to run without it.
    host_repo_root: Arc<Option<String>>,
    // Trainer image tag launched for a full training run (TRAINER_IMAGE).
    trainer_image: Arc<String>,
    // Pass `--gpus all` to the trainer container (TRAINER_USE_GPU != "0").
    trainer_gpu: Arc<bool>,
    db: Arc<Mutex<Connection>>,
    // Serializes training-queue dispatch so the enqueue-triggered kick and the
    // periodic scheduler never launch the same queued job twice.
    dispatch_lock: Arc<tokio::sync::Mutex<()>>,
    // In-flight F5 synthetic-positive generation jobs, keyed by slug, so the app
    // can kick one off and poll its progress without a second job being launched
    // for the same wake word.
    synth_jobs: Arc<Mutex<std::collections::HashMap<String, SynthJob>>>,
}

/// State of one F5 synthetic-positive generation run for a wake word.
#[derive(Debug, Clone, Serialize)]
struct SynthJob {
    running: bool,
    requested: usize,
    wrote: usize,
    error: Option<String>,
}

fn now_ms() -> i64 {
    Utc::now().timestamp_millis()
}

/// Parse an RFC 3339 / ISO 8601 timestamp (as written by train_job.sh) to epoch
/// milliseconds, or None if it does not parse.
fn rfc3339_ms(value: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|dt| dt.timestamp_millis())
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct ServerSettings {
    sync_server_url: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SettingsRequest {
    sync_server_url: Option<String>,
}

#[derive(Debug, Serialize)]
struct SettingsResponse {
    status: &'static str,
    settings: ServerSettings,
}

#[derive(Debug, Serialize)]
struct SyncResponse {
    status: &'static str,
    archive: String,
    wake_word_slug: String,
    clip_count: usize,
    imported_count: usize,
    validate_output: String,
    import_output: String,
    alignment_output: String,
    warnings: Vec<String>,
    whisper_server_url: Option<String>,
}

#[derive(Debug, Serialize)]
struct ProjectsResponse {
    status: &'static str,
    projects: Vec<ProjectSummary>,
}

/// A project a device is registering with the server so it propagates to the
/// user's other devices even before any recording exists for it. The device's
/// own project id travels as `external_id` so the round-tripped project keeps a
/// stable identity across devices.
#[derive(Debug, Deserialize)]
struct CreateProjectRequest {
    #[serde(alias = "external_id")]
    id: Option<String>,
    slug: String,
    phrase: String,
}

#[derive(Debug, Serialize)]
struct ProjectSummary {
    id: String,
    slug: String,
    phrase: String,
    created_at_millis: i64,
    bulk_slice_count: usize,
    positive_count: usize,
    negative_count: usize,
    background_count: usize,
    /// Negatives available from every *other* project, reusable for this one.
    pooled_negative_count: usize,
}

#[derive(Debug, Serialize)]
struct BulkReviewResponse {
    status: &'static str,
    wake_word_slug: String,
    clips: Vec<BulkReviewClip>,
}

#[derive(Debug, Serialize)]
struct BulkRecordingIdsResponse {
    status: &'static str,
    wake_word_slug: String,
    recording_ids: Vec<String>,
    /// Each recording id paired with its source-WAV SHA-256 (null for legacy
    /// rows), so a device can skip re-uploading takes the server already holds.
    checksums: Vec<RecordingChecksum>,
}

#[derive(Debug, Serialize)]
struct RecordingChecksum {
    id: String,
    sha256: Option<String>,
}

#[derive(Debug, Serialize)]
struct RecordingDetailItem {
    id: String,
    is_background: bool,
    is_test: bool,
    /// How the take was recorded/sliced: positive/negative/hard_negative/
    /// background/test, or `mixed` for legacy single-script takes. Lets the app
    /// group Review by recording kind.
    kind: String,
    recorded_at: String,
    duration_ms: i64,
    positive_count: i64,
    negative_count: i64,
    background_count: i64,
    device_manufacturer: Option<String>,
    device_model: Option<String>,
    app_version: Option<String>,
    input_route: Option<String>,
    session_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct RecordingDetailsResponse {
    status: &'static str,
    wake_word_slug: String,
    recordings: Vec<RecordingDetailItem>,
}

#[derive(Debug, Deserialize, Serialize)]
struct BulkAlignmentResponse {
    status: String,
    wake_word_slug: String,
    source_recording: String,
    script: String,
    words: Vec<WhisperWord>,
    cuts: Vec<BulkAlignmentCut>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct BulkAlignmentCut {
    id: String,
    label: String,
    start_sec: f64,
    end_sec: f64,
}

#[derive(Debug, Serialize)]
struct BulkReviewClip {
    id: String,
    label: String,
    spoken_phrase: String,
    source_recording: String,
    source_start_sec: f64,
    source_end_sec: f64,
    duration_ms: u64,
    average_probability: f64,
    word_count: usize,
    category: String,
    file_name: String,
}

#[derive(Debug, Serialize)]
struct DeleteReviewClipResponse {
    status: &'static str,
    deleted: bool,
}

#[derive(Debug, Serialize)]
struct ReprocessResponse {
    status: &'static str,
    wake_word_slug: String,
    recording_count: usize,
    positives: usize,
    negatives: usize,
    hard_negatives: usize,
    background: usize,
    dropped_positives: usize,
    alignment_output: String,
    warnings: Vec<String>,
    whisper_server_url: Option<String>,
}

/// Rolling wake-word detection over a stored bulk recording, with each true
/// target-phrase utterance (located from the current Whisper transcript) tagged
/// with the model's peak score in its window. The full curve is returned so a
/// client can re-threshold the detection/false-positive counts for free.
#[derive(Debug, Serialize)]
struct ScoreResponse {
    status: &'static str,
    wake_word_slug: String,
    source_recording: String,
    phrase: String,
    /// The archived run id scored, or null when the current model was used.
    run: Option<String>,
    mode: String,
    window_ms: u64,
    step_ms: u64,
    keep_ms: u64,
    duration_ms: f64,
    threshold: f64,
    times_ms: Vec<f64>,
    scores: Vec<f64>,
    targets: Vec<ScoreTarget>,
    /// Counts at `threshold`: targets that peaked at/above it, targets that did
    /// not, and above-threshold detections landing outside any target window.
    true_positives: usize,
    false_negatives: usize,
    false_positives: usize,
}

/// One occurrence of the trigger phrase in the transcript, with the model's
/// best score anywhere in the detection window aligned to the phrase tail.
#[derive(Debug, Serialize)]
struct ScoreTarget {
    text: String,
    start_ms: i64,
    end_ms: i64,
    peak_score: f64,
    peak_time_ms: f64,
    detected: bool,
}

/// One test take's stored grade against a model, as returned to the app. Mirrors
/// `db::ScoreGrade` with the recording id kept so the app can match it to a card.
#[derive(Debug, Serialize)]
struct ScoreGradeItem {
    recording_id: String,
    run: Option<String>,
    threshold: f64,
    peak_score: f64,
    max_score: f64,
    has_target: bool,
    target_count: i64,
    true_positives: i64,
    false_negatives: i64,
    false_positives: i64,
    detected: bool,
    scored_at_ms: i64,
}

impl From<db::ScoreGrade> for ScoreGradeItem {
    fn from(g: db::ScoreGrade) -> Self {
        Self {
            recording_id: g.recording_id,
            run: g.run_id,
            threshold: g.threshold,
            peak_score: g.peak_score,
            max_score: g.max_score,
            has_target: g.has_target,
            target_count: g.target_count,
            true_positives: g.true_positives,
            false_negatives: g.false_negatives,
            false_positives: g.false_positives,
            detected: g.detected,
            scored_at_ms: g.created_at_ms,
        }
    }
}

/// Model-wide totals across all graded test takes: how many takes exist, how
/// many are graded against this model, and the summed target/miss/false-positive
/// counts that headline the Model test view.
#[derive(Debug, Serialize)]
struct GradeTotals {
    test_takes: usize,
    graded: usize,
    targets: i64,
    true_positives: i64,
    false_negatives: i64,
    false_positives: i64,
    detections: usize,
}

/// The Model test view's grade payload: per-take grades plus the model totals,
/// for one model (`run` null = current) in one detection mode.
#[derive(Debug, Serialize)]
struct ModelGradesResponse {
    status: &'static str,
    wake_word_slug: String,
    run: Option<String>,
    model_fp: String,
    mode: String,
    totals: GradeTotals,
    grades: Vec<ScoreGradeItem>,
}

/// The scorer service's `/score` reply: parallel time/score arrays. The scorer
/// emits fractional milliseconds, so times are floats.
#[derive(Debug, Deserialize)]
struct ScorerCurve {
    #[serde(default)]
    duration_ms: f64,
    #[serde(default)]
    times_ms: Vec<f64>,
    #[serde(default)]
    scores: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct DeleteRecordingResponse {
    status: &'static str,
    deleted: bool,
    removed_files: usize,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    schema_version: u64,
    wake_word: WakeWord,
    clips: Vec<Clip>,
    #[serde(default)]
    bulk_recordings: Vec<BulkRecording>,
    #[serde(default)]
    background_recordings: Vec<BackgroundRecording>,
    /// Test-only takes. Transcribed for word timings so the model can be scored
    /// against them, but never sliced into training data. Recording ids carry the
    /// `test_` prefix so every downstream path can keep them out of the pool.
    #[serde(default)]
    test_recordings: Vec<BulkRecording>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct WakeWord {
    slug: String,
    #[serde(flatten)]
    extra: Value,
}

impl WakeWord {
    fn phrase(&self) -> String {
        self.extra
            .get("phrase")
            .and_then(Value::as_str)
            .unwrap_or(&self.slug)
            .to_string()
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Clip {
    id: String,
    file: String,
    label: String,
    #[serde(default)]
    spoken_phrase: String,
    #[serde(flatten)]
    extra: Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct BulkRecording {
    id: String,
    file: String,
    script: String,
    /// How to slice this take: `positive`, `negative`, `hard_negative`, or (for
    /// takes from older app builds) empty, which the server treats as `mixed`.
    #[serde(default)]
    kind: String,
    #[serde(default)]
    recorded_at: String,
    #[serde(default)]
    duration_ms: u64,
    #[serde(flatten)]
    extra: Value,
}

/// A long ambient/background noise take. Unlike a bulk recording it is not
/// transcribed; the server slices it into fixed-length background clips.
#[derive(Debug, Clone, Deserialize, Serialize)]
struct BackgroundRecording {
    id: String,
    file: String,
    #[serde(default)]
    recorded_at: String,
    #[serde(default)]
    duration_ms: u64,
    #[serde(flatten)]
    extra: Value,
}

#[derive(Debug, Deserialize)]
struct WhisperResponse {
    #[serde(default)]
    text: String,
    #[serde(default)]
    segments: Vec<WhisperSegment>,
}

#[derive(Debug, Deserialize)]
struct WhisperSegment {
    #[serde(default)]
    start: f64,
    #[serde(default)]
    end: f64,
    #[serde(default)]
    words: Vec<WhisperWord>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct WhisperWord {
    word: String,
    start: f64,
    end: f64,
    #[serde(default)]
    probability: f64,
}

#[derive(Debug, Default)]
struct AlignmentSummary {
    recordings: usize,
    positives: usize,
    negatives: usize,
    hard_negatives: usize,
    background: usize,
    dropped_positives: usize,
    warnings: Vec<String>,
}

impl AlignmentSummary {
    /// Fold another summary's counts and warnings into this one. Used to combine
    /// the bulk-alignment and background-slicing passes over a single upload.
    fn absorb(&mut self, other: AlignmentSummary) {
        self.recordings += other.recordings;
        self.positives += other.positives;
        self.negatives += other.negatives;
        self.hard_negatives += other.hard_negatives;
        self.background += other.background;
        self.dropped_positives += other.dropped_positives;
        self.warnings.extend(other.warnings);
    }
}

#[tokio::main]
async fn main() {
    let bind_addr = env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:8765".to_string());
    let data_root = env::var("DATA_ROOT").unwrap_or_else(|_| "/data/real".to_string());
    let incoming_root =
        env::var("INCOMING_ROOT").unwrap_or_else(|_| "/incoming/bundles".to_string());
    let settings_path =
        env::var("SETTINGS_PATH").unwrap_or_else(|_| "/data/server_settings.json".to_string());
    let settings_path = PathBuf::from(settings_path);
    let db_path = env::var("DB_PATH").unwrap_or_else(|_| "/data/trainer.db".to_string());
    let db = db::open(&PathBuf::from(&db_path)).expect("open database");
    println!("database at {db_path}");

    let whisper_url = env::var("WHISPER_SERVER_URL")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    match &whisper_url {
        Some(url) => println!("whisper server at {url}"),
        None => println!("WHISPER_SERVER_URL not set; transcription will fail until configured"),
    }

    let scorer_url = env::var("SCORER_SERVER_URL")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    match &scorer_url {
        Some(url) => println!("scorer server at {url}"),
        None => println!("SCORER_SERVER_URL not set; model-test scoring disabled"),
    }

    let models_root = env::var("MODELS_DIR")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "/output".to_string());

    let host_repo_root = env::var("HOST_REPO_ROOT")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    match &host_repo_root {
        Some(root) => println!("host repo root {root}; training enabled"),
        None => println!("HOST_REPO_ROOT not set; training endpoints disabled"),
    }
    let trainer_image = env::var("TRAINER_IMAGE")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "livekit-wakeword-trainer:latest".to_string());
    let trainer_gpu = env::var("TRAINER_USE_GPU")
        .map(|value| value.trim() != "0")
        .unwrap_or(true);

    let state = AppState {
        data_root: Arc::new(PathBuf::from(data_root)),
        incoming_root: Arc::new(PathBuf::from(incoming_root)),
        settings: Arc::new(Mutex::new(load_settings(&settings_path))),
        settings_path: Arc::new(settings_path),
        whisper_url: Arc::new(whisper_url),
        scorer_url: Arc::new(scorer_url),
        models_root: Arc::new(PathBuf::from(models_root)),
        host_repo_root: Arc::new(host_repo_root),
        trainer_image: Arc::new(trainer_image),
        trainer_gpu: Arc::new(trainer_gpu),
        db: Arc::new(Mutex::new(db)),
        dispatch_lock: Arc::new(tokio::sync::Mutex::new(())),
        synth_jobs: Arc::new(Mutex::new(std::collections::HashMap::new())),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/settings", get(get_settings).post(update_settings))
        .route("/projects", get(projects).post(create_project))
        .route("/sync", post(sync))
        .route("/reprocess/:slug", post(reprocess_project))
        .route("/reprocess/:slug/:recording_id", post(reprocess_recording))
        .route("/bulk/:slug/recordings", get(bulk_recording_ids))
        .route("/bulk/:slug/recordings/detail", get(bulk_recording_details))
        .route(
            "/bulk/:slug/:recording_id",
            axum::routing::delete(delete_recording),
        )
        .route("/review/:slug/bulk", get(bulk_review))
        .route(
            "/review/:slug/bulk/:recording_id/alignment",
            get(bulk_alignment),
        )
        .route(
            "/review/:slug/bulk/:recording_id/audio",
            get(bulk_source_audio),
        )
        .route("/score/:slug/:recording_id", get(score_recording))
        .route("/score-all/:slug", post(score_all_test_takes))
        .route("/score-grades/:slug", get(model_test_grades))
        .route("/models", get(list_models))
        .route("/models/runs", get(list_model_runs))
        .route("/train/:slug", post(start_training))
        .route("/train/:slug/status", get(training_status))
        .route("/train/:slug/log", get(training_log))
        .route("/train/:slug/cancel", post(cancel_training))
        .route("/queue", get(training_queue))
        .route("/queue/:id", axum::routing::delete(delete_queue_entry))
        .route(
            "/review/:slug/:category/:file_name",
            get(review_audio).delete(delete_review_clip),
        )
        .route(
            "/synth/:slug",
            axum::routing::delete(delete_synth),
        )
        .route("/synth/:slug/sample", get(synthetic_samples))
        .route("/synth/:slug/audio/:file_name", get(synthetic_audio))
        .route("/synth/:slug/generate", post(generate_synth))
        .route("/synth/:slug/generate/status", get(generate_synth_status))
        .layer(DefaultBodyLimit::max(512 * 1024 * 1024))
        .with_state(state.clone());

    // Drive the training queue: reconcile finished runs and dispatch the next
    // queued job on a timer, independent of any request.
    if state.host_repo_root.is_some() {
        tokio::spawn(training_scheduler(state.clone()));
    }

    let addr: SocketAddr = bind_addr.parse().expect("invalid BIND_ADDR");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("bind failed");
    println!("listening on http://{addr}");
    axum::serve(listener, app).await.expect("server failed");
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn get_settings(State(state): State<AppState>) -> Json<SettingsResponse> {
    let settings = state
        .settings
        .lock()
        .expect("settings lock poisoned")
        .clone();
    Json(SettingsResponse {
        status: "ok",
        settings,
    })
}

async fn update_settings(
    State(state): State<AppState>,
    Json(request): Json<SettingsRequest>,
) -> Result<Json<SettingsResponse>, AppError> {
    let settings = ServerSettings {
        sync_server_url: clean_optional_url(request.sync_server_url),
    };
    save_settings(&state.settings_path, &settings)?;
    *state.settings.lock().expect("settings lock poisoned") = settings.clone();
    Ok(Json(SettingsResponse {
        status: "saved",
        settings,
    }))
}

async fn projects(State(state): State<AppState>) -> Result<Json<ProjectsResponse>, AppError> {
    let rows = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::project_summaries(&conn).map_err(db_error)?
    };
    // Cross-wake-word reuse: every other project's plain negatives, plus their
    // positives (reused as hard negatives), are available to this project. Hard
    // negatives are project-scoped, so they never enter the shared pool.
    let total_poolable_negatives: i64 = rows.iter().map(|r| r.poolable_negative_count).sum();
    let total_positives: i64 = rows.iter().map(|r| r.positive_count).sum();
    let projects = rows
        .into_iter()
        .map(|row| {
            let pooled = (total_poolable_negatives - row.poolable_negative_count)
                + (total_positives - row.positive_count);
            ProjectSummary {
                id: row.external_id,
                slug: row.slug,
                phrase: row.phrase,
                created_at_millis: row.created_at_ms,
                bulk_slice_count: row.bulk_slice_count as usize,
                positive_count: row.positive_count as usize,
                negative_count: row.negative_count as usize,
                background_count: row.background_count as usize,
                pooled_negative_count: pooled.max(0) as usize,
            }
        })
        .collect();
    Ok(Json(ProjectsResponse {
        status: "ok",
        projects,
    }))
}

/// Register a project on the server so it shows up on the user's other devices
/// without waiting for a recording to be synced. Idempotent on slug: creating
/// the same wake word again just refreshes its phrase and id.
async fn create_project(
    State(state): State<AppState>,
    Json(request): Json<CreateProjectRequest>,
) -> Result<Json<Value>, AppError> {
    let slug = request.slug.trim().to_string();
    let phrase = request.phrase.trim().to_string();
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request(format!("unsafe slug: {slug:?}")));
    }
    if phrase.is_empty() {
        return Err(AppError::bad_request("phrase is required"));
    }
    let external_id = request
        .id
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    {
        let conn = state.db.lock().expect("db lock poisoned");
        db::upsert_project(&conn, &slug, &phrase, external_id, now_ms()).map_err(db_error)?;
    }
    Ok(Json(json!({ "status": "ok", "slug": slug })))
}

async fn sync(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Json<SyncResponse>, AppError> {
    if body.is_empty() {
        return Err(AppError::bad_request("empty upload"));
    }
    let whisper_server_url = resolve_whisper_url(&state, &headers);

    fs::create_dir_all(&*state.incoming_root)?;
    let archive = state
        .incoming_root
        .join(format!("bundle_{}.zip", Utc::now().timestamp_millis()));
    fs::write(&archive, &body)?;

    let extract_root = env::temp_dir().join(format!(
        "livekit_trainer_bundle_{}",
        Utc::now().timestamp_millis()
    ));
    fs::create_dir_all(&extract_root)?;

    let result = async {
        extract_zip(&body, &extract_root)?;
        let manifest = read_manifest(&extract_root)?;
        validate_manifest(&manifest)?;
        let warnings = validate_wavs(&extract_root, &manifest)?;
        let imported_count =
            import_bundle(&extract_root, &manifest, &state.data_root, &state.db)?;
        let mut alignment = align_bulk_recordings(
            &extract_root,
            &manifest,
            &state.data_root,
            &state.db,
            whisper_server_url.as_deref(),
        )
        .await?;
        let background =
            align_background_recordings(&extract_root, &manifest, &state.data_root, &state.db)
                .await?;
        alignment.absorb(background);
        let test = align_test_recordings(
            &extract_root,
            &manifest,
            &state.data_root,
            &state.db,
            whisper_server_url.as_deref(),
        )
        .await?;
        alignment.absorb(test);
        let archive_name = archive
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("bundle.zip");
        let import_output = format!(
            "Imported {imported_count} clips into data/real/{}",
            manifest.wake_word.slug
        );
        let validate_output = validation_summary(&warnings, manifest.clips.len());
        Ok(SyncResponse {
            status: "imported",
            archive: format!("incoming/bundles/{archive_name}"),
            wake_word_slug: manifest.wake_word.slug,
            clip_count: manifest.clips.len(),
            imported_count,
            validate_output,
            import_output,
            alignment_output: alignment_summary(&alignment),
            warnings,
            whisper_server_url,
        })
    }
    .await;

    let _ = fs::remove_dir_all(&extract_root);
    result.map(Json)
}

/// Resolve the Whisper server URL from the request header, falling back to the
/// WHISPER_SERVER_URL environment variable captured at startup. Shared by the
/// upload and reprocess paths.
fn resolve_whisper_url(state: &AppState, headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-whisper-server-url")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| state.whisper_url.as_ref().clone())
}

async fn reprocess_project(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(slug): AxumPath<String>,
) -> Result<Json<ReprocessResponse>, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request(format!(
            "unsafe wake word slug: {slug}"
        )));
    }
    let recordings = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::recordings_for_reprocess(&conn, &slug).map_err(db_error)?
    };
    run_reprocess(&state, &headers, &slug, recordings).await
}

async fn reprocess_recording(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath((slug, recording_id)): AxumPath<(String, String)>,
) -> Result<Json<ReprocessResponse>, AppError> {
    if !is_safe_slug(&slug) || !is_safe_recording_id(&recording_id) {
        return Err(AppError::bad_request("unsafe reprocess path"));
    }
    let recording = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::recording_meta(&conn, &recording_id).map_err(db_error)?
    };
    let recording = recording
        .filter(|meta| meta.project_slug == slug)
        .ok_or_else(|| AppError::bad_request(format!("unknown recording: {recording_id}")))?;
    run_reprocess(&state, &headers, &slug, vec![recording]).await
}

/// Re-run alignment for a set of stored recordings from their already-saved
/// source WAVs, without a fresh upload. Cuts identically to the sync path.
async fn run_reprocess(
    state: &AppState,
    headers: &HeaderMap,
    slug: &str,
    recordings: Vec<db::RecordingMeta>,
) -> Result<Json<ReprocessResponse>, AppError> {
    let whisper_server_url = resolve_whisper_url(state, headers);
    let (phrase, external_id) = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::project_phrase(&conn, slug).map_err(db_error)?
    }
    .ok_or_else(|| AppError::bad_request(format!("unknown project: {slug}")))?;

    let dest_root = state.data_root.join(slug);
    let mut summary = AlignmentSummary::default();

    let whisper = whisper_server_url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    for meta in &recordings {
        let source = dest_root
            .join("bulk_source")
            .join(format!("{}.wav", safe_filename(&meta.id)));
        if !source.is_file() {
            summary.recordings += 1;
            summary.warnings.push(format!(
                "{}: stored source WAV missing; re-sync this recording",
                meta.id
            ));
            continue;
        }
        // Background takes re-chunk deterministically and need no Whisper; only
        // scripted bulk reads require an alignment server.
        let whisper_url = if meta.script == BACKGROUND_SCRIPT_MARKER {
            ""
        } else {
            match whisper {
                Some(url) => url,
                None => {
                    summary.recordings += 1;
                    summary
                        .warnings
                        .push(format!("{}: no Whisper server URL configured", meta.id));
                    continue;
                }
            }
        };
        align_one_recording(
            &meta.id,
            &meta.script,
            &meta.kind,
            &meta.recorded_at,
            meta.duration_ms.max(0) as u64,
            &source,
            slug,
            &phrase,
            external_id.as_deref(),
            &dest_root,
            &state.db,
            whisper_url,
            // Reprocess has no manifest; the upsert keeps existing provenance.
            &db::CaptureMeta::default(),
            &mut summary,
        )
        .await?;
    }

    Ok(Json(ReprocessResponse {
        status: "reprocessed",
        wake_word_slug: slug.to_string(),
        recording_count: recordings.len(),
        positives: summary.positives,
        negatives: summary.negatives,
        hard_negatives: summary.hard_negatives,
        background: summary.background,
        dropped_positives: summary.dropped_positives,
        alignment_output: alignment_summary(&summary),
        warnings: summary.warnings.clone(),
        whisper_server_url,
    }))
}

async fn delete_recording(
    State(state): State<AppState>,
    AxumPath((slug, recording_id)): AxumPath<(String, String)>,
) -> Result<Json<DeleteRecordingResponse>, AppError> {
    if !is_safe_slug(&slug) || !is_safe_recording_id(&recording_id) {
        return Err(AppError::bad_request("unsafe delete path"));
    }
    let paths = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::active_slice_paths(&conn, &recording_id).map_err(db_error)?
    };
    let mut removed_files = 0usize;
    for path in paths {
        let path = PathBuf::from(path);
        if path.is_file() && fs::remove_file(&path).is_ok() {
            removed_files += 1;
        }
    }
    let source = state
        .data_root
        .join(&slug)
        .join("bulk_source")
        .join(format!("{}.wav", safe_filename(&recording_id)));
    if source.is_file() && fs::remove_file(&source).is_ok() {
        removed_files += 1;
    }
    let deleted = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::delete_recording(&conn, &recording_id).map_err(db_error)?
    };
    Ok(Json(DeleteRecordingResponse {
        status: "deleted",
        deleted,
        removed_files,
    }))
}

async fn bulk_review(
    State(state): State<AppState>,
    AxumPath(slug): AxumPath<String>,
) -> Result<Json<BulkReviewResponse>, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request(format!(
            "unsafe wake word slug: {slug}"
        )));
    }
    let rows = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::review_slices(&conn, &slug).map_err(db_error)?
    };
    let clips = rows
        .into_iter()
        .map(|row| BulkReviewClip {
            id: row.id,
            label: row.label,
            spoken_phrase: row.spoken_phrase,
            source_recording: row.source_recording,
            source_start_sec: row.source_start_ms as f64 / 1000.0,
            source_end_sec: row.source_end_ms as f64 / 1000.0,
            duration_ms: row.duration_ms.max(0) as u64,
            average_probability: row.avg_probability,
            word_count: row.word_count.max(0) as usize,
            category: row.category,
            file_name: row.file_name,
        })
        .collect();
    Ok(Json(BulkReviewResponse {
        status: "ok",
        wake_word_slug: slug,
        clips,
    }))
}

async fn bulk_recording_ids(
    State(state): State<AppState>,
    AxumPath(slug): AxumPath<String>,
) -> Result<Json<BulkRecordingIdsResponse>, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request(format!(
            "unsafe wake word slug: {slug}"
        )));
    }
    let checksums = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::recording_checksums(&conn, &slug).map_err(db_error)?
    };
    let recording_ids = checksums.iter().map(|(id, _)| id.clone()).collect();
    let checksums = checksums
        .into_iter()
        .map(|(id, sha256)| RecordingChecksum { id, sha256 })
        .collect();
    Ok(Json(BulkRecordingIdsResponse {
        status: "ok",
        wake_word_slug: slug.clone(),
        recording_ids,
        checksums,
    }))
}

async fn bulk_recording_details(
    State(state): State<AppState>,
    AxumPath(slug): AxumPath<String>,
) -> Result<Json<RecordingDetailsResponse>, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request(format!(
            "unsafe wake word slug: {slug}"
        )));
    }
    let details = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::recording_details(&conn, &slug).map_err(db_error)?
    };
    let recordings = details
        .into_iter()
        .map(|d| RecordingDetailItem {
            is_background: d.id.starts_with("background_"),
            is_test: d.id.starts_with("test_"),
            kind: d.kind,
            id: d.id,
            recorded_at: d.recorded_at,
            duration_ms: d.duration_ms,
            positive_count: d.positive_count,
            negative_count: d.negative_count,
            background_count: d.background_count,
            device_manufacturer: d.device_manufacturer,
            device_model: d.device_model,
            app_version: d.app_version,
            input_route: d.input_route,
            session_id: d.session_id,
        })
        .collect();
    Ok(Json(RecordingDetailsResponse {
        status: "ok",
        wake_word_slug: slug,
        recordings,
    }))
}

async fn bulk_alignment(
    State(state): State<AppState>,
    AxumPath((slug, recording_id)): AxumPath<(String, String)>,
) -> Result<Json<BulkAlignmentResponse>, AppError> {
    if !is_safe_slug(&slug) || !is_safe_recording_id(&recording_id) {
        return Err(AppError::bad_request("unsafe bulk alignment path"));
    }
    let (script, words, cuts) = {
        let conn = state.db.lock().expect("db lock poisoned");
        let script = db::recording_script(&conn, &recording_id)
            .map_err(db_error)?
            .ok_or_else(|| AppError::bad_request(format!("unknown recording: {recording_id}")))?;
        let words = db::current_transcript_words(&conn, &recording_id).map_err(db_error)?;
        let cuts = db::recording_cuts(&conn, &recording_id).map_err(db_error)?;
        (script, words, cuts)
    };
    let response = BulkAlignmentResponse {
        status: "ok".to_string(),
        wake_word_slug: slug,
        source_recording: recording_id,
        script,
        words: words
            .into_iter()
            .map(|word| WhisperWord {
                word: word.word,
                start: word.start_ms as f64 / 1000.0,
                end: word.end_ms as f64 / 1000.0,
                probability: word.probability,
            })
            .collect(),
        cuts: cuts
            .into_iter()
            .map(|cut| BulkAlignmentCut {
                id: cut.id,
                label: cut.label,
                start_sec: cut.start_ms as f64 / 1000.0,
                end_sec: cut.end_ms as f64 / 1000.0,
            })
            .collect(),
    };
    Ok(Json(response))
}

async fn bulk_source_audio(
    State(state): State<AppState>,
    AxumPath((slug, recording_id)): AxumPath<(String, String)>,
) -> Result<impl IntoResponse, AppError> {
    if !is_safe_slug(&slug) || !is_safe_recording_id(&recording_id) {
        return Err(AppError::bad_request("unsafe bulk audio path"));
    }
    let path = state
        .data_root
        .join(&slug)
        .join("bulk_source")
        .join(format!("{}.wav", safe_filename(&recording_id)));
    let bytes = fs::read(path)?;
    Ok(([(header::CONTENT_TYPE, "audio/wav")], bytes))
}

/// Resolve the scorer service URL from the request header, falling back to the
/// SCORER_SERVER_URL environment variable captured at startup.
fn resolve_scorer_url(state: &AppState, headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-scorer-server-url")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| state.scorer_url.as_ref().clone())
}

/// Score a stored bulk recording against a trained model and overlay the
/// current Whisper transcript so each true target utterance is scored honestly.
/// This is the model-test diagnostic: synthetic eval recall hides the streaming
/// gap, so we replay real takes and check whether the phrase actually fires
/// mid-sentence. Query params: `mode` (full|reset), `step_ms`, `keep_ms`,
/// `threshold`.
async fn score_recording(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath((slug, recording_id)): AxumPath<(String, String)>,
    RawQuery(query): RawQuery,
) -> Result<Json<ScoreResponse>, AppError> {
    if !is_safe_slug(&slug) || !is_safe_recording_id(&recording_id) {
        return Err(AppError::bad_request("unsafe score path"));
    }
    let params = parse_query(query.as_deref());
    let mode = match params.get("mode").map(String::as_str) {
        Some("full") | None => "full",
        Some("reset") => "reset",
        Some(other) => return Err(AppError::bad_request(format!("unknown mode: {other}"))),
    };
    let step_ms = params
        .get("step_ms")
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v >= 1)
        .unwrap_or(if mode == "reset" { 40 } else { 10 });
    let keep_ms = params
        .get("keep_ms")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(700);
    let threshold = params
        .get("threshold")
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .unwrap_or(0.5);
    // Optional archived run id: score a specific past model version instead of
    // the mutable current model, so retraining never overwrites a run's scores.
    let run = match params.get("run").map(String::as_str) {
        Some(r) if is_safe_run_id(r) => Some(r.to_string()),
        Some(_) => return Err(AppError::bad_request("unsafe run id")),
        None => None,
    };

    let scorer_url = resolve_scorer_url(&state, &headers).ok_or_else(|| {
        AppError::bad_request("scorer not configured; set SCORER_SERVER_URL or x-scorer-server-url")
    })?;
    let force = params.get("nocache").map(String::as_str) == Some("1");

    let response = compute_score(
        &state,
        &scorer_url,
        &slug,
        &recording_id,
        mode,
        step_ms,
        keep_ms,
        threshold,
        run.as_deref(),
        force,
    )
    .await?;
    Ok(Json(response))
}

/// Score one stored recording against a model, cache its curve, persist the
/// derived grade (so the take's score on this model survives without re-running
/// the scorer), and return the full response. Shared by the single-recording
/// endpoint and the bulk "score all test takes" endpoint. `force` bypasses the
/// cached curve (the `nocache=1` path).
#[allow(clippy::too_many_arguments)]
async fn compute_score(
    state: &AppState,
    scorer_url: &str,
    slug: &str,
    recording_id: &str,
    mode: &str,
    step_ms: u64,
    keep_ms: u64,
    threshold: f64,
    run: Option<&str>,
    force: bool,
) -> Result<ScoreResponse, AppError> {
    let (phrase, words) = {
        let conn = state.db.lock().expect("db lock poisoned");
        let phrase = db::project_phrase(&conn, slug)
            .map_err(db_error)?
            .map(|(phrase, _external_id)| phrase)
            .unwrap_or_else(|| slug.replace('_', " "));
        let words = db::current_transcript_words(&conn, recording_id).map_err(db_error)?;
        (phrase, words)
    };

    // Fingerprint the trained model so a cached curve is reused only while the
    // model that produced it is unchanged; retraining flips the fingerprint and
    // the next request re-scores. An archived run is immutable, so its id alone
    // is a stable fingerprint and keeps each run's cached curves distinct from
    // the current model's and from each other.
    let model_fp = match run {
        Some(r) => format!("run-{r}"),
        None => model_fingerprint(&state.models_root, slug),
    };

    let cached = if force {
        None
    } else {
        let conn = state.db.lock().expect("db lock poisoned");
        db::cached_score_curve(&conn, recording_id, mode, step_ms, keep_ms, &model_fp)
            .map_err(db_error)?
    };

    let curve = if let Some((duration_ms, times_ms, scores)) = cached {
        ScorerCurve { duration_ms, times_ms, scores }
    } else {
        let wav_path = state
            .data_root
            .join(slug)
            .join("bulk_source")
            .join(format!("{}.wav", safe_filename(recording_id)));
        let wav = fs::read(&wav_path)
            .map_err(|_| AppError::bad_request(format!("no source audio for {recording_id}")))?;

        let curve = run_scorer(scorer_url, wav, slug, run, mode, step_ms, keep_ms).await?;
        let conn = state.db.lock().expect("db lock poisoned");
        db::store_score_curve(
            &conn,
            recording_id,
            mode,
            step_ms,
            keep_ms,
            &model_fp,
            curve.duration_ms,
            &curve.times_ms,
            &curve.scores,
            now_ms(),
        )
        .map_err(db_error)?;
        curve
    };

    let targets = locate_targets(&phrase, &words, &curve, threshold);
    let true_positives = targets.iter().filter(|t| t.detected).count();
    let false_negatives = targets.len() - true_positives;
    let false_positives = count_false_positives(&curve, &targets, threshold);

    // A representative single score for the take: the peak over located target
    // phrases, or the whole-curve maximum when the take has no target phrase (so
    // a negative take still shows how hot the model got). Persist the grade.
    let max_score = curve.scores.iter().cloned().fold(0.0_f64, f64::max);
    let has_target = !targets.is_empty();
    let target_peak = targets.iter().map(|t| t.peak_score).fold(0.0_f64, f64::max);
    let peak_score = if has_target { target_peak } else { max_score };
    {
        let conn = state.db.lock().expect("db lock poisoned");
        db::store_score_grade(
            &conn,
            &model_fp,
            mode,
            &db::ScoreGrade {
                recording_id: recording_id.to_string(),
                run_id: run.map(str::to_string),
                threshold,
                peak_score,
                max_score,
                has_target,
                target_count: targets.len() as i64,
                true_positives: true_positives as i64,
                false_negatives: false_negatives as i64,
                false_positives: false_positives as i64,
                detected: peak_score >= threshold,
                created_at_ms: now_ms(),
            },
        )
        .map_err(db_error)?;
    }

    Ok(ScoreResponse {
        status: "ok",
        wake_word_slug: slug.to_string(),
        source_recording: recording_id.to_string(),
        phrase,
        run: run.map(str::to_string),
        mode: mode.to_string(),
        window_ms: 2000,
        step_ms,
        keep_ms,
        duration_ms: curve.duration_ms,
        threshold,
        times_ms: curve.times_ms,
        scores: curve.scores,
        targets,
        true_positives,
        false_negatives,
        false_positives,
    })
}

/// Query params shared by the score endpoints: detection mode and its scan
/// params plus the threshold and optional archived run id. Parsed once so the
/// single and bulk endpoints agree on defaults.
struct ScoreParams {
    mode: &'static str,
    step_ms: u64,
    keep_ms: u64,
    threshold: f64,
    run: Option<String>,
}

fn parse_score_params(query: Option<&str>) -> Result<ScoreParams, AppError> {
    let params = parse_query(query);
    let mode = match params.get("mode").map(String::as_str) {
        Some("full") | None => "full",
        Some("reset") => "reset",
        Some(other) => return Err(AppError::bad_request(format!("unknown mode: {other}"))),
    };
    let step_ms = params
        .get("step_ms")
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v >= 1)
        .unwrap_or(if mode == "reset" { 40 } else { 10 });
    let keep_ms = params
        .get("keep_ms")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(700);
    let threshold = params
        .get("threshold")
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .unwrap_or(0.5);
    let run = match params.get("run").map(String::as_str) {
        Some(r) if is_safe_run_id(r) => Some(r.to_string()),
        Some(_) => return Err(AppError::bad_request("unsafe run id")),
        None => None,
    };
    Ok(ScoreParams { mode, step_ms, keep_ms, threshold, run })
}

/// Score every test take for a wake word against one model in a single request,
/// so the Model test view can grade the whole set without the app firing a
/// score per take. Runs the takes one at a time (the scorer is a single-GPU
/// service) and returns the resulting grades plus the model totals.
async fn score_all_test_takes(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(slug): AxumPath<String>,
    RawQuery(query): RawQuery,
) -> Result<Json<ModelGradesResponse>, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request("unsafe score path"));
    }
    let sp = parse_score_params(query.as_deref())?;
    let scorer_url = resolve_scorer_url(&state, &headers).ok_or_else(|| {
        AppError::bad_request("scorer not configured; set SCORER_SERVER_URL or x-scorer-server-url")
    })?;
    let ids = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::test_take_ids(&conn, &slug).map_err(db_error)?
    };
    for id in &ids {
        // A take with no source audio or transcript would fail; skip it so one
        // bad take can't sink the whole batch.
        if let Err(error) = compute_score(
            &state,
            &scorer_url,
            &slug,
            id,
            sp.mode,
            sp.step_ms,
            sp.keep_ms,
            sp.threshold,
            sp.run.as_deref(),
            false,
        )
        .await
        {
            eprintln!("score_all: skipping {id}: {}", error.message);
        }
    }
    grades_response(&state, &slug, sp.mode, sp.run.as_deref())
}

/// The stored grades for a model's test takes plus the model totals, read back
/// without re-running the scorer. Backs the Model test view's per-take cards and
/// its misses / false-positive statistics.
async fn model_test_grades(
    State(state): State<AppState>,
    AxumPath(slug): AxumPath<String>,
    RawQuery(query): RawQuery,
) -> Result<Json<ModelGradesResponse>, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request("unsafe score path"));
    }
    let sp = parse_score_params(query.as_deref())?;
    grades_response(&state, &slug, sp.mode, sp.run.as_deref())
}

/// Assemble the grades response: every stored grade for this model plus totals.
fn grades_response(
    state: &AppState,
    slug: &str,
    mode: &str,
    run: Option<&str>,
) -> Result<Json<ModelGradesResponse>, AppError> {
    let model_fp = match run {
        Some(r) => format!("run-{r}"),
        None => model_fingerprint(&state.models_root, slug),
    };
    let (test_takes, grades) = {
        let conn = state.db.lock().expect("db lock poisoned");
        let ids = db::test_take_ids(&conn, slug).map_err(db_error)?;
        let grades = db::grades_for_model(&conn, slug, &model_fp, mode).map_err(db_error)?;
        (ids.len(), grades)
    };
    let grades: Vec<ScoreGradeItem> = grades.into_iter().map(ScoreGradeItem::from).collect();
    let totals = GradeTotals {
        test_takes,
        graded: grades.len(),
        targets: grades.iter().map(|g| g.target_count).sum(),
        true_positives: grades.iter().map(|g| g.true_positives).sum(),
        false_negatives: grades.iter().map(|g| g.false_negatives).sum(),
        false_positives: grades.iter().map(|g| g.false_positives).sum(),
        detections: grades.iter().filter(|g| g.detected).count(),
    };
    Ok(Json(ModelGradesResponse {
        status: "ok",
        wake_word_slug: slug.to_string(),
        run: run.map(str::to_string),
        model_fp,
        mode: mode.to_string(),
        totals,
        grades,
    }))
}

/// Minimal `a=b&c=d` query parser; avoids pulling in axum's `query` feature.
fn parse_query(raw: Option<&str>) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    let Some(raw) = raw else { return map };
    for pair in raw.split('&').filter(|p| !p.is_empty()) {
        let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
        map.insert(key.to_string(), value.to_string());
    }
    map
}

/// Fingerprint the trained `.onnx` for a wake word so cached score curves stay
/// valid only while that exact model file is in place. Uses size + modified
/// time — cheap and flips on any retrain/export. Missing model or unreadable
/// metadata yields a stable sentinel so caching still works but never masks a
/// swapped model (a later present model changes the fingerprint).
fn model_fingerprint(models_root: &Path, slug: &str) -> String {
    let path = models_root.join(slug).join(format!("{slug}.onnx"));
    let Ok(meta) = fs::metadata(&path) else {
        return "nomodel".to_string();
    };
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{}-{}", meta.len(), mtime)
}

/// A run id is a compact UTC timestamp like `20260719T153000Z`; restrict it to
/// ASCII alphanumerics so it can never escape the run directory.
fn is_safe_run_id(run_id: &str) -> bool {
    !run_id.is_empty() && run_id.chars().all(|ch| ch.is_ascii_alphanumeric())
}

/// Read a finished run's manifest (and its metrics sidecar) into a `ModelRecord`,
/// merging in the fields already parsed from the training status body. Returns
/// None if the run id is unsafe or the manifest is missing/unreadable.
fn read_model_manifest(
    models_root: &Path,
    slug: &str,
    run_id: &str,
    status: &Value,
) -> Option<db::ModelRecord> {
    if !is_safe_slug(slug) || !is_safe_run_id(run_id) {
        return None;
    }
    let run_dir = models_root.join(slug).join("runs").join(run_id);
    let manifest_raw = fs::read_to_string(run_dir.join("manifest.json")).ok()?;
    let manifest: Value = serde_json::from_str(&manifest_raw).ok()?;
    let metrics = fs::read_to_string(run_dir.join(format!("{slug}_metrics.json"))).ok();
    let mstr = |k: &str| manifest.get(k).and_then(Value::as_str).map(str::to_string);
    let mi64 = |k: &str| manifest.get(k).and_then(Value::as_i64);
    let sstr = |k: &str| status.get(k).and_then(Value::as_str).map(str::to_string);
    // The manifest carries the full eval object; the metrics sidecar is stored
    // raw. Prefer manifest-recorded knobs (the effective, resolved values) and
    // fall back to the terminal status body for the few older fields.
    let eval = manifest
        .get("eval")
        .filter(|v| !v.is_null())
        .map(|v| v.to_string());
    Some(db::ModelRecord {
        slug: slug.to_string(),
        phrase: mstr("phrase").or_else(|| sstr("phrase")).unwrap_or_default(),
        run_id: run_id.to_string(),
        onnx_path: mstr("onnx_path").unwrap_or_else(|| format!("runs/{run_id}/{slug}.onnx")),
        onnx_sha256: mstr("onnx_sha256"),
        pt_sha256: mstr("pt_sha256"),
        steps: mi64("steps").or_else(|| status.get("steps").and_then(Value::as_i64)),
        model_size: mstr("model_size").or_else(|| sstr("model_size")),
        personal: manifest
            .get("personal")
            .and_then(Value::as_bool)
            .or_else(|| status.get("personal").and_then(Value::as_bool))
            .unwrap_or(false),
        git_commit: mstr("git_commit"),
        metrics,
        started_at: mstr("started_at").or_else(|| sstr("started_at")),
        finished_at: mstr("finished_at").or_else(|| sstr("updated_at")),
        model_type: mstr("model_type"),
        token_type: mstr("token_type"),
        positive_boost: mi64("positive_boost"),
        n_samples: mi64("n_samples"),
        n_samples_val: mi64("n_samples_val"),
        positive_per_batch: mi64("positive_per_batch"),
        target_fp_per_hour: manifest.get("target_fp_per_hour").and_then(Value::as_f64),
        context_fix: manifest.get("context_fix").and_then(Value::as_bool),
        real_positive: mi64("real_positive"),
        real_negative: mi64("real_negative"),
        real_background: mi64("real_background"),
        trainer_image: mstr("trainer_image"),
        onnx_bytes: mi64("onnx_bytes"),
        eval,
        params_json: Some(manifest_raw),
        synth_positive: mi64("synth_positive"),
        kokoro_positive: mi64("kokoro_positive"),
        total_positive_input: mi64("total_positive_input"),
    })
}

/// POST the WAV to the scorer service and parse its rolling-score curve.
async fn run_scorer(
    scorer_url: &str,
    wav: Vec<u8>,
    slug: &str,
    run: Option<&str>,
    mode: &str,
    step_ms: u64,
    keep_ms: u64,
) -> Result<ScorerCurve, AppError> {
    let part = reqwest::multipart::Part::bytes(wav)
        .file_name(format!("{slug}.wav"))
        .mime_str("audio/wav")
        .map_err(|error| AppError::internal(format!("prepare scorer upload: {error}")))?;
    let mut form = reqwest::multipart::Form::new()
        .part("file", part)
        .text("slug", slug.to_string())
        .text("mode", mode.to_string())
        .text("step_ms", step_ms.to_string())
        .text("keep_ms", keep_ms.to_string());
    if let Some(run) = run {
        form = form.text("run", run.to_string());
    }
    let endpoint = format!("{}/score", scorer_url.trim_end_matches('/'));
    let response = reqwest::Client::new()
        .post(endpoint)
        .multipart(form)
        .send()
        .await
        .map_err(|error| AppError::internal(format!("scorer request failed: {error}")))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| AppError::internal(format!("scorer response read failed: {error}")))?;
    if !status.is_success() {
        return Err(AppError::internal(format!(
            "scorer returned {status}: {}",
            body.trim()
        )));
    }
    serde_json::from_str(&body)
        .map_err(|error| AppError::internal(format!("scorer response JSON failed: {error}")))
}

/// Normalize a spoken token for phrase matching: lowercase, keep only letters
/// and digits. Whisper emits leading spaces and stray punctuation on words.
fn normalize_token(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Find every run of transcript words matching the trigger phrase and attach the
/// model's peak score in the detection window aligned to the phrase tail. The
/// model fires when the phrase ends the 2s window, so the peak sits near the
/// last word's end; we search a small band around it.
fn locate_targets(
    phrase: &str,
    words: &[db::WordRow],
    curve: &ScorerCurve,
    threshold: f64,
) -> Vec<ScoreTarget> {
    let tokens: Vec<String> = phrase
        .split_whitespace()
        .map(normalize_token)
        .filter(|t| !t.is_empty())
        .collect();
    if tokens.is_empty() || words.is_empty() {
        return Vec::new();
    }
    let normalized: Vec<String> = words.iter().map(|w| normalize_token(&w.word)).collect();
    // First pass: locate every phrase occurrence and its Whisper start/end.
    let mut occ: Vec<(i64, i64, String)> = Vec::new();
    let mut i = 0;
    while i + tokens.len() <= normalized.len() {
        if normalized[i..i + tokens.len()] == tokens[..] {
            let start_ms = words[i].start_ms;
            let end_ms = words[i + tokens.len() - 1].end_ms;
            let text = words[i..i + tokens.len()]
                .iter()
                .map(|w| w.word.trim())
                .collect::<Vec<_>>()
                .join(" ");
            occ.push((start_ms, end_ms, text));
            i += tokens.len();
        } else {
            i += 1;
        }
    }
    // Second pass: score each occurrence at the model's peak within its drift
    // window (neighbor-clamped so dense repeats can't claim each other's firing).
    let ends: Vec<f64> = occ.iter().map(|(_, end, _)| *end as f64).collect();
    let windows = target_windows(&ends);
    occ.into_iter()
        .zip(windows)
        .map(|((start_ms, end_ms, text), (lo, hi))| {
            let (peak_time_ms, peak_score) = peak_in_window(curve, lo, hi);
            ScoreTarget {
                text,
                start_ms,
                end_ms,
                peak_score,
                peak_time_ms,
                detected: peak_score >= threshold,
            }
        })
        .collect()
}

/// Model firing can land up to ~1s from Whisper's reported word time.
const MAX_DRIFT_MS: f64 = 1200.0;

/// One detection search window per located phrase, in time order. Whisper word
/// timings drift up to ~1s from where the tail-aligned model actually fires, so
/// each window spans `end ± MAX_DRIFT_MS`. To stop a dense script (many "all
/// set" a couple seconds apart) from letting neighbors claim the same firing,
/// each window is clamped to the midpoints between adjacent phrase ends.
fn target_windows(ends: &[f64]) -> Vec<(f64, f64)> {
    ends.iter()
        .enumerate()
        .map(|(k, &e)| {
            let mut lo = e - MAX_DRIFT_MS;
            let mut hi = e + MAX_DRIFT_MS;
            if k > 0 {
                lo = lo.max((ends[k - 1] + e) / 2.0);
            }
            if k + 1 < ends.len() {
                hi = hi.min((e + ends[k + 1]) / 2.0);
            }
            (lo.max(0.0), hi)
        })
        .collect()
}

/// Highest score (and its time) among curve points within [lo_ms, hi_ms].
fn peak_in_window(curve: &ScorerCurve, lo_ms: f64, hi_ms: f64) -> (f64, f64) {
    let mut best = (lo_ms.max(0.0), 0.0_f64);
    for (t, s) in curve.times_ms.iter().zip(curve.scores.iter()) {
        if *t >= lo_ms && *t <= hi_ms && *s >= best.1 {
            best = (*t, *s);
        }
    }
    best
}

/// Count above-threshold detections that do not overlap any target window. A
/// contiguous run above threshold counts once. A detection whose time falls in
/// any target's search band is a true positive, not a false one.
fn count_false_positives(curve: &ScorerCurve, targets: &[ScoreTarget], threshold: f64) -> usize {
    let ends: Vec<f64> = targets.iter().map(|t| t.end_ms as f64).collect();
    let bands = target_windows(&ends);
    let mut count = 0;
    let mut in_run = false;
    for (t, s) in curve.times_ms.iter().zip(curve.scores.iter()) {
        let hot = *s >= threshold;
        let near_target = bands.iter().any(|(lo, hi)| *t >= *lo && *t <= *hi);
        if hot && !near_target {
            if !in_run {
                count += 1;
                in_run = true;
            }
        } else {
            in_run = false;
        }
    }
    count
}

async fn review_audio(
    State(state): State<AppState>,
    AxumPath((slug, category, file_name)): AxumPath<(String, String, String)>,
) -> Result<impl IntoResponse, AppError> {
    let path = review_clip_path(&state.data_root, &slug, &category, &file_name)?;
    let bytes = fs::read(path)?;
    Ok(([(header::CONTENT_TYPE, "audio/wav")], bytes))
}

async fn delete_review_clip(
    State(state): State<AppState>,
    AxumPath((slug, category, file_name)): AxumPath<(String, String, String)>,
) -> Result<Json<DeleteReviewClipResponse>, AppError> {
    let path = review_clip_path(&state.data_root, &slug, &category, &file_name)?;
    let marked = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::delete_slice_by_file(&conn, &slug, &category, &file_name).map_err(db_error)?
    };
    let removed_file = if path.exists() {
        fs::remove_file(path)?;
        true
    } else {
        false
    };
    Ok(Json(DeleteReviewClipResponse {
        status: "deleted",
        deleted: marked || removed_file,
    }))
}

/// Directory holding the F5 voice-cloned positives for a slug. The trainer
/// writes these to `<repo>/data/synth_f5/<slug>/positive`; the container mounts
/// the repo `data/` at the parent of `data_root` (DATA_ROOT=/data/real →
/// /data), so the synth bucket is a sibling of `data_root`.
fn synth_positive_dir(data_root: &Path, slug: &str) -> PathBuf {
    data_root
        .parent()
        .unwrap_or(data_root)
        .join("synth_f5")
        .join(slug)
        .join("positive")
}

#[derive(Serialize)]
struct SyntheticSample {
    id: String,
    file_name: String,
    text: String,
}

#[derive(Serialize)]
struct SyntheticSamplesResponse {
    slug: String,
    phrase: String,
    total: usize,
    sampled: usize,
    samples: Vec<SyntheticSample>,
}

/// Return a representative sample of the F5 synthetic positives so the Review
/// page can spot-check them by ear. With potentially thousands of clips we take
/// an evenly-spaced stride across the sorted batch (start, middle, end), not the
/// first N — deterministic, no RNG. The clips are all the wake phrase, so each
/// sample carries the project phrase as its label text.
/// Delete every F5 synthetic positive for a wake word (the whole
/// data/synth_f5/<slug> tree). Synth clips are filesystem-only with no DB rows,
/// so this is a plain directory removal. Refuses while a generation run is in
/// flight so we don't yank the directory out from under it.
async fn delete_synth(
    State(state): State<AppState>,
    AxumPath(slug): AxumPath<String>,
) -> Result<Json<Value>, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request(format!("unsafe wake word slug: {slug}")));
    }
    {
        let jobs = state.synth_jobs.lock().expect("synth jobs lock poisoned");
        if jobs.get(&slug).map(|j| j.running).unwrap_or(false) {
            return Err(AppError::bad_request(
                "a generation run is in progress; wait for it to finish before deleting",
            ));
        }
    }
    // The synth bucket is data/synth_f5/<slug>/positive; remove its <slug> parent
    // so nothing for this wake word is left behind.
    let slug_dir = synth_positive_dir(&state.data_root, &slug)
        .parent()
        .map(Path::to_path_buf);
    let removed = match slug_dir {
        Some(dir) if dir.exists() => {
            let count = count_wavs(&synth_positive_dir(&state.data_root, &slug));
            fs::remove_dir_all(&dir)
                .map_err(|e| AppError::internal(format!("failed to delete synth dir: {e}")))?;
            count
        }
        _ => 0,
    };
    Ok(Json(json!({ "status": "ok", "slug": slug, "deleted": removed })))
}

async fn synthetic_samples(
    State(state): State<AppState>,
    AxumPath(slug): AxumPath<String>,
) -> Result<Json<SyntheticSamplesResponse>, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request(format!("unsafe wake word slug: {slug}")));
    }
    let dir = synth_positive_dir(&state.data_root, &slug);
    let mut files: Vec<String> = Vec::new();
    if dir.is_dir() {
        for entry in fs::read_dir(&dir)? {
            let name = entry?.file_name().to_string_lossy().to_string();
            if name.ends_with(".wav") {
                files.push(name);
            }
        }
    }
    files.sort();
    let total = files.len();

    const MAX_SAMPLES: usize = 24;
    let sampled: Vec<String> = if total <= MAX_SAMPLES {
        files
    } else {
        let stride = total as f64 / MAX_SAMPLES as f64;
        (0..MAX_SAMPLES)
            .map(|i| files[((i as f64) * stride) as usize].clone())
            .collect()
    };

    let phrase = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::project_phrase(&conn, &slug)
            .map_err(db_error)?
            .map(|(phrase, _)| phrase)
            .unwrap_or_default()
    };

    let samples = sampled
        .iter()
        .map(|name| SyntheticSample {
            id: name.trim_end_matches(".wav").to_string(),
            file_name: name.clone(),
            text: phrase.clone(),
        })
        .collect();

    Ok(Json(SyntheticSamplesResponse {
        slug,
        phrase,
        total,
        sampled: sampled.len(),
        samples,
    }))
}

/// Serve one F5 synthetic positive WAV by file name. Mirrors `review_audio`.
async fn synthetic_audio(
    State(state): State<AppState>,
    AxumPath((slug, file_name)): AxumPath<(String, String)>,
) -> Result<impl IntoResponse, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request(format!("unsafe wake word slug: {slug}")));
    }
    if file_name.contains('/')
        || file_name.contains('\\')
        || file_name.contains("..")
        || !file_name.ends_with(".wav")
    {
        return Err(AppError::bad_request(format!("unsafe file name: {file_name}")));
    }
    let path = synth_positive_dir(&state.data_root, &slug).join(&file_name);
    let bytes = fs::read(path)?;
    Ok(([(header::CONTENT_TYPE, "audio/wav")], bytes))
}

/// The container running the resident F5-TTS model (speech_services stack).
fn f5_container() -> String {
    env::var("F5_CONTAINER").unwrap_or_else(|_| "speech-f5tts".to_string())
}

/// Does this directory hold at least one `.wav`?
fn dir_has_wav(dir: &Path) -> bool {
    fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .any(|e| e.file_name().to_string_lossy().ends_with(".wav"))
        })
        .unwrap_or(false)
}

/// How many `*.wav` files a directory holds (0 if missing/unreadable).
fn count_wavs(dir: &Path) -> usize {
    fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_string_lossy().ends_with(".wav"))
                .count()
        })
        .unwrap_or(0)
}

/// Resolve the F5 cloning-reference directory for a slug: the user's real
/// positive clips, whose transcript is exactly the phrase and whose length is
/// F5-friendly. Enrollment has been retired as a reference — a long passage
/// starved and leaked its own tail text into the output ("warm gold"), so F5
/// now only ever clones from real positives.
fn resolve_synth_refs(data_root: &Path, slug: &str) -> Result<PathBuf, AppError> {
    let positive_dir = data_root.join(slug).join("positive");
    if dir_has_wav(&positive_dir) {
        Ok(positive_dir)
    } else {
        Err(AppError::bad_request(
            "no positive clips to clone from; record positive takes first",
        ))
    }
}

/// Drive one F5 batch for a slug: resolve its reference clips, run the resident
/// generator, and land the resampled clips in the synth bucket. Returns how many
/// clips this run produced. Shared by the manual `/synth/generate` endpoint and
/// the train-time pre-generation step.
async fn generate_synth_batch(
    state: &AppState,
    slug: &str,
    phrase: &str,
    count: usize,
) -> Result<usize, AppError> {
    let refs = resolve_synth_refs(&state.data_root, slug)?;
    let synth_dir = synth_positive_dir(&state.data_root, slug);
    let cp_refs = refs.to_string_lossy().to_string();
    let cp_out = synth_dir.to_string_lossy().to_string();
    let cp_gen_py = "/trainer/scripts/f5_gen_positives.py".to_string();
    run_synth_generation(
        slug, phrase, count, &refs, &cp_refs, &cp_gen_py, &cp_out, &synth_dir,
    )
    .await
}

/// Ensure the F5 synth bucket for `slug` holds `vt.f5_count` voice-cloned
/// positives before the trainer assembles them, generating a fresh batch when it
/// is short. Reuses an existing batch that is already large enough (so repeated
/// trains don't re-pay F5's cost); otherwise clears the bucket and regenerates
/// exactly `f5_count` (the generator names clips `f5_00000..`, so a clean bucket
/// yields exactly the count the user asked for with no stale leftovers). Writes
/// an `f5gen` phase into train_status.json so the app's training screen shows it.
/// Non-fatal: on any failure it logs, records the error, and returns whatever
/// clips exist so training still proceeds. Returns the final clip count.
async fn ensure_f5_positives(
    state: &AppState,
    slug: &str,
    phrase: &str,
    vt: &ValidatedTrain,
) -> usize {
    let target = vt.f5_count as usize;
    let synth_dir = synth_positive_dir(&state.data_root, slug);
    if target == 0 {
        return count_wavs(&synth_dir);
    }
    let existing = count_wavs(&synth_dir);
    if existing >= target {
        return existing;
    }
    // Can we clone at all? Resolve references before touching the bucket so a
    // slug with no positives keeps whatever clips it already has.
    if let Err(e) = resolve_synth_refs(&state.data_root, slug) {
        eprintln!("f5 pregen: no reference clips for {slug}: {}", e.message);
        return existing;
    }
    // Don't collide with a manual generation already running for this slug.
    {
        let mut jobs = state.synth_jobs.lock().expect("synth jobs lock poisoned");
        if jobs.get(slug).map(|j| j.running).unwrap_or(false) {
            eprintln!(
                "f5 pregen: a manual generation is already running for {slug}; \
                 training on the {existing} existing clips"
            );
            return existing;
        }
        jobs.insert(
            slug.to_string(),
            SynthJob {
                running: true,
                requested: target,
                wrote: existing,
                error: None,
            },
        );
    }
    write_f5_status(
        state,
        slug,
        phrase,
        vt,
        target,
        0,
        &format!("generating {target} voice-cloned positives (F5)"),
    );
    // Fresh batch: clear the bucket so the count is exactly `target` (no stale
    // clips from a previous, larger run linger).
    let _ = fs::remove_dir_all(&synth_dir);
    let result = generate_synth_batch(state, slug, phrase, target).await;
    let wrote = *result.as_ref().unwrap_or(&0);
    if let Err(e) = &result {
        eprintln!("f5 pregen failed for {slug}: {}", e.message);
    }
    {
        let mut jobs = state.synth_jobs.lock().expect("synth jobs lock poisoned");
        if let Some(j) = jobs.get_mut(slug) {
            j.running = false;
            j.wrote = wrote;
            j.error = result.as_ref().err().map(|e| e.message.clone());
        }
    }
    let msg = match &result {
        Ok(w) => format!("generated {w} voice-cloned positives (F5)"),
        Err(e) => format!(
            "F5 generation failed: {}; training on existing clips",
            e.message
        ),
    };
    write_f5_status(state, slug, phrase, vt, target, wrote, &msg);
    count_wavs(&synth_dir)
}

/// Write the pre-training `f5gen` phase into a slug's train_status.json. The
/// trainer overwrites this file with its own phases once it launches; until then
/// the app sees a `running` status stepped `f5gen`, with the F5 clip counts.
fn write_f5_status(
    state: &AppState,
    slug: &str,
    phrase: &str,
    vt: &ValidatedTrain,
    requested: usize,
    wrote: usize,
    message: &str,
) {
    let dir = state.models_root.join(slug);
    if fs::create_dir_all(&dir).is_err() {
        return;
    }
    let now = Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
    let body = json!({
        "slug": slug,
        "phrase": phrase,
        "state": "running",
        "step": "f5gen",
        "exit_code": 0,
        "message": message,
        "steps": vt.steps,
        "model_size": vt.model_size,
        "personal": vt.personal,
        "f5_requested": requested,
        "f5_wrote": wrote,
        "started_at": now,
        "updated_at": now,
    });
    let _ = fs::write(dir.join("train_status.json"), body.to_string());
}

/// Run `docker` and fail with its stderr if the command exits non-zero.
async fn docker_ok(args: Vec<String>) -> Result<std::process::Output, AppError> {
    let output = run_docker(args).await?;
    if !output.status.success() {
        return Err(AppError::internal(format!(
            "docker command failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(output)
}

#[derive(Serialize)]
struct GenerateSynthResponse {
    status: &'static str,
    slug: String,
    requested: usize,
}

/// Kick off an F5 voice-cloned positive batch for a wake word, seeded by the
/// user's real positive takes. Runs the resident F5 model inside the speech-f5tts
/// container over the mounted docker socket, then copies the 16 kHz clips into the
/// slug's synth bucket. Returns immediately; progress is polled via the status
/// endpoint.
async fn generate_synth(
    State(state): State<AppState>,
    AxumPath(slug): AxumPath<String>,
    RawQuery(query): RawQuery,
) -> Result<Json<GenerateSynthResponse>, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request(format!("unsafe wake word slug: {slug}")));
    }
    let params = parse_query(query.as_deref());
    let count: usize = params
        .get("count")
        .and_then(|v| v.parse().ok())
        .unwrap_or(60)
        .clamp(1, 1000);

    // Clone from the user's SHORT real positive clips: their transcript is exactly
    // the phrase and their length is F5-friendly. (F5 sizes the spoken output from
    // the reference's rate, so a long passage starves the short wake phrase and
    // leaks its own text.) Refuse early if there is nothing to clone from.
    let refs_server_dir = resolve_synth_refs(&state.data_root, &slug)?;

    let phrase = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::project_phrase(&conn, &slug)
            .map_err(db_error)?
            .map(|(phrase, _)| phrase)
            .unwrap_or_default()
    };
    if phrase.trim().is_empty() {
        return Err(AppError::bad_request(
            "unknown wake phrase for this slug",
        ));
    }

    {
        let mut jobs = state.synth_jobs.lock().expect("synth jobs lock poisoned");
        if jobs.get(&slug).map(|j| j.running).unwrap_or(false) {
            return Err(AppError::bad_request(
                "a generation run is already in progress for this wake word",
            ));
        }
        jobs.insert(
            slug.clone(),
            SynthJob {
                running: true,
                requested: count,
                wrote: 0,
                error: None,
            },
        );
    }

    // These paths feed `docker cp`, whose local side is resolved inside THIS
    // container, so they must be container paths (the repo `data/` is mounted at
    // /data, trainer scripts at /trainer), not host paths.
    let synth_server_dir = synth_positive_dir(&state.data_root, &slug);
    let cp_refs = refs_server_dir.to_string_lossy().to_string();
    let cp_gen_py = "/trainer/scripts/f5_gen_positives.py".to_string();
    let cp_out = synth_server_dir.to_string_lossy().to_string();

    let task_state = state.clone();
    let task_slug = slug.clone();
    let task_phrase = phrase.clone();
    tokio::spawn(async move {
        let result = run_synth_generation(
            &task_slug,
            &task_phrase,
            count,
            &refs_server_dir,
            &cp_refs,
            &cp_gen_py,
            &cp_out,
            &synth_server_dir,
        )
        .await;
        let mut jobs = task_state.synth_jobs.lock().expect("synth jobs lock poisoned");
        let entry = jobs.entry(task_slug).or_insert(SynthJob {
            running: false,
            requested: count,
            wrote: 0,
            error: None,
        });
        entry.running = false;
        match result {
            Ok(wrote) => {
                entry.wrote = wrote;
                entry.error = None;
            }
            Err(e) => entry.error = Some(e.message),
        }
    });

    Ok(Json(GenerateSynthResponse {
        status: "ok",
        slug,
        requested: count,
    }))
}

/// Drive one F5 batch end to end: stage ≤8 references into the F5 container, run
/// the resident generator (writing 16 kHz clips), copy them into the synth
/// bucket, and clean up. Returns how many clips this run produced. Every path
/// handed to `docker cp` is a path INSIDE this sync-server container, since the
/// docker CLI resolves the local side of a cp against its own filesystem — the
/// repo `data/` is mounted at /data and trainer scripts at /trainer.
#[allow(clippy::too_many_arguments)]
async fn run_synth_generation(
    slug: &str,
    phrase: &str,
    count: usize,
    refs_server_dir: &Path,
    cp_refs: &str,
    cp_gen_py: &str,
    cp_out: &str,
    synth_server_dir: &Path,
) -> Result<usize, AppError> {
    let container = f5_container();
    let stamp = now_ms();
    let scratch = format!("/tmp/f5gen_{slug}_{stamp}");
    let crefs = format!("{scratch}/refs");
    let cout = format!("{scratch}/out");

    docker_ok(vec![
        "exec".into(),
        container.clone(),
        "mkdir".into(),
        "-p".into(),
        crefs.clone(),
        cout.clone(),
    ])
    .await?;

    // Stage a small, clean subset of references (rotated inside the python).
    let mut names: Vec<String> = fs::read_dir(refs_server_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.ends_with(".wav"))
        .collect();
    names.sort();
    names.truncate(8);
    if names.is_empty() {
        return Err(AppError::internal("no reference wavs to stage"));
    }
    for name in &names {
        docker_ok(vec![
            "cp".into(),
            format!("{cp_refs}/{name}"),
            format!("{container}:{crefs}/{name}"),
        ])
        .await?;
        // An enrollment reference ships its exact passage in a sibling .txt; stage
        // it too so F5 gets the right ref_text.
        let sidecar = format!("{}.txt", name.trim_end_matches(".wav"));
        if refs_server_dir.join(&sidecar).is_file() {
            docker_ok(vec![
                "cp".into(),
                format!("{cp_refs}/{sidecar}"),
                format!("{container}:{crefs}/{sidecar}"),
            ])
            .await?;
        }
    }

    // Stage and run the resident-model generator, writing 16 kHz clips directly.
    docker_ok(vec![
        "cp".into(),
        cp_gen_py.to_string(),
        format!("{container}:{scratch}/gen.py"),
    ])
    .await?;
    docker_ok(vec![
        "exec".into(),
        container.clone(),
        "python3".into(),
        format!("{scratch}/gen.py"),
        "--refs-dir".into(),
        crefs.clone(),
        "--ref-text".into(),
        phrase.to_string(),
        "--gen-text".into(),
        phrase.to_string(),
        "--out-dir".into(),
        cout.clone(),
        "--count".into(),
        count.to_string(),
        "--out-sr".into(),
        "16000".into(),
        // Fidelity knobs (F5 defaults unless overridden in the environment).
        // Raise F5_NFE_STEP for a sharper, more faithful render (slower);
        // raise F5_CFG_STRENGTH to hew closer to the user's timbre.
        "--nfe-step".into(),
        env::var("F5_NFE_STEP").unwrap_or_else(|_| "32".to_string()),
        "--cfg-strength".into(),
        env::var("F5_CFG_STRENGTH").unwrap_or_else(|_| "2.0".to_string()),
    ])
    .await?;

    // Copy the finished clips into the synth bucket (dir must exist first).
    fs::create_dir_all(synth_server_dir)?;
    docker_ok(vec![
        "cp".into(),
        format!("{container}:{cout}/."),
        format!("{cp_out}/"),
    ])
    .await?;

    // Count what this run produced from the container's output dir.
    let counted = docker_ok(vec![
        "exec".into(),
        container.clone(),
        "sh".into(),
        "-c".into(),
        format!("ls -1 {cout}/*.wav 2>/dev/null | wc -l"),
    ])
    .await?;
    let wrote = String::from_utf8_lossy(&counted.stdout)
        .trim()
        .parse::<usize>()
        .unwrap_or(0);

    // Best-effort scratch cleanup.
    let _ = run_docker(vec![
        "exec".into(),
        container,
        "rm".into(),
        "-rf".into(),
        scratch,
    ])
    .await;

    Ok(wrote)
}

#[derive(Serialize)]
struct GenerateSynthStatusResponse {
    slug: String,
    running: bool,
    requested: usize,
    wrote: usize,
    error: Option<String>,
    idle: bool,
}

/// Report the state of a slug's F5 generation run for the app to poll.
async fn generate_synth_status(
    State(state): State<AppState>,
    AxumPath(slug): AxumPath<String>,
) -> Result<Json<GenerateSynthStatusResponse>, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request(format!("unsafe wake word slug: {slug}")));
    }
    let job = {
        let jobs = state.synth_jobs.lock().expect("synth jobs lock poisoned");
        jobs.get(&slug).cloned()
    };
    Ok(Json(match job {
        Some(j) => GenerateSynthStatusResponse {
            slug,
            running: j.running,
            requested: j.requested,
            wrote: j.wrote,
            error: j.error,
            idle: false,
        },
        None => GenerateSynthStatusResponse {
            slug,
            running: false,
            requested: 0,
            wrote: 0,
            error: None,
            idle: true,
        },
    }))
}

fn load_settings(path: &Path) -> ServerSettings {
    fs::read_to_string(path)
        .ok()
        .and_then(|contents| serde_json::from_str(&contents).ok())
        .unwrap_or_default()
}

fn save_settings(path: &Path, settings: &ServerSettings) -> Result<(), AppError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let contents = serde_json::to_string_pretty(settings)
        .map_err(|error| AppError::internal(format!("serialize settings: {error}")))?;
    fs::write(path, contents)?;
    Ok(())
}

fn clean_optional_url(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn extract_zip(bytes: &[u8], dest: &Path) -> Result<(), AppError> {
    let reader = Cursor::new(bytes);
    let mut archive = ZipArchive::new(reader)
        .map_err(|error| AppError::bad_request(format!("invalid zip: {error}")))?;

    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .map_err(|error| AppError::bad_request(format!("invalid zip entry: {error}")))?;
        let name = entry.name();
        let output = safe_join(dest, name)?;
        if entry.is_dir() {
            fs::create_dir_all(&output)?;
            continue;
        }
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = File::create(output)?;
        io::copy(&mut entry, &mut file)?;
    }
    Ok(())
}

fn read_manifest(bundle: &Path) -> Result<Manifest, AppError> {
    let manifest_path = bundle.join("manifest.json");
    let contents = fs::read_to_string(&manifest_path)
        .map_err(|error| AppError::bad_request(format!("missing manifest: {error}")))?;
    serde_json::from_str(&contents)
        .map_err(|error| AppError::bad_request(format!("invalid manifest JSON: {error}")))
}

fn validate_manifest(manifest: &Manifest) -> Result<(), AppError> {
    if manifest.schema_version != 1 {
        return Err(AppError::bad_request(format!(
            "unsupported schema_version: {}",
            manifest.schema_version
        )));
    }
    if !is_safe_slug(&manifest.wake_word.slug) {
        return Err(AppError::bad_request(format!(
            "unsafe wake_word.slug: {}",
            manifest.wake_word.slug
        )));
    }
    for (index, clip) in manifest.clips.iter().enumerate() {
        if clip.id.is_empty() {
            return Err(AppError::bad_request(format!("clip {index} missing id")));
        }
        if clip.file.is_empty() {
            return Err(AppError::bad_request(format!("clip {index} missing file")));
        }
        if category_for_label(&clip.label).is_none() {
            return Err(AppError::bad_request(format!(
                "clip {index} has unknown label {}",
                clip.label
            )));
        }
    }
    for (index, recording) in manifest.bulk_recordings.iter().enumerate() {
        if recording.id.is_empty() {
            return Err(AppError::bad_request(format!(
                "bulk recording {index} missing id"
            )));
        }
        if recording.file.is_empty() {
            return Err(AppError::bad_request(format!(
                "bulk recording {index} missing file"
            )));
        }
        // Only a mixed/legacy take needs a script to align against. A token
        // (`positive`) or plain `negative` take is a straight repeated read with
        // no prompt, so an empty script is expected; a `hard_negative` take does
        // carry its near-miss prompt but is not required to.
        let scripted_kind = matches!(recording.kind.trim(), "" | "mixed");
        if scripted_kind && recording.script.trim().is_empty() {
            return Err(AppError::bad_request(format!(
                "bulk recording {index} missing script"
            )));
        }
    }
    for (index, recording) in manifest.background_recordings.iter().enumerate() {
        if recording.id.is_empty() {
            return Err(AppError::bad_request(format!(
                "background recording {index} missing id"
            )));
        }
        if recording.file.is_empty() {
            return Err(AppError::bad_request(format!(
                "background recording {index} missing file"
            )));
        }
    }
    for (index, recording) in manifest.test_recordings.iter().enumerate() {
        if recording.id.is_empty() {
            return Err(AppError::bad_request(format!(
                "test recording {index} missing id"
            )));
        }
        if !recording.id.starts_with("test_") {
            return Err(AppError::bad_request(format!(
                "test recording {index} id {} must start with test_",
                recording.id
            )));
        }
        if recording.file.is_empty() {
            return Err(AppError::bad_request(format!(
                "test recording {index} missing file"
            )));
        }
        // A test take is a free-form spoken take scored against a model; it needs
        // no script, so an empty one is fine.
    }
    Ok(())
}

fn validate_wavs(bundle: &Path, manifest: &Manifest) -> Result<Vec<String>, AppError> {
    let mut warnings = Vec::new();
    for clip in &manifest.clips {
        let path = safe_join(bundle, &clip.file)?;
        let mut reader = WavReader::open(&path).map_err(|error| {
            AppError::bad_request(format!("{}: cannot read WAV: {error}", clip.id))
        })?;
        let spec = reader.spec();
        let duration = reader.duration() as f64 / spec.sample_rate as f64;
        if spec.channels != 1 {
            warnings.push(format!(
                "{}: expected mono, got {} channels",
                clip.id, spec.channels
            ));
        }
        if spec.bits_per_sample != 16 {
            warnings.push(format!(
                "{}: expected 16-bit PCM, got {} bits",
                clip.id, spec.bits_per_sample
            ));
        }
        if spec.sample_rate != 16_000 {
            warnings.push(format!(
                "{}: expected 16000 Hz, got {}",
                clip.id, spec.sample_rate
            ));
        }
        if duration <= 0.0 {
            warnings.push(format!("{}: zero duration", clip.id));
        }
        if duration > 5.0 {
            warnings.push(format!("{}: long clip {:.2}s", clip.id, duration));
        }

        let mut peak = 0i32;
        let mut total = 0f64;
        let mut count = 0f64;
        for sample in reader.samples::<i16>() {
            let sample = sample
                .map_err(|error| AppError::bad_request(format!("bad WAV sample: {error}")))?;
            let sample_i32 = i32::from(sample);
            peak = peak.max(sample_i32.abs());
            total += f64::from(sample) * f64::from(sample);
            count += 1.0;
        }
        let rms = if count > 0.0 {
            (total / count).sqrt()
        } else {
            0.0
        };
        if peak >= 32_760 {
            warnings.push(format!("{}: possible clipping, peak={peak}", clip.id));
        }
        if rms < 50.0 && duration > 0.0 {
            warnings.push(format!("{}: very quiet audio, rms={rms:.1}", clip.id));
        }
    }
    for recording in &manifest.bulk_recordings {
        let path = safe_join(bundle, &recording.file)?;
        let reader = WavReader::open(&path).map_err(|error| {
            AppError::bad_request(format!("{}: cannot read bulk WAV: {error}", recording.id))
        })?;
        let spec = reader.spec();
        if spec.channels != 1 {
            warnings.push(format!(
                "{}: expected mono bulk recording, got {} channels",
                recording.id, spec.channels
            ));
        }
        if spec.bits_per_sample != 16 {
            warnings.push(format!(
                "{}: expected 16-bit PCM bulk recording, got {} bits",
                recording.id, spec.bits_per_sample
            ));
        }
        if spec.sample_rate != 16_000 {
            warnings.push(format!(
                "{}: expected 16000 Hz bulk recording, got {}",
                recording.id, spec.sample_rate
            ));
        }
    }
    for recording in &manifest.background_recordings {
        let path = safe_join(bundle, &recording.file)?;
        let reader = WavReader::open(&path).map_err(|error| {
            AppError::bad_request(format!(
                "{}: cannot read background WAV: {error}",
                recording.id
            ))
        })?;
        let spec = reader.spec();
        if spec.channels != 1 {
            warnings.push(format!(
                "{}: expected mono background recording, got {} channels",
                recording.id, spec.channels
            ));
        }
        if spec.bits_per_sample != 16 {
            warnings.push(format!(
                "{}: expected 16-bit PCM background recording, got {} bits",
                recording.id, spec.bits_per_sample
            ));
        }
        if spec.sample_rate != 16_000 {
            warnings.push(format!(
                "{}: expected 16000 Hz background recording, got {}",
                recording.id, spec.sample_rate
            ));
        }
    }
    for recording in &manifest.test_recordings {
        let path = safe_join(bundle, &recording.file)?;
        let reader = WavReader::open(&path).map_err(|error| {
            AppError::bad_request(format!("{}: cannot read test WAV: {error}", recording.id))
        })?;
        let spec = reader.spec();
        if spec.channels != 1 {
            warnings.push(format!(
                "{}: expected mono test recording, got {} channels",
                recording.id, spec.channels
            ));
        }
        if spec.bits_per_sample != 16 {
            warnings.push(format!(
                "{}: expected 16-bit PCM test recording, got {} bits",
                recording.id, spec.bits_per_sample
            ));
        }
        if spec.sample_rate != 16_000 {
            warnings.push(format!(
                "{}: expected 16000 Hz test recording, got {}",
                recording.id, spec.sample_rate
            ));
        }
    }
    Ok(warnings)
}

fn import_bundle(
    bundle: &Path,
    manifest: &Manifest,
    data_root: &Path,
    db: &Mutex<Connection>,
) -> Result<usize, AppError> {
    let slug = manifest.wake_word.slug.clone();
    let phrase = manifest.wake_word.phrase();
    let external_id = manifest
        .wake_word
        .extra
        .get("id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty());
    let dest_root = data_root.join(&slug);
    fs::create_dir_all(&dest_root)?;

    let now = now_ms();
    {
        let conn = db.lock().expect("db lock poisoned");
        db::upsert_project(&conn, &slug, &phrase, external_id, now).map_err(db_error)?;
    }

    let mut imported = 0;
    for clip in &manifest.clips {
        let category = category_for_label(&clip.label).expect("validated label");
        let src = safe_join(bundle, &clip.file)?;
        if !src.is_file() {
            return Err(AppError::bad_request(format!(
                "missing clip file: {}",
                clip.file
            )));
        }

        let dest_dir = dest_root.join(category);
        fs::create_dir_all(&dest_dir)?;
        let clip_phrase = if clip.spoken_phrase.is_empty() {
            clip.label.as_str()
        } else {
            clip.spoken_phrase.as_str()
        };
        let dest = dest_dir.join(format!("{}_{}.wav", clip.id, safe_filename(clip_phrase)));
        if dest.exists() {
            continue;
        }

        fs::copy(&src, &dest)?;
        let row = db::ClipRow {
            id: clip.id.clone(),
            project_slug: slug.clone(),
            label: clip.label.clone(),
            category: category.to_string(),
            spoken_phrase: clip_phrase.to_string(),
            wav_path: dest.to_string_lossy().to_string(),
            source_file: clip.file.clone(),
            bundle: Some("server_sync".to_string()),
        };
        let conn = db.lock().expect("db lock poisoned");
        if db::insert_clip(&conn, &row, now).map_err(db_error)? {
            imported += 1;
        }
    }
    Ok(imported)
}

/// Pull per-take capture provenance out of a recording's flattened `extra`
/// JSON. The app nests it under a `capture` object; missing or empty values
/// become `None` so they never overwrite prior provenance on reprocess.
fn capture_from_extra(extra: &Value) -> db::CaptureMeta {
    let capture = extra.get("capture");
    let text = |key: &str| {
        capture
            .and_then(|value| value.get(key))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToString::to_string)
    };
    let number = |key: &str| {
        capture
            .and_then(|value| value.get(key))
            .and_then(Value::as_i64)
            .filter(|value| *value > 0)
    };
    db::CaptureMeta {
        device_manufacturer: text("device_manufacturer"),
        device_model: text("device_model"),
        os_version: text("os_version"),
        app_version: text("app_version"),
        input_route: text("input_route"),
        source_sample_rate_hz: number("source_sample_rate_hz"),
        source_channels: number("source_channels"),
        session_id: text("session_id"),
    }
}

async fn align_bulk_recordings(
    bundle: &Path,
    manifest: &Manifest,
    data_root: &Path,
    db: &Mutex<Connection>,
    whisper_url: Option<&str>,
) -> Result<AlignmentSummary, AppError> {
    let mut summary = AlignmentSummary::default();
    if manifest.bulk_recordings.is_empty() {
        return Ok(summary);
    }
    let Some(whisper_url) = whisper_url.map(str::trim).filter(|value| !value.is_empty()) else {
        summary
            .warnings
            .push("bulk recordings present but no Whisper server URL configured".to_string());
        return Ok(summary);
    };

    let slug = manifest.wake_word.slug.clone();
    let phrase = manifest.wake_word.phrase();
    let external_id = manifest
        .wake_word
        .extra
        .get("id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string);
    let dest_root = data_root.join(&slug);

    for recording in &manifest.bulk_recordings {
        let source = match safe_join(bundle, &recording.file) {
            Ok(source) => source,
            Err(error) => {
                summary.recordings += 1;
                summary
                    .warnings
                    .push(format!("{}: {}", recording.id, error.message));
                continue;
            }
        };
        align_one_recording(
            &recording.id,
            &recording.script,
            &recording.kind,
            &recording.recorded_at,
            recording.duration_ms,
            &source,
            &slug,
            &phrase,
            external_id.as_deref(),
            &dest_root,
            db,
            whisper_url,
            &capture_from_extra(&recording.extra),
            &mut summary,
        )
        .await?;
    }

    Ok(summary)
}

/// Slice every background take in the bundle into fixed-length background clips.
/// Independent of Whisper, so it runs even when no Whisper URL is configured.
async fn align_background_recordings(
    bundle: &Path,
    manifest: &Manifest,
    data_root: &Path,
    db: &Mutex<Connection>,
) -> Result<AlignmentSummary, AppError> {
    let mut summary = AlignmentSummary::default();
    if manifest.background_recordings.is_empty() {
        return Ok(summary);
    }

    let slug = manifest.wake_word.slug.clone();
    let phrase = manifest.wake_word.phrase();
    let external_id = manifest
        .wake_word
        .extra
        .get("id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string);
    let dest_root = data_root.join(&slug);

    for recording in &manifest.background_recordings {
        let source = match safe_join(bundle, &recording.file) {
            Ok(source) => source,
            Err(error) => {
                summary.recordings += 1;
                summary
                    .warnings
                    .push(format!("{}: {}", recording.id, error.message));
                continue;
            }
        };
        summary.recordings += 1;
        slice_background_recording(
            &recording.id,
            &recording.recorded_at,
            recording.duration_ms,
            &source,
            &slug,
            &phrase,
            external_id.as_deref(),
            &dest_root,
            db,
            &capture_from_extra(&recording.extra),
            &mut summary,
        )?;
    }

    Ok(summary)
}

/// Transcribe every test take in the bundle for word timings, but cut no
/// training slices. Requires Whisper, since scoring needs to locate the wake
/// phrase inside the take; without it the take is stored source-only.
async fn align_test_recordings(
    bundle: &Path,
    manifest: &Manifest,
    data_root: &Path,
    db: &Mutex<Connection>,
    whisper_url: Option<&str>,
) -> Result<AlignmentSummary, AppError> {
    let mut summary = AlignmentSummary::default();
    if manifest.test_recordings.is_empty() {
        return Ok(summary);
    }
    let Some(whisper_url) = whisper_url.map(str::trim).filter(|value| !value.is_empty()) else {
        summary
            .warnings
            .push("test recordings present but no Whisper server URL configured".to_string());
        return Ok(summary);
    };

    let slug = manifest.wake_word.slug.clone();
    let phrase = manifest.wake_word.phrase();
    let external_id = manifest
        .wake_word
        .extra
        .get("id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string);
    let dest_root = data_root.join(&slug);

    for recording in &manifest.test_recordings {
        let source = match safe_join(bundle, &recording.file) {
            Ok(source) => source,
            Err(error) => {
                summary.recordings += 1;
                summary
                    .warnings
                    .push(format!("{}: {}", recording.id, error.message));
                continue;
            }
        };
        // Route through align_one_recording so the shared `test_` branch handles
        // it identically to a reprocess pass.
        align_one_recording(
            &recording.id,
            &recording.script,
            &recording.kind,
            &recording.recorded_at,
            recording.duration_ms,
            &source,
            &slug,
            &phrase,
            external_id.as_deref(),
            &dest_root,
            db,
            whisper_url,
            &capture_from_extra(&recording.extra),
            &mut summary,
        )
        .await?;
    }

    Ok(summary)
}

/// Store a test take: transcribe it so scoring can locate the wake phrase,
/// persist the source WAV and transcript, and record ZERO slices. A test take
/// must never contribute training clips, so this path writes nothing under the
/// positive/negative/background directories.
async fn align_test_one(
    recording_id: &str,
    script: &str,
    recorded_at: &str,
    duration_ms: u64,
    source: &Path,
    slug: &str,
    phrase: &str,
    external_id: Option<&str>,
    dest_root: &Path,
    db: &Mutex<Connection>,
    whisper_url: &str,
    capture: &db::CaptureMeta,
    summary: &mut AlignmentSummary,
) -> Result<(), AppError> {
    let whisper = match transcribe_with_words(whisper_url, source, None).await {
        Ok(whisper) => whisper,
        Err(error) => {
            summary
                .warnings
                .push(format!("{}: {}", recording_id, error.message));
            return Ok(());
        }
    };
    let words = whisper_words(&whisper);

    // Persist the raw source recording, then drop any slice files a prior pass
    // left behind (e.g. if this id was ever mis-processed as a bulk take).
    let (source_wav, source_sha256) = match store_bulk_source(dest_root, recording_id, source) {
        Ok(stored) => stored,
        Err(error) => {
            summary
                .warnings
                .push(format!("{}: {}", recording_id, error.message));
            return Ok(());
        }
    };
    let old_paths = {
        let conn = db.lock().expect("db lock poisoned");
        db::active_slice_paths(&conn, recording_id).map_err(db_error)?
    };
    for old in old_paths {
        let old = PathBuf::from(old);
        if old.is_file() {
            let _ = fs::remove_file(old);
        }
    }

    let word_rows = words
        .iter()
        .map(|word| db::WordRow {
            word: word.word.trim().to_string(),
            start_ms: (word.start * 1000.0).round() as i64,
            end_ms: (word.end * 1000.0).round() as i64,
            probability: word.probability,
        })
        .collect();
    let prompts = script
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToString::to_string)
        .collect();

    let alignment = db::RecordingAlignment {
        recording_id: recording_id.to_string(),
        project_slug: slug.to_string(),
        phrase: phrase.to_string(),
        external_id: external_id.map(ToString::to_string),
        script: script.to_string(),
        kind: "test".to_string(),
        recorded_at: recorded_at.to_string(),
        duration_ms: duration_ms as i64,
        source_wav: source_wav.to_string_lossy().to_string(),
        source_sha256: Some(source_sha256),
        bundle: None,
        transcript_text: whisper.text.trim().to_string(),
        whisper_url: Some(whisper_url.to_string()),
        words: word_rows,
        prompts,
        // Intentionally empty: test takes contribute no training data.
        slices: Vec::new(),
        capture: capture.clone(),
    };
    {
        let mut conn = db.lock().expect("db lock poisoned");
        db::store_recording_alignment(&mut conn, &alignment, now_ms()).map_err(db_error)?;
    }
    if words.is_empty() {
        summary.warnings.push(format!(
            "{}: test take stored but Whisper returned no words; scoring can still run",
            recording_id
        ));
    }
    Ok(())
}

/// Store an enrollment take whole. Enrollment reads are the voice-cloning
/// reference for F5-TTS — one clean take of the user reading a fixed passage —
/// so they must never be sliced into training clips and never touch Whisper: the
/// passage text is already known (it is the recording's `script`), so there is
/// nothing to align. This mirrors `align_test_one` (persist the raw source, drop
/// any stale slices, write an alignment with an empty slice list) but skips
/// transcription entirely. Reprocess re-enters here off the persisted `kind`.
fn store_enrollment_whole(
    recording_id: &str,
    script: &str,
    recorded_at: &str,
    duration_ms: u64,
    source: &Path,
    slug: &str,
    phrase: &str,
    external_id: Option<&str>,
    dest_root: &Path,
    db: &Mutex<Connection>,
    capture: &db::CaptureMeta,
    summary: &mut AlignmentSummary,
) -> Result<(), AppError> {
    let (source_wav, source_sha256) = match store_bulk_source(dest_root, recording_id, source) {
        Ok(stored) => stored,
        Err(error) => {
            summary
                .warnings
                .push(format!("{}: {}", recording_id, error.message));
            return Ok(());
        }
    };
    let old_paths = {
        let conn = db.lock().expect("db lock poisoned");
        db::active_slice_paths(&conn, recording_id).map_err(db_error)?
    };
    for old in old_paths {
        let old = PathBuf::from(old);
        if old.is_file() {
            let _ = fs::remove_file(old);
        }
    }

    let prompts = script
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToString::to_string)
        .collect();

    let alignment = db::RecordingAlignment {
        recording_id: recording_id.to_string(),
        project_slug: slug.to_string(),
        phrase: phrase.to_string(),
        external_id: external_id.map(ToString::to_string),
        script: script.to_string(),
        kind: "enrollment".to_string(),
        recorded_at: recorded_at.to_string(),
        duration_ms: duration_ms as i64,
        source_wav: source_wav.to_string_lossy().to_string(),
        source_sha256: Some(source_sha256),
        bundle: None,
        // The passage text is known; store it as the transcript without Whisper.
        transcript_text: script.trim().to_string(),
        whisper_url: None,
        words: Vec::new(),
        prompts,
        // Intentionally empty: enrollment reads are the F5 reference, not training data.
        slices: Vec::new(),
        capture: capture.clone(),
    };
    {
        let mut conn = db.lock().expect("db lock poisoned");
        db::store_recording_alignment(&mut conn, &alignment, now_ms()).map_err(db_error)?;
    }

    // Also drop the take into a stable `enrollment/` bucket with a sidecar
    // transcript, so the F5 generator can find the reference audio + its exact
    // text without querying the DB. The passage is known, so the .txt is precise.
    if let Err(error) = write_enrollment_reference(dest_root, recording_id, &source_wav, script) {
        summary
            .warnings
            .push(format!("{}: {}", recording_id, error.message));
    }
    Ok(())
}

/// Copy an enrollment take into `<dest_root>/enrollment/<id>.wav` and write its
/// transcript to `<id>.txt` beside it, giving the F5 generator a self-describing
/// reference (audio + exact ref_text) with no DB lookup.
fn write_enrollment_reference(
    dest_root: &Path,
    recording_id: &str,
    source_wav: &Path,
    script: &str,
) -> Result<(), AppError> {
    let dir = dest_root.join("enrollment");
    fs::create_dir_all(&dir)?;
    let safe = safe_filename(recording_id);
    fs::copy(source_wav, dir.join(format!("{safe}.wav")))?;
    fs::write(dir.join(format!("{safe}.txt")), script.trim())?;
    Ok(())
}

/// Align and slice one already-materialized source WAV. Shared by the upload
/// path (source from the bundle) and the reprocess path (source is the stored
/// bulk_source WAV), so both cut slices identically.
async fn align_one_recording(
    recording_id: &str,
    script: &str,
    kind: &str,
    recorded_at: &str,
    duration_ms: u64,
    source: &Path,
    slug: &str,
    phrase: &str,
    external_id: Option<&str>,
    dest_root: &Path,
    db: &Mutex<Connection>,
    whisper_url: &str,
    capture: &db::CaptureMeta,
    summary: &mut AlignmentSummary,
) -> Result<(), AppError> {
    summary.recordings += 1;
    if recording_id.starts_with("test_") {
        // Test takes exist only to score a trained model. Transcribe them for
        // word timings but cut zero training slices, so they can never enter the
        // positive/negative/background pool. This branch also fires on reprocess,
        // since it keys off the persisted `test_` id prefix.
        return align_test_one(
            recording_id,
            script,
            recorded_at,
            duration_ms,
            source,
            slug,
            phrase,
            external_id,
            dest_root,
            db,
            whisper_url,
            capture,
            summary,
        )
        .await;
    }
    if kind.trim() == "enrollment" {
        // Enrollment reads are the F5 voice-cloning reference: store the whole
        // take, cut zero slices, skip Whisper. Keyed off the persisted kind so
        // reprocess re-enters here too.
        return store_enrollment_whole(
            recording_id,
            script,
            recorded_at,
            duration_ms,
            source,
            slug,
            phrase,
            external_id,
            dest_root,
            db,
            capture,
            summary,
        );
    }
    if script == BACKGROUND_SCRIPT_MARKER {
        // Background takes carry no speech to align; chop into fixed windows and
        // skip Whisper entirely. This branch also fires on reprocess, since the
        // marker is persisted in the recording's script column.
        return slice_background_recording(
            recording_id,
            recorded_at,
            duration_ms,
            source,
            slug,
            phrase,
            external_id,
            dest_root,
            db,
            capture,
            summary,
        );
    }
    if kind.trim() == "positive" && script == ENERGY_POSITIVE_SCRIPT_MARKER {
        // The project marked this wake word as a non-lexical sound, so slice this
        // positive take by burst energy and skip Whisper entirely — a single
        // burst Whisper happened to catch must not collapse the whole take to one
        // positive. Reprocess re-enters here because the marker is persisted.
        return slice_positive_by_energy(
            recording_id,
            script,
            recorded_at,
            duration_ms,
            source,
            slug,
            phrase,
            external_id,
            dest_root,
            db,
            "",
            whisper_url,
            capture,
            summary,
        );
    }
    let whisper = match transcribe_with_words(whisper_url, source, None).await {
        Ok(whisper) => whisper,
        Err(error) => {
            summary
                .warnings
                .push(format!("{}: {}", recording_id, error.message));
            return Ok(());
        }
    };
    let words = whisper_words(&whisper);
    if words.is_empty() {
        // Non-lexical positive take (e.g. a fast "beep beep") — Whisper heard no
        // words, so fall back to energy/VAD burst slicing for positives only.
        // Negatives and hard negatives with no words carry no useful training
        // data, so they keep the plain warning-and-skip behavior.
        if kind.trim() == "positive" {
            return slice_positive_by_energy(
                recording_id,
                script,
                recorded_at,
                duration_ms,
                source,
                slug,
                phrase,
                external_id,
                dest_root,
                db,
                whisper.text.trim(),
                whisper_url,
                capture,
                summary,
            );
        }
        summary.warnings.push(format!(
            "{}: Whisper returned no word timestamps",
            recording_id
        ));
        return Ok(());
    }

    let phrase_words = normalized_words(phrase);
    if phrase_words.is_empty() {
        summary.warnings.push(format!(
            "{}: wake phrase has no alignable words",
            recording_id
        ));
        return Ok(());
    }

    // Route by take kind. Each straight recording knows what it is, so the server
    // no longer has to guess positives vs negatives from one mixed script.
    //  - `positive`: the wake phrase said many times; every aligned occurrence is
    //    a positive, no near-miss frames to split out.
    //  - `negative` / `hard_negative`: ordinary speech or near-miss phrases with
    //    no true wake phrase; the whole read is chopped by the negative pass and
    //    filed under the matching label.
    //  - `mixed` (legacy, or older app builds that send no kind): the original
    //    behavior — split aligned occurrences into positives and the
    //    near-miss-framed hard negatives inferred from surrounding cue words.
    let take_kind = kind.trim();
    let expects_positive = !matches!(take_kind, "negative" | "hard_negative");
    let (positive_ranges, hard_negative_ranges): (Vec<_>, Vec<_>) = match take_kind {
        "positive" => (phrase_ranges(&words, &phrase_words), Vec::new()),
        "negative" | "hard_negative" => (Vec::new(), Vec::new()),
        _ => phrase_ranges(&words, &phrase_words)
            .into_iter()
            .partition(|(first, last)| !is_hard_negative_context(&words, *first, *last)),
    };
    // Whether to run the generic negative pass, and the label its slices carry.
    // A token take skips negatives entirely so silent gaps never become clips.
    let (run_negatives, negative_label): (bool, &str) = match take_kind {
        "positive" => (false, "negative"),
        "hard_negative" => (true, "hard_negative"),
        _ => (true, "negative"),
    };
    if expects_positive && positive_ranges.is_empty() {
        summary.warnings.push(format!(
            "{}: no aligned wake phrase occurrences found in transcript {:?}",
            recording_id,
            whisper.text.trim()
        ));
    }

    // Persist the raw source recording, then remove any stale slice files
    // this recording produced on a previous pass before writing the new set.
    let (source_wav, source_sha256) = match store_bulk_source(dest_root, recording_id, source) {
        Ok(stored) => stored,
        Err(error) => {
            summary
                .warnings
                .push(format!("{}: {}", recording_id, error.message));
            return Ok(());
        }
    };
    let old_paths = {
        let conn = db.lock().expect("db lock poisoned");
        db::active_slice_paths(&conn, recording_id).map_err(db_error)?
    };
    for old in old_paths {
        let old = PathBuf::from(old);
        if old.is_file() {
            let _ = fs::remove_file(old);
        }
    }

    let source_end = wav_duration_seconds(source)
        .unwrap_or_else(|_| words.last().map(|word| word.end).unwrap_or(0.0));

        let mut occupied = Vec::new();
        let mut slice_rows = Vec::new();
        for (first, last) in positive_ranges.iter() {
            let context_first = positive_context_first(&words, *first, *last);
            let (start, end) = padded_bounds(
                &words,
                context_first,
                *last,
                source_end,
                CUT_LEAD_PADDING_SECONDS,
                POSITIVE_TAIL_PADDING_SECONDS,
                false,
            );
            let (start, end) = clamp_slice_span(start, end, true);
            let (visible_first, visible_last) =
                visible_range(&words, context_first, *last, start, end);
            let slice_words = &words[visible_first..=visible_last];
            let clip_id = bulk_clip_hash_id(
                recording_id,
                recorded_at,
                duration_ms,
                "positive",
                start,
                end,
                slice_words,
            );
            let file_name = format!("{}_{}.wav", safe_filename(&clip_id), safe_filename(phrase));
            let dest = dest_root.join("positive").join(&file_name);
            if write_wav_slice(source, &dest, start, end)? {
                // Verify the cut audio actually contains the wake phrase, judged
                // on its own with no script prompt. Whisper word timings are
                // unstable, so a phrase the alignment placed here may not really
                // be in the slice; drop it rather than poison training.
                let heard = transcribe_with_words(whisper_url, &dest, None)
                    .await
                    .ok()
                    .map(|response| transcript_tail_has_phrase(&response.text, &phrase_words));
                match heard {
                    Some(false) => {
                        let _ = fs::remove_file(&dest);
                        summary.dropped_positives += 1;
                        summary.warnings.push(format!(
                            "{}: dropped positive at {:.2}-{:.2}s; wake phrase not heard in slice",
                            recording_id, start, end
                        ));
                    }
                    // Kept when the phrase is heard, or when verification itself
                    // failed (a transient Whisper error should not discard data).
                    _ => {
                        slice_rows.push(build_slice_row(
                            &clip_id,
                            "positive",
                            "positive",
                            &dest,
                            &file_name,
                            start,
                            end,
                            slice_words,
                        ));
                        summary.positives += 1;
                    }
                }
            }
            occupied.push((start, end));
        }

        // Hard negatives: the wake phrase captured in a near-miss frame. Cut with
        // the same generous bounds as a positive so the whole phrase is present,
        // but file it under the negative category with a distinct label, and mark
        // it occupied so the generic negative pass does not re-cut the same words.
        for (first, last) in hard_negative_ranges.iter() {
            let context_first = positive_context_first(&words, *first, *last);
            let (start, end) = padded_bounds(
                &words,
                context_first,
                *last,
                source_end,
                CUT_LEAD_PADDING_SECONDS,
                POSITIVE_TAIL_PADDING_SECONDS,
                false,
            );
            let (start, end) = clamp_slice_span(start, end, true);
            let (visible_first, visible_last) =
                visible_range(&words, context_first, *last, start, end);
            let slice_words = &words[visible_first..=visible_last];
            let clip_id = bulk_clip_hash_id(
                recording_id,
                recorded_at,
                duration_ms,
                "hard_negative",
                start,
                end,
                slice_words,
            );
            let phrase_text = slice_words
                .iter()
                .map(|word| word.word.trim())
                .collect::<Vec<_>>()
                .join(" ");
            let file_name = format!(
                "{}_{}.wav",
                safe_filename(&clip_id),
                safe_filename(&phrase_text)
            );
            // Hard negatives live in their own on-disk category so the pooling
            // assembler keeps them scoped to this project instead of borrowing
            // them into every other wake word's negative pool.
            let dest = dest_root.join("hard_negative").join(&file_name);
            if write_wav_slice(source, &dest, start, end)? {
                slice_rows.push(build_slice_row(
                    &clip_id,
                    "hard_negative",
                    "hard_negative",
                    &dest,
                    &file_name,
                    start,
                    end,
                    slice_words,
                ));
                summary.hard_negatives += 1;
            }
            occupied.push((start, end));
        }

        // Generic negative pass: every stretch of speech not already claimed by a
        // positive or hard-negative cut becomes a negative clip. For a `negative`
        // or `hard_negative` take `occupied` is empty, so this chops the whole
        // read; `negative_label` files it under the matching label. Skipped for a
        // token take so its silent gaps never turn into stray clips.
        if run_negatives {
            let negative_ranges = negative_ranges(&words, &occupied);
            for (_, _, word_start, word_end) in negative_ranges.iter() {
                let (start, end) = padded_bounds(
                    &words,
                    *word_start,
                    *word_end,
                    source_end,
                    CUT_LEAD_PADDING_SECONDS,
                    NEGATIVE_TAIL_PADDING_SECONDS,
                    true,
                );
                let (start, end) = clamp_slice_span(start, end, false);
                let (visible_first, visible_last) =
                    visible_range(&words, *word_start, *word_end, start, end);
                let slice_words = &words[visible_first..=visible_last];
                let clip_id = bulk_clip_hash_id(
                    recording_id,
                    recorded_at,
                    duration_ms,
                    negative_label,
                    start,
                    end,
                    slice_words,
                );
                let phrase_text = slice_words
                    .iter()
                    .map(|word| word.word.trim())
                    .collect::<Vec<_>>()
                    .join(" ");
                let file_name = format!(
                    "{}_{}.wav",
                    safe_filename(&clip_id),
                    safe_filename(&phrase_text)
                );
                // A hard-negative take files its whole read under the hard_negative
                // category (project-scoped); everything else is a pooled negative.
                let neg_category = if negative_label == "hard_negative" {
                    "hard_negative"
                } else {
                    "negative"
                };
                let dest = dest_root.join(neg_category).join(&file_name);
                if write_wav_slice(source, &dest, start, end)? {
                    slice_rows.push(build_slice_row(
                        &clip_id,
                        negative_label,
                        neg_category,
                        &dest,
                        &file_name,
                        start,
                        end,
                        slice_words,
                    ));
                    if negative_label == "hard_negative" {
                        summary.hard_negatives += 1;
                    } else {
                        summary.negatives += 1;
                    }
                }
            }
        }

        let word_rows = words
            .iter()
            .map(|word| db::WordRow {
                word: word.word.trim().to_string(),
                start_ms: (word.start * 1000.0).round() as i64,
                end_ms: (word.end * 1000.0).round() as i64,
                probability: word.probability,
            })
            .collect();
        let prompts = script
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(ToString::to_string)
            .collect();

        let alignment = db::RecordingAlignment {
            recording_id: recording_id.to_string(),
            project_slug: slug.to_string(),
            phrase: phrase.to_string(),
            external_id: external_id.map(ToString::to_string),
            script: script.to_string(),
            // Persist the resolved kind so reprocess slices this take the same way
            // without a manifest; empty (older app) normalizes to legacy mixed.
            kind: if take_kind.is_empty() { "mixed".to_string() } else { take_kind.to_string() },
            recorded_at: recorded_at.to_string(),
            duration_ms: duration_ms as i64,
            source_wav: source_wav.to_string_lossy().to_string(),
            source_sha256: Some(source_sha256),
            bundle: None,
            transcript_text: whisper.text.trim().to_string(),
            whisper_url: Some(whisper_url.to_string()),
            words: word_rows,
            prompts,
            slices: slice_rows,
            capture: capture.clone(),
        };
        {
            let mut conn = db.lock().expect("db lock poisoned");
            db::store_recording_alignment(&mut conn, &alignment, now_ms()).map_err(db_error)?;
        }

    Ok(())
}

/// Fixed-length background windows covering `[0, total)` seconds. A trailing
/// remnant shorter than the minimum is dropped rather than kept as a stub clip
/// the trainer would pad. Pure so the chunking is unit-testable without audio IO.
fn background_chunk_bounds(total: f64) -> Vec<(f64, f64)> {
    let mut bounds = Vec::new();
    let mut index = 0usize;
    loop {
        let start = index as f64 * BACKGROUND_CHUNK_SECONDS;
        if start >= total {
            break;
        }
        let end = (start + BACKGROUND_CHUNK_SECONDS).min(total);
        if end - start < BACKGROUND_MIN_CHUNK_SECONDS {
            break;
        }
        bounds.push((start, end));
        index += 1;
    }
    bounds
}

/// Slice one ambient background take into fixed-length background clips without
/// transcription. Shared by the upload path and reprocess (which re-enters via
/// `align_one_recording` on the background script marker), so both chunk a stored
/// source WAV identically. Persists a recording row carrying the marker script so
/// later reprocesses keep treating it as background.
fn slice_background_recording(
    recording_id: &str,
    recorded_at: &str,
    duration_ms: u64,
    source: &Path,
    slug: &str,
    phrase: &str,
    external_id: Option<&str>,
    dest_root: &Path,
    db: &Mutex<Connection>,
    capture: &db::CaptureMeta,
    summary: &mut AlignmentSummary,
) -> Result<(), AppError> {
    let (source_wav, source_sha256) = match store_bulk_source(dest_root, recording_id, source) {
        Ok(stored) => stored,
        Err(error) => {
            summary
                .warnings
                .push(format!("{}: {}", recording_id, error.message));
            return Ok(());
        }
    };
    // Remove any slice files a previous pass produced before writing the new set.
    let old_paths = {
        let conn = db.lock().expect("db lock poisoned");
        db::active_slice_paths(&conn, recording_id).map_err(db_error)?
    };
    for old in old_paths {
        let old = PathBuf::from(old);
        if old.is_file() {
            let _ = fs::remove_file(old);
        }
    }

    let total = wav_duration_seconds(source).unwrap_or(0.0);
    if total < BACKGROUND_MIN_CHUNK_SECONDS {
        summary.warnings.push(format!(
            "{}: background recording too short to slice ({:.2}s)",
            recording_id, total
        ));
    }

    let mut slice_rows = Vec::new();
    for (start, end) in background_chunk_bounds(total) {
        let clip_id = bulk_clip_hash_id(
            recording_id,
            recorded_at,
            duration_ms,
            "background",
            start,
            end,
            &[],
        );
        let file_name = format!("{}_background.wav", safe_filename(&clip_id));
        let dest = dest_root.join("background").join(&file_name);
        if write_wav_slice(source, &dest, start, end)? {
            slice_rows.push(build_slice_row(
                &clip_id,
                "background",
                "background",
                &dest,
                &file_name,
                start,
                end,
                &[],
            ));
            summary.background += 1;
        }
    }

    let alignment = db::RecordingAlignment {
        recording_id: recording_id.to_string(),
        project_slug: slug.to_string(),
        phrase: phrase.to_string(),
        external_id: external_id.map(ToString::to_string),
        script: BACKGROUND_SCRIPT_MARKER.to_string(),
        kind: "background".to_string(),
        recorded_at: recorded_at.to_string(),
        duration_ms: duration_ms as i64,
        source_wav: source_wav.to_string_lossy().to_string(),
        source_sha256: Some(source_sha256),
        bundle: None,
        transcript_text: String::new(),
        whisper_url: None,
        words: Vec::new(),
        prompts: Vec::new(),
        slices: slice_rows,
        capture: capture.clone(),
    };
    {
        let mut conn = db.lock().expect("db lock poisoned");
        db::store_recording_alignment(&mut conn, &alignment, now_ms()).map_err(db_error)?;
    }

    Ok(())
}

/// Segment a mono PCM take into sound-burst windows by short-frame RMS energy.
/// Pure (no audio IO) so the burst detection is unit-testable. Returns padded,
/// clamped `(start, end)` spans in seconds, in order. Used as the positive-take
/// fallback when Whisper finds no words: each repeated sound burst (e.g. one
/// fast "beep beep") becomes one positive clip.
fn energy_burst_bounds(samples: &[i16], sample_rate: f64) -> Vec<(f64, f64)> {
    if sample_rate <= 0.0 || samples.is_empty() {
        return Vec::new();
    }
    let frame_len = ((ENERGY_FRAME_SECONDS * sample_rate).round() as usize).max(1);
    // Short-frame RMS envelope. Frame `i` covers `[i*frame_len, ..)` samples.
    let mut rms: Vec<f64> = Vec::new();
    let mut i = 0;
    while i < samples.len() {
        let end = (i + frame_len).min(samples.len());
        let mut sum = 0.0f64;
        for &s in &samples[i..end] {
            let v = s as f64;
            sum += v * v;
        }
        rms.push((sum / (end - i) as f64).sqrt());
        i += frame_len;
    }
    if rms.is_empty() {
        return Vec::new();
    }
    // Noise floor as a low percentile of frame energy; peak as the loudest frame.
    let mut sorted = rms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let floor = sorted[(sorted.len() as f64 * 0.10) as usize];
    let peak = *sorted.last().unwrap();
    // A flat take (no burst rises meaningfully above its own floor) yields nothing.
    if peak - floor < 1.0 {
        return Vec::new();
    }
    let open = floor + ENERGY_OPEN_FRACTION * (peak - floor);
    let close = floor + ENERGY_CLOSE_FRACTION * (peak - floor);
    let frame_secs = frame_len as f64 / sample_rate;

    // Hysteresis walk: open a burst above `open`, close it below `close`.
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut voiced = false;
    let mut run_start = 0usize;
    for (idx, &energy) in rms.iter().enumerate() {
        if voiced {
            if energy < close {
                runs.push((run_start, idx));
                voiced = false;
            }
        } else if energy >= open {
            voiced = true;
            run_start = idx;
        }
    }
    if voiced {
        runs.push((run_start, rms.len()));
    }

    // Merge runs whose silent gap is short enough to be inside one utterance,
    // then drop anything too short to be a real burst.
    let total = samples.len() as f64 / sample_rate;
    let mut bursts: Vec<(f64, f64)> = Vec::new();
    for (start_frame, end_frame) in runs {
        let start = start_frame as f64 * frame_secs;
        let end = (end_frame as f64 * frame_secs).min(total);
        match bursts.last_mut() {
            Some(last) if start - last.1 < ENERGY_MERGE_GAP_SECONDS => last.1 = end,
            _ => bursts.push((start, end)),
        }
    }
    bursts
        .into_iter()
        .filter(|(start, end)| end - start >= ENERGY_MIN_BURST_SECONDS)
        .map(|(start, end)| {
            (
                (start - ENERGY_LEAD_PADDING_SECONDS).max(0.0),
                (end + ENERGY_TAIL_PADDING_SECONDS).min(total),
            )
        })
        .collect()
}

/// Slice a non-lexical positive take into positive clips by energy bursts, used
/// when Whisper returned no words. Mirrors `slice_background_recording`: stores
/// the source WAV, clears stale slices, writes one positive per detected burst,
/// and persists the alignment (with the resolved `positive` kind so reprocess
/// re-enters here). No per-slice Whisper verification — there are no words to
/// confirm against, which is the whole reason this path exists.
#[allow(clippy::too_many_arguments)]
fn slice_positive_by_energy(
    recording_id: &str,
    script: &str,
    recorded_at: &str,
    duration_ms: u64,
    source: &Path,
    slug: &str,
    phrase: &str,
    external_id: Option<&str>,
    dest_root: &Path,
    db: &Mutex<Connection>,
    transcript_text: &str,
    whisper_url: &str,
    capture: &db::CaptureMeta,
    summary: &mut AlignmentSummary,
) -> Result<(), AppError> {
    let (source_wav, source_sha256) = match store_bulk_source(dest_root, recording_id, source) {
        Ok(stored) => stored,
        Err(error) => {
            summary
                .warnings
                .push(format!("{}: {}", recording_id, error.message));
            return Ok(());
        }
    };
    let old_paths = {
        let conn = db.lock().expect("db lock poisoned");
        db::active_slice_paths(&conn, recording_id).map_err(db_error)?
    };
    for old in old_paths {
        let old = PathBuf::from(old);
        if old.is_file() {
            let _ = fs::remove_file(old);
        }
    }

    // Read the whole take as mono 16-bit PCM to compute the energy envelope.
    let mut reader = WavReader::open(source)
        .map_err(|error| AppError::bad_request(format!("cannot read WAV for slicing: {error}")))?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_format != SampleFormat::Int || spec.bits_per_sample != 16 {
        return Err(AppError::bad_request(
            "energy slicing requires mono 16-bit PCM WAV",
        ));
    }
    let samples: Vec<i16> = reader
        .samples::<i16>()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| AppError::bad_request(format!("bad WAV sample while slicing: {error}")))?;
    let bursts = energy_burst_bounds(&samples, spec.sample_rate as f64);
    if bursts.is_empty() {
        summary.warnings.push(format!(
            "{}: no sound bursts found for energy-sliced positive take",
            recording_id
        ));
    }

    let mut slice_rows = Vec::new();
    for (start, end) in bursts {
        let (start, end) = clamp_slice_span(start, end, true);
        let clip_id = bulk_clip_hash_id(
            recording_id,
            recorded_at,
            duration_ms,
            "positive",
            start,
            end,
            &[],
        );
        let file_name = format!("{}_{}.wav", safe_filename(&clip_id), safe_filename(phrase));
        let dest = dest_root.join("positive").join(&file_name);
        if write_wav_slice(source, &dest, start, end)? {
            slice_rows.push(build_slice_row(
                &clip_id,
                "positive",
                "positive",
                &dest,
                &file_name,
                start,
                end,
                &[],
            ));
            summary.positives += 1;
        }
    }

    // The energy marker is an internal routing sentinel, not a read-aloud prompt,
    // so keep it out of the displayed prompt list — but leave it in the persisted
    // `script` below so reprocess re-enters the energy path.
    let prompts = script
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && *line != ENERGY_POSITIVE_SCRIPT_MARKER)
        .map(ToString::to_string)
        .collect();
    let alignment = db::RecordingAlignment {
        recording_id: recording_id.to_string(),
        project_slug: slug.to_string(),
        phrase: phrase.to_string(),
        external_id: external_id.map(ToString::to_string),
        script: script.to_string(),
        kind: "positive".to_string(),
        recorded_at: recorded_at.to_string(),
        duration_ms: duration_ms as i64,
        source_wav: source_wav.to_string_lossy().to_string(),
        source_sha256: Some(source_sha256),
        bundle: None,
        transcript_text: transcript_text.to_string(),
        whisper_url: Some(whisper_url.to_string()),
        words: Vec::new(),
        prompts,
        slices: slice_rows,
        capture: capture.clone(),
    };
    {
        let mut conn = db.lock().expect("db lock poisoned");
        db::store_recording_alignment(&mut conn, &alignment, now_ms()).map_err(db_error)?;
    }
    Ok(())
}

fn build_slice_row(
    clip_id: &str,
    label: &str,
    category: &str,
    dest: &Path,
    file_name: &str,
    start_sec: f64,
    end_sec: f64,
    words: &[WhisperWord],
) -> db::SliceRow {
    let spoken_phrase = words
        .iter()
        .map(|word| word.word.trim())
        .collect::<Vec<_>>()
        .join(" ");
    let avg_probability = if words.is_empty() {
        0.0
    } else {
        words.iter().map(|word| word.probability).sum::<f64>() / words.len() as f64
    };
    db::SliceRow {
        id: clip_id.to_string(),
        label: label.to_string(),
        category: category.to_string(),
        spoken_phrase,
        source_start_ms: (start_sec * 1000.0).round() as i64,
        source_end_ms: (end_sec * 1000.0).round() as i64,
        avg_probability,
        word_count: words.len() as i64,
        wav_path: dest.to_string_lossy().to_string(),
        file_name: file_name.to_string(),
    }
}

async fn transcribe_with_words(
    whisper_url: &str,
    wav_path: &Path,
    prompt: Option<&str>,
) -> Result<WhisperResponse, AppError> {
    let bytes = fs::read(wav_path)?;
    let file_name = wav_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("audio.wav")
        .to_string();
    let part = reqwest::multipart::Part::bytes(bytes)
        .file_name(file_name)
        .mime_str("audio/wav")
        .map_err(|error| AppError::internal(format!("prepare Whisper upload: {error}")))?;
    // The prompt is deliberately omitted for alignment and verification: on this
    // clean, scripted audio it does not improve accuracy and it biases Whisper
    // toward reporting the scripted words even where the audio differs.
    let mut form = reqwest::multipart::Form::new()
        .part("file", part)
        .text("response_format", "verbose_json")
        .text("temperature", "0.0")
        .text("no_context", "true")
        .text("word_timestamps", "true");
    if let Some(prompt) = prompt.filter(|value| !value.trim().is_empty()) {
        form = form.text("prompt", prompt.to_string());
    }
    let endpoint = format!("{}/inference", whisper_url.trim_end_matches('/'));
    let response = reqwest::Client::new()
        .post(endpoint)
        .multipart(form)
        .send()
        .await
        .map_err(|error| AppError::internal(format!("Whisper request failed: {error}")))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| AppError::internal(format!("Whisper response read failed: {error}")))?;
    if !status.is_success() {
        return Err(AppError::internal(format!(
            "Whisper returned {status}: {}",
            body.trim()
        )));
    }
    serde_json::from_str(&body)
        .map_err(|error| AppError::internal(format!("Whisper response JSON failed: {error}")))
}

fn whisper_words(response: &WhisperResponse) -> Vec<WhisperWord> {
    response
        .segments
        .iter()
        .flat_map(|segment| {
            let offset = word_timestamp_offset(segment);
            segment.words.iter().cloned().map(move |mut word| {
                if offset > 0.0 {
                    word.start += offset;
                    word.end += offset;
                }
                word
            })
        })
        .filter(|word| word.end >= word.start)
        .collect()
}

fn word_timestamp_offset(segment: &WhisperSegment) -> f64 {
    let Some(first_word) = segment.words.first() else {
        return 0.0;
    };
    let Some(last_word) = segment.words.last() else {
        return 0.0;
    };
    let start_delta = segment.start - first_word.start;
    let end_delta = segment.end - last_word.end;
    if start_delta > 0.5 && end_delta > 0.5 {
        start_delta
    } else {
        0.0
    }
}

fn phrase_ranges(words: &[WhisperWord], phrase_words: &[String]) -> Vec<(usize, usize)> {
    let normalized: Vec<String> = words
        .iter()
        .map(|word| normalize_word(&word.word))
        .collect();
    let mut ranges = Vec::new();
    for start in 0..normalized.len() {
        let end = start + phrase_words.len();
        if end <= normalized.len() && normalized[start..end] == *phrase_words {
            ranges.push((start, end - 1));
        }
    }
    ranges
}

/// Pick the earliest word to include as lead-in context for a tail-aligned
/// positive, filling up to POSITIVE_MAX_SECONDS of speech before the phrase end.
fn positive_context_first(words: &[WhisperWord], first: usize, last: usize) -> usize {
    let anchor_end = words[last].end;
    let earliest_start = (anchor_end - POSITIVE_MAX_SECONDS).max(0.0);
    let mut context_first = first;
    while context_first > 0 && words[context_first - 1].start >= earliest_start {
        context_first -= 1;
    }
    context_first
}

/// Expand a word-index range into padded second bounds. The start is nudged
/// earlier and the end later so onsets and (crucially) the wake-phrase tail are
/// not clipped.
///
/// `clamp_tail_to_neighbor` controls how far the trailing edge may grow. For
/// negatives it is `true`: the end must not run into the next word, so a
/// negative can never accidentally swallow an adjacent wake phrase. For
/// positives it is `false`: the wake phrase sits at the very end and Whisper
/// routinely places its end (and the following word's start) too early, so the
/// tail is allowed to grow past the neighbor up to the recording end — capturing
/// a positive that would otherwise lose its wake phrase matters more than a
/// little trailing audio.
fn padded_bounds(
    words: &[WhisperWord],
    first: usize,
    last: usize,
    source_end: f64,
    lead: f64,
    tail: f64,
    clamp_tail_to_neighbor: bool,
) -> (f64, f64) {
    let raw_start = words[first].start.max(0.0);
    let prev_end = if first > 0 {
        words[first - 1].end.max(0.0)
    } else {
        0.0
    };
    // Move the start earlier, but not into the previous word.
    let start_floor = prev_end.min(raw_start);
    let start = (raw_start - lead).clamp(start_floor, raw_start);

    let raw_end = words[last].end.max(raw_start);
    let neighbor_cap = if clamp_tail_to_neighbor && last + 1 < words.len() {
        words[last + 1].start
    } else {
        source_end
    };
    let end_cap = neighbor_cap.max(raw_end).min(source_end.max(raw_end));
    let end = (raw_end + tail).clamp(raw_end, end_cap);
    (start, end)
}

/// Enforce the hard maximum slice length on already-padded bounds. Positives
/// pass `keep_tail = true` so the wake phrase (which ends the clip) is never
/// trimmed — the start moves in instead; negatives keep their start and move
/// the end in.
fn clamp_slice_span(start: f64, end: f64, keep_tail: bool) -> (f64, f64) {
    if end - start <= MAX_SLICE_SECONDS {
        return (start, end);
    }
    if keep_tail {
        (end - MAX_SLICE_SECONDS, end)
    } else {
        (start, start + MAX_SLICE_SECONDS)
    }
}

/// The still-audible sub-range of `first..=last` after the bounds were clamped,
/// so the stored transcript/word timings match what the slice actually contains
/// instead of over-claiming words that got trimmed off.
fn visible_range(
    words: &[WhisperWord],
    first: usize,
    last: usize,
    start: f64,
    end: f64,
) -> (usize, usize) {
    let mut visible_first = first;
    while visible_first < last && words[visible_first].end <= start {
        visible_first += 1;
    }
    let mut visible_last = last;
    while visible_last > visible_first && words[visible_last].start >= end {
        visible_last -= 1;
    }
    (visible_first, visible_last)
}

fn wav_duration_seconds(path: &Path) -> Result<f64, AppError> {
    let reader = WavReader::open(path)
        .map_err(|error| AppError::bad_request(format!("cannot read WAV duration: {error}")))?;
    let spec = reader.spec();
    if spec.sample_rate == 0 {
        return Ok(0.0);
    }
    Ok(reader.duration() as f64 / spec.sample_rate as f64)
}

fn is_hard_negative_context(words: &[WhisperWord], first: usize, last: usize) -> bool {
    let context_start = first.saturating_sub(8);
    let context_end = (last + 5).min(words.len().saturating_sub(1));
    let context = words[context_start..=context_end]
        .iter()
        .map(|word| normalize_word(&word.word))
        .collect::<Vec<_>>();
    contains_word_sequence(&context, &["near", "match"])
        || contains_word_sequence(&context, &["hard", "negative"])
        || contains_word_sequence(&context, &["not", "the", "wake", "phrase"])
        || contains_word_sequence(&context, &["similar", "phrase"])
}

fn contains_word_sequence(words: &[String], phrase: &[&str]) -> bool {
    words
        .windows(phrase.len())
        .any(|window| window.iter().map(String::as_str).eq(phrase.iter().copied()))
}

fn negative_ranges(
    words: &[WhisperWord],
    occupied: &[(f64, f64)],
) -> Vec<(f64, f64, usize, usize)> {
    let mut ranges = Vec::new();
    let mut index = 0;
    while index < words.len() {
        if word_overlaps_ranges(&words[index], occupied) {
            index += 1;
            continue;
        }

        let start_word = index;
        if words[start_word].end - words[start_word].start > NEGATIVE_TARGET_SECONDS {
            index += 1;
            continue;
        }

        let mut end_word = start_word;
        while end_word + 1 < words.len()
            && !word_overlaps_ranges(&words[end_word + 1], occupied)
            && words[end_word + 1].end - words[start_word].start <= NEGATIVE_TARGET_SECONDS
        {
            end_word += 1;
        }

        ranges.push((
            words[start_word].start.max(0.0),
            words[end_word].end,
            start_word,
            end_word,
        ));
        index = end_word + 1;
    }
    ranges
}

fn word_overlaps_ranges(word: &WhisperWord, ranges: &[(f64, f64)]) -> bool {
    ranges
        .iter()
        .any(|(start, end)| word.start < *end && word.end > *start)
}

fn bulk_clip_hash_id(
    recording_id: &str,
    recorded_at: &str,
    recording_duration_ms: u64,
    category: &str,
    start_sec: f64,
    end_sec: f64,
    words: &[WhisperWord],
) -> String {
    let spoken_phrase = words
        .iter()
        .map(|word| word.word.trim())
        .collect::<Vec<_>>()
        .join(" ");
    let duration_ms = ((end_sec - start_sec).max(0.0) * 1000.0).round() as u64;
    let input = json!({
        "recording_id": recording_id,
        "recorded_at": recorded_at,
        "recording_duration_ms": recording_duration_ms,
        "category": category,
        "source_start_ms": (start_sec * 1000.0).round() as i64,
        "source_end_ms": (end_sec * 1000.0).round() as i64,
        "duration_ms": duration_ms,
        "spoken_phrase": spoken_phrase,
        "words": words.iter().map(|word| {
            json!({
                "word": word.word.trim(),
                "start_ms": (word.start * 1000.0).round() as i64,
                "end_ms": (word.end * 1000.0).round() as i64,
            })
        }).collect::<Vec<_>>(),
    });
    let mut hasher = Sha256::new();
    hasher.update(input.to_string().as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Persist a recording's source WAV under `bulk_source/` and return both the
/// stored path and its full-file SHA-256. The checksum lets a device ask which
/// takes the server already holds and skip re-uploading unchanged ones.
fn store_bulk_source(
    dest_root: &Path,
    recording_id: &str,
    source: &Path,
) -> Result<(PathBuf, String), AppError> {
    let dir = dest_root.join("bulk_source");
    fs::create_dir_all(&dir)?;
    let dest = dir.join(format!("{}.wav", safe_filename(recording_id)));
    // Reprocess passes the already-stored source as the input; copying a file
    // onto itself would truncate it to zero bytes, so skip when they match.
    let same = match (source.canonicalize(), dest.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    };
    if !same {
        fs::copy(source, &dest)?;
    }
    let sha256 = file_sha256(&dest)?;
    Ok((dest, sha256))
}

/// Streaming SHA-256 of a file's raw bytes, hex-encoded. Matches the digest the
/// Android client computes over the same WAV so the two can be compared.
fn file_sha256(path: &Path) -> Result<String, AppError> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn write_wav_slice(
    source: &Path,
    dest: &Path,
    start_sec: f64,
    end_sec: f64,
) -> Result<bool, AppError> {
    let mut reader = WavReader::open(source)
        .map_err(|error| AppError::bad_request(format!("cannot read WAV for slicing: {error}")))?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_format != SampleFormat::Int || spec.bits_per_sample != 16 {
        return Err(AppError::bad_request(
            "bulk slicing requires mono 16-bit PCM WAV",
        ));
    }
    let sample_rate = spec.sample_rate as f64;
    let start_sample = (start_sec * sample_rate).floor().max(0.0) as usize;
    let end_sample = (end_sec * sample_rate).ceil().max(start_sample as f64) as usize;
    if end_sample <= start_sample {
        return Ok(false);
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    let samples: Vec<i16> = reader
        .samples::<i16>()
        .skip(start_sample)
        .take(end_sample - start_sample)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| AppError::bad_request(format!("bad WAV sample while slicing: {error}")))?;
    if samples.is_empty() {
        return Ok(false);
    }
    let mut writer = WavWriter::create(
        dest,
        WavSpec {
            channels: 1,
            sample_rate: spec.sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        },
    )
    .map_err(|error| AppError::internal(format!("create WAV slice: {error}")))?;
    for sample in samples {
        writer
            .write_sample(sample)
            .map_err(|error| AppError::internal(format!("write WAV slice: {error}")))?;
    }
    writer
        .finalize()
        .map_err(|error| AppError::internal(format!("finish WAV slice: {error}")))?;
    Ok(true)
}

fn alignment_summary(summary: &AlignmentSummary) -> String {
    if summary.recordings == 0 {
        if !summary.warnings.is_empty() {
            let mut output = "No bulk recordings were aligned".to_string();
            output.push_str("\nWarnings:");
            for warning in &summary.warnings {
                output.push_str("\n- ");
                output.push_str(warning);
            }
            return output;
        }
        return "No bulk recordings".to_string();
    }
    let mut output = format!(
        "Aligned {} recordings into {} positives and {} negatives",
        summary.recordings, summary.positives, summary.negatives
    );
    if summary.hard_negatives > 0 {
        output.push_str(&format!(
            " (incl. {} hard negatives)",
            summary.hard_negatives
        ));
    }
    if summary.background > 0 {
        output.push_str(&format!(", plus {} background clips", summary.background));
    }
    if summary.dropped_positives > 0 {
        output.push_str(&format!(
            " ({} positives dropped: wake phrase not heard)",
            summary.dropped_positives
        ));
    }
    if !summary.warnings.is_empty() {
        output.push_str("\nWarnings:");
        for warning in &summary.warnings {
            output.push_str("\n- ");
            output.push_str(warning);
        }
    }
    output
}

fn review_clip_path(
    data_root: &Path,
    slug: &str,
    category: &str,
    file_name: &str,
) -> Result<PathBuf, AppError> {
    if !is_safe_slug(slug) {
        return Err(AppError::bad_request(format!(
            "unsafe wake word slug: {slug}"
        )));
    }
    if !matches!(category, "positive" | "negative" | "hard_negative" | "background") {
        return Err(AppError::bad_request(format!(
            "unsafe category: {category}"
        )));
    }
    if file_name.contains('/')
        || file_name.contains('\\')
        || file_name.contains("..")
        || !file_name.ends_with(".wav")
    {
        return Err(AppError::bad_request(format!(
            "unsafe file name: {file_name}"
        )));
    }
    Ok(data_root.join(slug).join(category).join(file_name))
}

fn normalized_words(value: &str) -> Vec<String> {
    value
        .split_whitespace()
        .map(normalize_word)
        .filter(|word| !word.is_empty())
        .collect()
}

/// True when the wake phrase appears at (or very near) the END of the normalized
/// transcript. Positives are tail-aligned, so the wake phrase must be the last
/// thing spoken; `TAIL_SLACK` words of trailing audio are tolerated for the tail
/// padding. Requiring the phrase at the tail — not just anywhere — rejects
/// slices that were cut too early and only contain the lead-in (e.g. "the next
/// words are ...") even when a flaky Whisper pass imagines the phrase mid-slice.
fn transcript_tail_has_phrase(text: &str, phrase_words: &[String]) -> bool {
    const TAIL_SLACK: usize = 2;
    if phrase_words.is_empty() {
        return false;
    }
    let heard = normalized_words(text);
    if heard.len() < phrase_words.len() {
        return false;
    }
    let last_start = heard.len() - phrase_words.len();
    for start in 0..=last_start {
        if &heard[start..start + phrase_words.len()] == phrase_words {
            let words_after = heard.len() - (start + phrase_words.len());
            if words_after <= TAIL_SLACK {
                return true;
            }
        }
    }
    false
}

fn normalize_word(value: &str) -> String {
    value
        .trim()
        .trim_matches(|ch: char| !ch.is_ascii_alphanumeric())
        .to_ascii_lowercase()
}

fn safe_join(root: &Path, relative: &str) -> Result<PathBuf, AppError> {
    let path = Path::new(relative);
    if path.is_absolute() {
        return Err(AppError::bad_request(format!(
            "path escapes bundle: {relative}"
        )));
    }
    for component in path.components() {
        if matches!(component, Component::ParentDir | Component::Prefix(_)) {
            return Err(AppError::bad_request(format!(
                "path escapes bundle: {relative}"
            )));
        }
    }
    Ok(root.join(path))
}

fn category_for_label(label: &str) -> Option<&'static str> {
    match label {
        "positive" | "false_negative" => Some("positive"),
        "negative" | "false_positive" => Some("negative"),
        "hard_negative" => Some("hard_negative"),
        "background" => Some("background"),
        _ => None,
    }
}

fn is_safe_slug(slug: &str) -> bool {
    let mut chars = slug.chars();
    match chars.next() {
        Some(ch) if ch.is_ascii_lowercase() || ch.is_ascii_digit() => {}
        _ => return false,
    }
    chars.all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
}

fn is_safe_recording_id(recording_id: &str) -> bool {
    !recording_id.is_empty()
        && recording_id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-'))
}

fn safe_filename(value: &str) -> String {
    let mut output = String::new();
    let mut last_dash = false;
    for ch in value.trim().to_ascii_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            output.push(ch);
            last_dash = false;
        } else if !last_dash {
            output.push('-');
            last_dash = true;
        }
    }
    let cleaned = output.trim_matches('-').to_string();
    if cleaned.is_empty() {
        "clip".to_string()
    } else {
        cleaned
    }
}

fn validation_summary(warnings: &[String], clip_count: usize) -> String {
    if warnings.is_empty() {
        format!("Validated {clip_count} clips")
    } else {
        let mut output = String::from("Warnings:");
        for warning in warnings {
            output.push_str("\n- ");
            output.push_str(warning);
        }
        output.push_str(&format!("\nValidated {clip_count} clips"));
        output
    }
}

/// Hyperparameters for a full training run. All optional; each falls back to the
/// same default as scripts/generate_config.py when omitted.
#[derive(Debug, Deserialize)]
struct TrainRequest {
    steps: Option<u32>,
    model_size: Option<String>,
    target_fp_per_hour: Option<f64>,
    n_samples: Option<u32>,
    n_samples_val: Option<u32>,
    personal: Option<bool>,
    positive_boost: Option<u32>,
    // How many F5 voice-cloned positives to generate (in the sync-server, before
    // the trainer launches) and fold into training. 0/omitted = no F5 clips.
    f5_count: Option<u32>,
    positive_per_batch: Option<u32>,
    real_samples_dir: Option<String>,
    // "start" | "end". Selects the leading-context recipe (see
    // generate_config.py --token-type). Defaults to "end" when omitted.
    token_type: Option<String>,
    // Realistic-positive compositing knobs (augment_realistic.py). All optional;
    // omitted values fall back to the trainer's sidecar defaults.
    realistic: Option<bool>,
    lead_probability: Option<f64>,
    real_lead_fraction: Option<f64>,
    synthetic_lead: Option<bool>,
    max_lead_ms: Option<u32>,
    lead_gap_min_ms: Option<u32>,
    lead_gap_max_ms: Option<u32>,
    margin_min_ms: Option<u32>,
    margin_max_ms: Option<u32>,
    snr_min_db: Option<f64>,
    snr_max_db: Option<f64>,
    background_augment: Option<bool>,
    voice_peak: Option<f64>,
}

fn training_container_name(slug: &str) -> String {
    format!("lkww-train-{slug}")
}

/// Append a `-e KEY=VALUE` pair to a docker argv.
fn push_env(args: &mut Vec<String>, key: &str, value: String) {
    args.push("-e".into());
    args.push(format!("{key}={value}"));
}

/// Run `docker` with the given args off the async executor, returning its output.
async fn run_docker(args: Vec<String>) -> Result<std::process::Output, AppError> {
    tokio::task::spawn_blocking(move || std::process::Command::new("docker").args(&args).output())
        .await
        .map_err(|e| AppError::internal(format!("docker task join failed: {e}")))?
        .map_err(|e| AppError::internal(format!("failed to run docker: {e}")))
}

/// Is a container by this name currently running? Uses `docker inspect`; a
/// missing container (the `--rm` run already exited) reports false.
async fn container_running(name: &str) -> bool {
    let name = name.to_string();
    let out = run_docker(vec![
        "inspect".into(),
        "-f".into(),
        "{{.State.Running}}".into(),
        name,
    ])
    .await;
    matches!(out, Ok(o) if o.status.success()
        && String::from_utf8_lossy(&o.stdout).trim() == "true")
}

/// Allowlisted trainer model sizes. generate_config.py writes this straight into
/// the YAML, so reject anything unexpected up front.
fn valid_model_size(size: &str) -> bool {
    matches!(size, "tiny" | "small" | "medium" | "large")
}

/// A real-samples dir is a plain relative path passed to the assembler/config.
fn valid_real_samples_dir(dir: &str) -> bool {
    !dir.is_empty()
        && dir.len() <= 128
        && dir
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '/' | '_' | '-'))
        && !dir.contains("..")
}

/// Fully-resolved, validated training hyperparameters. This is what gets stored
/// in the queue (as JSON) and rebuilt into the trainer container's argv at
/// dispatch, so a job launched minutes after it was enqueued still runs with the
/// exact settings the user chose.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ValidatedTrain {
    steps: u32,
    model_size: String,
    target_fp_per_hour: f64,
    personal: bool,
    positive_boost: u32,
    f5_count: u32,
    n_samples: Option<u32>,
    n_samples_val: Option<u32>,
    positive_per_batch: Option<u32>,
    real_samples_dir: String,
    token_type: String,
    // Realistic-positive compositing (see augment_realistic.py). Fully resolved
    // and clamped so the trainer container can render the sidecar verbatim.
    realistic: bool,
    lead_probability: f64,
    real_lead_fraction: f64,
    synthetic_lead: bool,
    max_lead_ms: u32,
    lead_gap_min_ms: u32,
    lead_gap_max_ms: u32,
    margin_min_ms: u32,
    margin_max_ms: u32,
    snr_min_db: f64,
    snr_max_db: f64,
    background_augment: bool,
    voice_peak: f64,
}

/// Validate + default a raw train request into a `ValidatedTrain`, rejecting bad
/// input at enqueue time so the queue only ever holds runnable jobs.
fn validate_train(request: TrainRequest) -> Result<ValidatedTrain, AppError> {
    let steps = request.steps.unwrap_or(50_000).clamp(100, 500_000);
    let model_size = request.model_size.unwrap_or_else(|| "medium".to_string());
    if !valid_model_size(&model_size) {
        return Err(AppError::bad_request(format!(
            "invalid model_size: {model_size}"
        )));
    }
    let target_fp_per_hour = request.target_fp_per_hour.unwrap_or(0.2);
    if !(target_fp_per_hour.is_finite() && (0.0..=100.0).contains(&target_fp_per_hour)) {
        return Err(AppError::bad_request("invalid target_fp_per_hour"));
    }
    let real_samples_dir = request
        .real_samples_dir
        .unwrap_or_else(|| "./data/train".to_string());
    if !valid_real_samples_dir(&real_samples_dir) {
        return Err(AppError::bad_request("invalid real_samples_dir"));
    }
    let token_type = request.token_type.unwrap_or_else(|| "end".to_string());
    if !matches!(token_type.as_str(), "start" | "end") {
        return Err(AppError::bad_request(
            "invalid token_type (expected start|end)",
        ));
    }

    // --- realistic-positive compositing knobs --------------------------------
    // Clamp fractions to [0,1], millisecond spans to a sane window, and dB to a
    // reasonable range; keep each range's min <= max so the trainer's randint /
    // uniform never sees an inverted interval.
    let clamp01 = |v: f64| v.clamp(0.0, 1.0);
    let realistic = request.realistic.unwrap_or(true);
    let lead_probability = clamp01(request.lead_probability.unwrap_or(0.75));
    let real_lead_fraction = clamp01(request.real_lead_fraction.unwrap_or(0.6));
    let voice_peak = request.voice_peak.unwrap_or(0.7).clamp(0.05, 1.0);
    let max_lead_ms = request.max_lead_ms.unwrap_or(900).min(2000);
    let mut lead_gap_min_ms = request.lead_gap_min_ms.unwrap_or(40).min(2000);
    let mut lead_gap_max_ms = request.lead_gap_max_ms.unwrap_or(300).min(2000);
    if lead_gap_min_ms > lead_gap_max_ms {
        std::mem::swap(&mut lead_gap_min_ms, &mut lead_gap_max_ms);
    }
    let mut margin_min_ms = request.margin_min_ms.unwrap_or(100).min(2000);
    let mut margin_max_ms = request.margin_max_ms.unwrap_or(700).min(2000);
    if margin_min_ms > margin_max_ms {
        std::mem::swap(&mut margin_min_ms, &mut margin_max_ms);
    }
    let mut snr_min_db = request.snr_min_db.unwrap_or(0.0).clamp(-10.0, 60.0);
    let mut snr_max_db = request.snr_max_db.unwrap_or(18.0).clamp(-10.0, 60.0);
    if !(snr_min_db.is_finite() && snr_max_db.is_finite()) {
        return Err(AppError::bad_request("invalid snr range"));
    }
    if snr_min_db > snr_max_db {
        std::mem::swap(&mut snr_min_db, &mut snr_max_db);
    }

    Ok(ValidatedTrain {
        steps,
        model_size,
        target_fp_per_hour,
        personal: request.personal.unwrap_or(false),
        positive_boost: request.positive_boost.unwrap_or(1).clamp(1, 50),
        f5_count: request.f5_count.unwrap_or(0).min(2000),
        n_samples: request.n_samples,
        n_samples_val: request.n_samples_val,
        positive_per_batch: request.positive_per_batch,
        real_samples_dir,
        token_type,
        realistic,
        lead_probability,
        real_lead_fraction,
        synthetic_lead: request.synthetic_lead.unwrap_or(true),
        max_lead_ms,
        lead_gap_min_ms,
        lead_gap_max_ms,
        margin_min_ms,
        margin_max_ms,
        snr_min_db,
        snr_max_db,
        background_augment: request.background_augment.unwrap_or(true),
        voice_peak,
    })
}

/// Enqueue a full training run. The training scheduler runs one trainer
/// container at a time and dispatches queued jobs oldest-first, so pressing
/// "start" several times (across one or many wake words) lines them up instead
/// of failing. No recorded clips are required: the trainer synthesizes positives
/// from the phrase, so a brand-new word can still train.
async fn start_training(
    State(state): State<AppState>,
    AxumPath(slug): AxumPath<String>,
    Json(request): Json<TrainRequest>,
) -> Result<Json<Value>, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request("invalid slug"));
    }
    if state.host_repo_root.is_none() {
        return Err(AppError::internal(
            "training disabled: HOST_REPO_ROOT not set on the sync-server",
        ));
    }

    // The wake word must exist so the dispatcher can supply its target phrase.
    let phrase = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::project_phrase(&conn, &slug).map_err(db_error)?
    };
    let Some((phrase, _external_id)) = phrase else {
        return Err(AppError {
            status: StatusCode::NOT_FOUND,
            message: format!("unknown wake word: {slug}"),
        });
    };

    let validated = validate_train(request)?;
    let params_json = serde_json::to_string(&validated)
        .map_err(|e| AppError::internal(format!("serialize train params: {e}")))?;

    let (id, position) = {
        let conn = state.db.lock().expect("db lock poisoned");
        let id = db::enqueue_training(&conn, &slug, &params_json, now_ms()).map_err(db_error)?;
        let position = db::queue_position(&conn, id).map_err(db_error)?.unwrap_or(0);
        (id, position)
    };

    // Nudge the scheduler so a job enqueued into an idle pipeline starts right
    // away instead of waiting for the next poll tick.
    let dispatch_state = state.clone();
    tokio::spawn(async move {
        dispatch_training(&dispatch_state).await;
    });

    Ok(Json(json!({
        "status": if position <= 1 { "started" } else { "queued" },
        "slug": slug,
        "phrase": phrase,
        "queue_id": id,
        "position": position,
        "steps": validated.steps,
        "model_size": validated.model_size,
        "personal": validated.personal,
        "token_type": validated.token_type,
    })))
}

/// Launch the trainer container for one validated job and return its container
/// name. Every hyperparameter travels as a `-e KEY=VALUE` pair (argv, never
/// shell-interpolated), so nothing here can inject shell.
async fn launch_train_container(
    state: &AppState,
    slug: &str,
    phrase: &str,
    vt: &ValidatedTrain,
) -> Result<String, AppError> {
    let Some(host_repo_root) = (*state.host_repo_root).clone() else {
        return Err(AppError::internal(
            "training disabled: HOST_REPO_ROOT not set on the sync-server",
        ));
    };
    let name = training_container_name(slug);
    // Clear any dead container left behind by a crash so the name is free.
    let _ = run_docker(vec!["rm".into(), "-f".into(), name.clone()]).await;

    let mut args: Vec<String> = vec![
        "run".into(),
        "-d".into(),
        "--rm".into(),
        "--name".into(),
        name.clone(),
    ];
    if *state.trainer_gpu {
        args.push("--gpus".into());
        args.push("all".into());
    }
    args.push("-v".into());
    args.push(format!("{host_repo_root}:/work"));
    // Mount the realistic-positive augment module over the installed one (same
    // mechanism as train_ctxfix.sh), so a training run composites positives as
    // one clear voice on a background bed without needing a trainer-image
    // rebuild. train_job.sh writes the sidecar this reads (AUG_REALISTIC_CONFIG).
    if vt.realistic {
        args.push("-v".into());
        args.push(format!(
            "{host_repo_root}/trainer/patches/augment_realistic.py:\
/opt/conda/lib/python3.11/site-packages/livekit/wakeword/data/augment.py:ro"
        ));
    }
    push_env(&mut args, "SLUG", slug.to_string());
    push_env(&mut args, "PHRASE", phrase.to_string());
    push_env(&mut args, "STEPS", vt.steps.to_string());
    push_env(&mut args, "MODEL_SIZE", vt.model_size.clone());
    push_env(&mut args, "TARGET_FP_PER_HOUR", vt.target_fp_per_hour.to_string());
    push_env(
        &mut args,
        "PERSONAL",
        if vt.personal { "1".into() } else { "0".into() },
    );
    push_env(&mut args, "POSITIVE_BOOST", vt.positive_boost.to_string());
    push_env(&mut args, "F5_COUNT", vt.f5_count.to_string());
    if let Some(n) = vt.n_samples {
        push_env(&mut args, "N_SAMPLES", n.to_string());
    }
    if let Some(n) = vt.n_samples_val {
        push_env(&mut args, "N_SAMPLES_VAL", n.to_string());
    }
    if let Some(n) = vt.positive_per_batch {
        push_env(&mut args, "POSITIVE_PER_BATCH", n.to_string());
    }
    push_env(&mut args, "REAL_SAMPLES_DIR", vt.real_samples_dir.clone());
    push_env(&mut args, "TOKEN_TYPE", vt.token_type.clone());
    // Realistic-positive compositing knobs -> train_job.sh sidecar.
    push_env(
        &mut args,
        "REALISTIC",
        if vt.realistic { "1".into() } else { "0".into() },
    );
    push_env(&mut args, "LEAD_PROBABILITY", vt.lead_probability.to_string());
    push_env(&mut args, "REAL_LEAD_FRACTION", vt.real_lead_fraction.to_string());
    push_env(
        &mut args,
        "SYNTHETIC_LEAD",
        if vt.synthetic_lead { "1".into() } else { "0".into() },
    );
    push_env(&mut args, "MAX_LEAD_MS", vt.max_lead_ms.to_string());
    push_env(&mut args, "LEAD_GAP_MIN_MS", vt.lead_gap_min_ms.to_string());
    push_env(&mut args, "LEAD_GAP_MAX_MS", vt.lead_gap_max_ms.to_string());
    push_env(&mut args, "MARGIN_MIN_MS", vt.margin_min_ms.to_string());
    push_env(&mut args, "MARGIN_MAX_MS", vt.margin_max_ms.to_string());
    push_env(&mut args, "SNR_MIN_DB", vt.snr_min_db.to_string());
    push_env(&mut args, "SNR_MAX_DB", vt.snr_max_db.to_string());
    push_env(
        &mut args,
        "BACKGROUND_AUGMENT",
        if vt.background_augment { "1".into() } else { "0".into() },
    );
    push_env(&mut args, "VOICE_PEAK", vt.voice_peak.to_string());
    // The job stamps its own image tag into the model manifest for provenance.
    push_env(&mut args, "TRAINER_IMAGE", (*state.trainer_image).clone());
    args.push((*state.trainer_image).clone());
    args.push("bash".into());
    args.push("-lc".into());
    args.push("bash /work/trainer/scripts/train_job.sh".into());

    // F5 voice-cloned positives are generated HERE, in the sync-server, before
    // the trainer launches: only this container can reach the speech-f5tts
    // service (the trainer container has no docker access). This tops the synth
    // bucket up to vt.f5_count; the trainer then pools them into positives. The
    // f5gen phase is written into train_status.json so the app's training screen
    // shows it. Non-fatal — training proceeds on whatever clips exist.
    ensure_f5_positives(state, slug, phrase, vt).await;

    let output = run_docker(args).await?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(AppError::internal(format!(
            "docker run failed: {}",
            stderr.trim()
        )));
    }
    Ok(name)
}

/// Render a queue row for the API, folding in its live position.
fn queue_entry_json(entry: &db::QueueEntry, position: Option<i64>) -> Value {
    let params: Value =
        serde_json::from_str(&entry.params_json).unwrap_or_else(|_| json!({}));
    json!({
        "id": entry.id,
        "slug": entry.slug,
        "state": entry.state,
        "position": position,
        "container_name": entry.container_name,
        "enqueued_at_ms": entry.enqueued_at_ms,
        "started_at_ms": entry.started_at_ms,
        "finished_at_ms": entry.finished_at_ms,
        "params": params,
    })
}

/// The live training pipeline: every queued or running job, oldest first.
async fn training_queue(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let entries = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::active_queue(&conn).map_err(db_error)?
    };
    // Position: running jobs are 0, queued jobs count 1..N in id order.
    let mut queued_rank = 0i64;
    let jobs: Vec<Value> = entries
        .iter()
        .map(|e| {
            let position = if e.state == "running" {
                Some(0)
            } else {
                queued_rank += 1;
                Some(queued_rank)
            };
            queue_entry_json(e, position)
        })
        .collect();
    Ok(Json(json!({ "status": "ok", "jobs": jobs })))
}

/// Cancel every queued/running job for a wake word: mark the rows canceled and
/// kill any live trainer container. The killed container's `--rm` cleanup plus
/// the status endpoint's reconciliation report it as stopped.
async fn cancel_training(
    State(state): State<AppState>,
    AxumPath(slug): AxumPath<String>,
) -> Result<Json<Value>, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request("invalid slug"));
    }
    let canceled = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::cancel_slug_jobs(&conn, &slug, now_ms()).map_err(db_error)?
    };
    // Kill any container that was actually running for these jobs.
    for entry in &canceled {
        if let Some(name) = &entry.container_name {
            let _ = run_docker(vec!["rm".into(), "-f".into(), name.clone()]).await;
        }
    }
    // Also kill the by-convention container name in case a run started without a
    // recorded container_name yet.
    let _ = run_docker(vec![
        "rm".into(),
        "-f".into(),
        training_container_name(&slug),
    ])
    .await;

    Ok(Json(json!({
        "status": "canceled",
        "slug": slug,
        "canceled": canceled.len(),
    })))
}

/// Cancel a single queue entry by id (a queued job, or the running one).
async fn delete_queue_entry(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<i64>,
) -> Result<Json<Value>, AppError> {
    let entry = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::cancel_queue_entry(&conn, id, now_ms()).map_err(db_error)?
    };
    let Some(entry) = entry else {
        return Err(AppError {
            status: StatusCode::NOT_FOUND,
            message: format!("no active queue entry {id}"),
        });
    };
    if let Some(name) = &entry.container_name {
        let _ = run_docker(vec!["rm".into(), "-f".into(), name.clone()]).await;
    } else if entry.state == "running" {
        let _ = run_docker(vec![
            "rm".into(),
            "-f".into(),
            training_container_name(&entry.slug),
        ])
        .await;
    }
    Ok(Json(json!({
        "status": "canceled",
        "id": id,
        "slug": entry.slug,
    })))
}

/// Advance the training pipeline once: reconcile any running job against its
/// live container, then launch the next queued job if the pipeline is idle.
/// Cheap and idempotent, so it is safe to call on a timer and after every
/// enqueue.
async fn dispatch_training(state: &AppState) {
    // One dispatch at a time: the enqueue kick and the periodic scheduler share
    // this, so a queued job is never launched by both at once.
    let _guard = state.dispatch_lock.lock().await;

    // 1. Reconcile: a job marked running whose container has exited is finished.
    let running = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::running_queue(&conn).unwrap_or_default()
    };
    let mut pipeline_busy = false;
    for entry in running {
        let name = entry
            .container_name
            .clone()
            .unwrap_or_else(|| training_container_name(&entry.slug));
        if container_running(&name).await {
            pipeline_busy = true;
        } else {
            // The trainer container is gone; the real outcome lives in the
            // per-slug train_status.json. Retire the queue row either way.
            let conn = state.db.lock().expect("db lock poisoned");
            let _ = db::finish_queue_entry(&conn, entry.id, "done", now_ms());
        }
    }
    if pipeline_busy {
        return;
    }

    // 2. Dispatch: launch the oldest queued job, if any.
    let next = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::next_queued(&conn).unwrap_or_default()
    };
    let Some(entry) = next else { return };

    // The project must still exist to supply its phrase; otherwise drop the job.
    let phrase = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::project_phrase(&conn, &entry.slug).ok().flatten()
    };
    let Some((phrase, _)) = phrase else {
        let conn = state.db.lock().expect("db lock poisoned");
        let _ = db::finish_queue_entry(&conn, entry.id, "failed", now_ms());
        return;
    };
    let Ok(vt) = serde_json::from_str::<ValidatedTrain>(&entry.params_json) else {
        let conn = state.db.lock().expect("db lock poisoned");
        let _ = db::finish_queue_entry(&conn, entry.id, "failed", now_ms());
        return;
    };

    match launch_train_container(state, &entry.slug, &phrase, &vt).await {
        Ok(name) => {
            let conn = state.db.lock().expect("db lock poisoned");
            let _ = db::mark_queue_running(&conn, entry.id, &name, now_ms());
        }
        Err(err) => {
            eprintln!("training dispatch failed for {}: {}", entry.slug, err.message);
            let conn = state.db.lock().expect("db lock poisoned");
            let _ = db::finish_queue_entry(&conn, entry.id, "failed", now_ms());
        }
    }
}

/// Background loop that keeps the training pipeline moving: reconcile finished
/// runs and dispatch the next queued job every few seconds.
async fn training_scheduler(state: AppState) {
    loop {
        dispatch_training(&state).await;
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    }
}

/// Read the training status JSON for a wake word, reconciled with whether the
/// trainer container is still alive.
/// The fixed trainer pipeline. `livekit-wakeword run` always executes these six
/// stages in order, each with a recognizable tqdm sub-progress bar in the log.
/// This is the template that lets us report per-step percentages.
const TRAIN_STEPS: [&str; 6] = [
    "Generate synthetic data",
    "Augment clips",
    "Extract features",
    "Train classifier",
    "Export to ONNX",
    "Evaluate model",
];

/// Rough share of wall-clock each stage takes, so overall percent tracks time
/// rather than step count. Training dominates. Sums to 100.
const TRAIN_STEP_WEIGHTS: [u32; 6] = [10, 12, 10, 53, 5, 10];

/// Where each stage does its heavy compute, surfaced so the app can tag each
/// step "C" (CPU) or "G" (GPU). Verified against the installed trainer source:
/// synthesis (VITS/VoxCPM via get_device()->cuda), training, and evaluation run
/// on the GPU. Feature extraction runs on CPU — the mel/embedding ONNX sessions
/// are pinned to CPUExecutionProvider. Clip augmentation (parallelized sox
/// effects) and ONNX export are CPU.
const TRAIN_STEP_DEVICES: [&str; 6] = ["gpu", "cpu", "cpu", "gpu", "cpu", "gpu"];

/// Read only the last `max_bytes` of a file — the trainer log grows to many MB,
/// and the recent tail is all we need to locate the current stage.
fn read_tail_bytes(path: &std::path::Path, max_bytes: u64) -> Option<String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = fs::File::open(path).ok()?;
    let len = f.metadata().ok()?.len();
    f.seek(SeekFrom::Start(len.saturating_sub(max_bytes))).ok()?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).ok()?;
    Some(String::from_utf8_lossy(&buf).into_owned())
}

/// Extract the last `NN%` token from a tqdm-style line, clamped to 0..=100.
fn last_percent(line: &str) -> Option<u8> {
    let bytes = line.as_bytes();
    let mut found = None;
    for i in 0..bytes.len() {
        if bytes[i] == b'%' {
            let mut j = i;
            while j > 0 && bytes[j - 1].is_ascii_digit() {
                j -= 1;
            }
            if j < i {
                if let Ok(n) = line[j..i].parse::<u16>() {
                    found = Some(n.min(100) as u8);
                }
            }
        }
    }
    found
}

/// Map a single log line to a 1-based pipeline step, via either the explicit
/// "Step N/6" banner or the tqdm description unique to each stage.
fn line_step(line: &str) -> Option<usize> {
    if let Some(pos) = line.find("Step ") {
        let rest = &line[pos + 5..];
        if let Some(slash) = rest.find("/6") {
            if let Ok(n) = rest[..slash].trim().parse::<usize>() {
                if (1..=6).contains(&n) {
                    return Some(n);
                }
            }
        }
    }
    if line.contains("Synthesizing clips")
        || line.contains("VoxCPM clips")
        || line.contains("Background (")
    {
        return Some(1);
    }
    if line.contains("Augmenting ") {
        return Some(2);
    }
    if line.contains("Features ") {
        return Some(3);
    }
    if line.contains("Phase ") {
        return Some(4);
    }
    None
}

/// The tqdm description prefix of a progress line, e.g. "Augmenting positive r0".
fn tqdm_desc(line: &str) -> String {
    if let Some(bar) = line.find("%|") {
        let head = line[..bar]
            .trim_end()
            .trim_end_matches(|c: char| c.is_ascii_digit())
            .trim_end();
        if let Some(colon) = head.rfind(':') {
            return head[..colon].trim().to_string();
        }
        return head.trim().to_string();
    }
    String::new()
}

/// Parse the trainer log tail into a templated, per-step progress view.
fn parse_train_progress(text: &str, run_state: &str) -> Value {
    // Split on both '\r' (tqdm redraws in place) and '\n'.
    let lines: Vec<&str> = text.split(|c| c == '\n' || c == '\r').collect();

    let mut cur_step = 0usize; // 0 = pre-pipeline (assemble / config / setup)
    let mut cur_percent = 0u8;
    let mut active_label = String::new();
    for &line in &lines {
        if let Some(step) = line_step(line) {
            if step > cur_step {
                cur_step = step;
                cur_percent = last_percent(line).unwrap_or(0);
                active_label = tqdm_desc(line);
            } else if step == cur_step {
                if let Some(p) = last_percent(line) {
                    cur_percent = p;
                }
                let d = tqdm_desc(line);
                if !d.is_empty() {
                    active_label = d;
                }
            }
        }
    }
    let done = run_state == "succeeded";
    if done {
        cur_step = 6;
        cur_percent = 100;
    }
    if active_label.is_empty() && (1..=6).contains(&cur_step) {
        active_label = TRAIN_STEPS[cur_step - 1].to_string();
    }

    let steps: Vec<Value> = TRAIN_STEPS
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let idx = i + 1;
            let (st, percent) = if done || idx < cur_step {
                ("done", 100u8)
            } else if idx == cur_step {
                ("active", cur_percent)
            } else {
                ("pending", 0u8)
            };
            json!({ "name": name, "state": st, "percent": percent, "device": TRAIN_STEP_DEVICES[i] })
        })
        .collect();

    let total: u32 = TRAIN_STEP_WEIGHTS.iter().sum();
    let mut acc = 0u32;
    for (i, w) in TRAIN_STEP_WEIGHTS.iter().enumerate() {
        let idx = i + 1;
        if done || idx < cur_step {
            acc += w;
        } else if idx == cur_step {
            acc += w * cur_percent as u32 / 100;
        }
    }
    let overall = (acc * 100 / total).min(100);

    json!({
        "steps": steps,
        "overall_percent": overall,
        "active_step": cur_step,
        "active_label": active_label,
    })
}

async fn training_status(
    State(state): State<AppState>,
    AxumPath(slug): AxumPath<String>,
) -> Result<Json<Value>, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request("invalid slug"));
    }
    let name = training_container_name(&slug);
    let running = container_running(&name).await;

    let status_path = state
        .models_root
        .join(&slug)
        .join("train_status.json");
    let file_status: Option<Value> = fs::read_to_string(&status_path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok());

    let mut body = match file_status {
        Some(mut status) => {
            // A "running" file with no live container means the run died without
            // writing a terminal status (crash / OOM / killed) — UNLESS it is the
            // pre-launch f5gen phase, which the sync-server writes while it clones
            // voice positives and before any trainer container exists.
            if !running
                && status.get("state").and_then(Value::as_str) == Some("running")
                && status.get("step").and_then(Value::as_str) != Some("f5gen")
            {
                status["state"] = json!("stopped");
                status["message"] = json!("trainer container exited before completing");
            }
            status
        }
        None => {
            if running {
                json!({ "slug": slug, "state": "starting", "message": "container launching" })
            } else {
                json!({ "slug": slug, "state": "none", "message": "no training run found" })
            }
        }
    };
    body["container_running"] = json!(running);
    body["has_model"] = json!(state
        .models_root
        .join(&slug)
        .join(format!("{slug}.onnx"))
        .exists());

    // Templated per-step progress parsed from the trainer log tail.
    {
        let run_state = body.get("state").and_then(Value::as_str).unwrap_or("none");
        let step = body.get("step").and_then(Value::as_str).unwrap_or("");
        // Skip the six-step trainer progress during f5gen: the trainer hasn't run
        // yet, so any train.log is stale from a prior run. The app renders the
        // f5gen phase from the status body instead.
        if matches!(run_state, "running" | "starting" | "succeeded") && step != "f5gen" {
            let log_path = state.models_root.join(&slug).join("train.log");
            if let Some(text) = read_tail_bytes(&log_path, 1_048_576) {
                body["progress"] = parse_train_progress(&text, run_state);
            }
        }
    }

    // Pull the fields we need for benchmarking and ETA out of the status body.
    let run_state = body
        .get("state")
        .and_then(Value::as_str)
        .unwrap_or("none")
        .to_string();
    let steps = body.get("steps").and_then(Value::as_i64).unwrap_or(0);
    let model_size = body
        .get("model_size")
        .and_then(Value::as_str)
        .map(str::to_string);
    let personal = body
        .get("personal")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let started_at = body
        .get("started_at")
        .and_then(Value::as_str)
        .map(str::to_string);
    let updated_at = body
        .get("updated_at")
        .and_then(Value::as_str)
        .map(str::to_string);
    let started_ms = started_at.as_deref().and_then(rfc3339_ms);

    // On a terminal state, compute the run duration and persist the benchmark.
    let terminal = matches!(run_state.as_str(), "succeeded" | "failed" | "stopped");
    if terminal {
        let finished_ms = updated_at.as_deref().and_then(rfc3339_ms);
        let duration_ms = match (started_ms, finished_ms) {
            (Some(a), Some(b)) if b >= a => Some(b - a),
            _ => None,
        };
        if let Some(d) = duration_ms {
            body["duration_ms"] = json!(d);
        }
        if let Some(started) = started_at.as_deref() {
            if steps > 0 {
                let conn = state.db.lock().expect("db lock poisoned");
                let _ = db::record_training_run(
                    &conn,
                    &slug,
                    steps,
                    model_size.as_deref(),
                    personal,
                    started,
                    updated_at.as_deref(),
                    duration_ms,
                    &run_state,
                );
            }
        }

        // A succeeded run leaves a versioned, checksummed artifact folder and a
        // manifest under output/<slug>/runs/<run_id>/. Fold that into the models
        // table so every trained model is retained and queryable, and flag this
        // freshly finished run as the current live model. Idempotent on the poll.
        if run_state == "succeeded" {
            if let Some(run_id) = body.get("run_id").and_then(Value::as_str) {
                if let Some(model) =
                    read_model_manifest(&state.models_root, &slug, run_id, &body)
                {
                    let conn = state.db.lock().expect("db lock poisoned");
                    let _ = db::record_model(&conn, &model, now_ms(), true);
                }
            }
        }
    }

    // Estimate total/remaining time from the benchmark history.
    let avg = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::avg_ms_per_step(&conn, model_size.as_deref()).map_err(db_error)?
    };
    if let (Some((avg_ms_per_step, based_on)), true) = (avg, steps > 0) {
        let estimated_total = (avg_ms_per_step * steps as f64) as i64;
        body["avg_ms_per_step"] = json!(avg_ms_per_step);
        body["based_on_runs"] = json!(based_on);
        body["estimated_total_ms"] = json!(estimated_total);
        if matches!(run_state.as_str(), "running" | "starting") {
            if let Some(started) = started_ms {
                let elapsed = (now_ms() - started).max(0);
                body["elapsed_ms"] = json!(elapsed);
                body["remaining_ms"] = json!((estimated_total - elapsed).max(0));
            }
        }
    }

    // Overlay the queue: a pending job (not yet launched) reports as "queued"
    // with its position, even though the trainer has written no status file yet.
    // Done last so the terminal-run bookkeeping above still ran for any prior run.
    {
        let entry = {
            let conn = state.db.lock().expect("db lock poisoned");
            db::active_entry_for_slug(&conn, &slug).map_err(db_error)?
        };
        if let Some(entry) = entry {
            body["queue_id"] = json!(entry.id);
            // The pre-launch f5gen phase is a queued entry with no container yet,
            // but it is actively generating — don't overwrite it with "queued".
            let f5gen = body.get("step").and_then(Value::as_str) == Some("f5gen");
            if entry.state == "queued" && !running && !f5gen {
                let position = {
                    let conn = state.db.lock().expect("db lock poisoned");
                    db::queue_position(&conn, entry.id).ok().flatten()
                };
                body["state"] = json!("queued");
                body["message"] = json!("waiting in the training queue");
                body["queue_position"] = json!(position);
            }
        }
    }

    Ok(Json(body))
}

/// Tail the training log for a wake word. `?tail=N` bounds the returned lines.
async fn training_log(
    State(state): State<AppState>,
    AxumPath(slug): AxumPath<String>,
    RawQuery(query): RawQuery,
) -> Result<Response, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request("invalid slug"));
    }
    let params = parse_query(query.as_deref());
    let tail = params
        .get("tail")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(200)
        .clamp(1, 5000);

    let log_path = state.models_root.join(&slug).join("train.log");
    let text = fs::read_to_string(&log_path).unwrap_or_default();
    let body = if text.is_empty() {
        "(no log yet)".to_string()
    } else {
        let lines: Vec<&str> = text.lines().collect();
        let start = lines.len().saturating_sub(tail);
        lines[start..].join("\n")
    };
    Ok(([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], body).into_response())
}

/// List trained models found under the output directory, with any metrics.
async fn list_models(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let mut models = Vec::new();
    if let Ok(entries) = fs::read_dir(&*state.models_root) {
        let mut dirs: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_dir())
            .collect();
        dirs.sort();
        for dir in dirs {
            let Some(slug) = dir.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if !is_safe_slug(slug) {
                continue;
            }
            let onnx = dir.join(format!("{slug}.onnx"));
            if !onnx.exists() {
                continue;
            }
            let metrics: Option<Value> = fs::read_to_string(dir.join(format!("{slug}_metrics.json")))
                .ok()
                .and_then(|text| serde_json::from_str(&text).ok());
            let modified_ms = fs::metadata(&onnx)
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_millis() as i64);
            models.push(json!({
                "slug": slug,
                "modified_ms": modified_ms,
                "metrics": metrics,
            }));
        }
    }
    Ok(Json(json!({ "status": "ok", "models": models })))
}

/// Full per-run training provenance from the `models` table: every knob, the
/// real vs synthetic data counts, the code revision, checksums, scores, and the
/// complete manifest, one entry per trained model, newest first. This is the
/// accounting ledger for comparing models and knowing exactly what produced each.
async fn list_model_runs(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let rows = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::list_model_records(&conn).map_err(db_error)?
    };
    let runs: Vec<Value> = rows
        .iter()
        .filter_map(|s| serde_json::from_str(s).ok())
        .collect();
    Ok(Json(json!({ "status": "ok", "runs": runs })))
}

#[derive(Debug)]
struct AppError {
    status: StatusCode,
    message: String,
}

impl AppError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

impl From<io::Error> for AppError {
    fn from(error: io::Error) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: error.to_string(),
        }
    }
}

fn db_error(error: rusqlite::Error) -> AppError {
    AppError::internal(format!("database error: {error}"))
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = Json(json!({"error": self.message}));
        (self.status, body).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_labels_to_categories() {
        assert_eq!(category_for_label("positive"), Some("positive"));
        assert_eq!(category_for_label("false_negative"), Some("positive"));
        assert_eq!(category_for_label("hard_negative"), Some("hard_negative"));
        assert_eq!(category_for_label("false_positive"), Some("negative"));
        assert_eq!(category_for_label("background"), Some("background"));
        assert_eq!(category_for_label("other"), None);
    }

    #[test]
    fn capture_from_extra_reads_nested_capture_and_skips_blanks() {
        let extra = serde_json::json!({
            "session_id": "top-level-ignored",
            "capture": {
                "device_manufacturer": "Google",
                "device_model": "Pixel 7",
                "input_route": "builtin_mic: Pixel 7",
                "source_sample_rate_hz": 48000,
                "source_channels": 1,
                "session_id": "sess-abc",
                "os_version": "",
                "app_version": null,
            }
        });
        let capture = capture_from_extra(&extra);
        assert_eq!(capture.device_manufacturer.as_deref(), Some("Google"));
        assert_eq!(capture.device_model.as_deref(), Some("Pixel 7"));
        assert_eq!(capture.input_route.as_deref(), Some("builtin_mic: Pixel 7"));
        assert_eq!(capture.source_sample_rate_hz, Some(48000));
        assert_eq!(capture.source_channels, Some(1));
        assert_eq!(capture.session_id.as_deref(), Some("sess-abc"));
        // Empty string and null collapse to None so reprocess never clobbers.
        assert_eq!(capture.os_version, None);
        assert_eq!(capture.app_version, None);
    }

    #[test]
    fn capture_from_extra_absent_object_is_all_none() {
        let capture = capture_from_extra(&serde_json::json!({ "notes": "x" }));
        assert!(capture.device_model.is_none());
        assert!(capture.source_sample_rate_hz.is_none());
        assert!(capture.session_id.is_none());
    }

    #[test]
    fn sanitizes_filenames() {
        assert_eq!(safe_filename("Beep Beep!"), "beep-beep");
        assert_eq!(safe_filename(""), "clip");
    }

    #[test]
    fn validates_safe_slugs() {
        assert!(is_safe_slug("beep_beep"));
        assert!(is_safe_slug("a1"));
        assert!(!is_safe_slug("_bad"));
        assert!(!is_safe_slug("bad-name"));
        assert!(!is_safe_slug("../bad"));
    }

    #[test]
    fn transcript_tail_has_phrase_requires_phrase_at_end() {
        let phrase = normalized_words("all set");
        // Phrase at the tail (with a little trailing slack) is accepted.
        assert!(transcript_tail_has_phrase("the next words are all set.", &phrase));
        assert!(transcript_tail_has_phrase("All Set!", &phrase));
        assert!(transcript_tail_has_phrase("say all set now", &phrase));
        // Phrase absent, or buried mid-slice by a flaky pass, is rejected.
        assert!(!transcript_tail_has_phrase("the next words are over", &phrase));
        assert!(!transcript_tail_has_phrase(
            "the next words are all in the same sentence",
            &phrase
        ));
        assert!(!transcript_tail_has_phrase("all is not set", &phrase));
    }

    #[test]
    fn positive_context_first_fills_tail_aligned_window() {
        let words = vec![
            test_word("alpha", 0.00, 0.20),
            test_word("bravo", 0.35, 0.55),
            test_word("charlie", 0.90, 1.10),
            test_word("wake", 1.30, 1.55),
            test_word("word", 1.65, 1.90),
        ];

        assert_eq!(positive_context_first(&words, 3, 4), 2);
    }

    #[test]
    fn padded_bounds_expand_without_crossing_neighbors() {
        let words = vec![
            test_word("alpha", 0.00, 0.20),
            test_word("bravo", 0.35, 0.55),
            test_word("charlie", 0.90, 1.10),
            test_word("wake", 1.30, 1.55),
            test_word("word", 1.65, 1.90),
        ];

        // Final word: start padded but not into the previous word; tail padded.
        let (start, end) = padded_bounds(&words, 2, 4, 3.0, 0.08, 0.18, false);
        assert!((start - 0.82).abs() < 1e-9);
        assert!((end - 2.08).abs() < 1e-9);

        // With neighbor clamping on (negatives), an interior end clamps to the
        // next word's start instead of overrunning it.
        let (_, clamped_end) = padded_bounds(&words, 2, 3, 3.0, 0.08, 0.18, true);
        assert!((clamped_end - 1.65).abs() < 1e-9);

        // With neighbor clamping off (positives), the tail grows past the next
        // word toward the recording end so the wake phrase is not clipped.
        let (_, open_end) = padded_bounds(&words, 2, 3, 3.0, 0.08, 0.18, false);
        assert!((open_end - 1.73).abs() < 1e-9);

        // A large lead never pulls the start before the previous word's end.
        let (floor_start, _) = padded_bounds(&words, 1, 1, 3.0, 0.50, 0.10, true);
        assert!((floor_start - 0.20).abs() < 1e-9);
    }

    #[test]
    fn negative_ranges_use_word_chunks_with_hard_max_duration() {
        let words = vec![
            test_word("one", 0.0, 0.2),
            test_word("two", 0.5, 0.7),
            test_word("wake", 1.7, 1.9),
            test_word("word", 2.0, 2.2),
            test_word("three", 2.7, 2.9),
            test_word("four", 3.4, 3.6),
            test_word("five", 4.1, 4.3),
            test_word("overlong", 5.0, 6.7),
        ];

        let ranges = negative_ranges(&words, &[(1.6, 2.3)]);

        assert_eq!(
            ranges,
            vec![(0.0, 0.7, 0, 1), (2.7, 3.6, 4, 5), (4.1, 4.3, 6, 6)]
        );
    }

    #[test]
    fn clamp_slice_span_enforces_hard_max() {
        // Within the cap: bounds are untouched.
        let (start, end) = clamp_slice_span(0.5, 1.5, true);
        assert!((start - 0.5).abs() < 1e-9 && (end - 1.5).abs() < 1e-9);

        // Positive over the cap: keep the tail, trim the start in.
        let (start, end) = clamp_slice_span(0.0, 1.8, true);
        assert!((start - 0.3).abs() < 1e-9);
        assert!((end - 1.8).abs() < 1e-9);
        assert!(end - start <= MAX_SLICE_SECONDS + 1e-9);

        // Negative over the cap: keep the start, trim the end in.
        let (start, end) = clamp_slice_span(0.0, 1.8, false);
        assert!((start - 0.0).abs() < 1e-9);
        assert!((end - 1.5).abs() < 1e-9);
    }

    #[test]
    fn energy_burst_bounds_splits_repeated_bursts() {
        let sr = 16000.0;
        let mut samples: Vec<i16> = Vec::new();
        let silence = |secs: f64, v: &mut Vec<i16>| {
            for _ in 0..(secs * sr) as usize {
                v.push(0);
            }
        };
        let tone = |secs: f64, v: &mut Vec<i16>| {
            for i in 0..(secs * sr) as usize {
                v.push(if i % 2 == 0 { 8000 } else { -8000 });
            }
        };
        // Lead silence, then a "beep beep" (two 0.10s tones with a 0.10s internal
        // gap — shorter than the merge gap, so they fuse into one clip), a ~1s
        // gap between repetitions, then a second "beep beep".
        silence(0.30, &mut samples);
        tone(0.10, &mut samples);
        silence(0.10, &mut samples);
        tone(0.10, &mut samples);
        silence(1.00, &mut samples);
        tone(0.10, &mut samples);
        silence(0.10, &mut samples);
        tone(0.10, &mut samples);
        silence(0.30, &mut samples);

        let bursts = energy_burst_bounds(&samples, sr);
        assert_eq!(bursts.len(), 2, "expected two merged beep-beep bursts, got {bursts:?}");
        // First burst: the two beeps are merged into a single span near 0.30s.
        assert!(bursts[0].0 > 0.15 && bursts[0].0 < 0.31);
        assert!(bursts[0].1 - bursts[0].0 > 0.25);
        // Second burst lands well after the ~1s inter-repetition gap.
        assert!(bursts[1].0 > 1.4);
    }

    #[test]
    fn energy_burst_bounds_ignores_flat_take() {
        // Pure silence and a low steady hiss both lack a burst rising above the
        // floor, so nothing is sliced.
        assert!(energy_burst_bounds(&vec![0i16; 32000], 16000.0).is_empty());
        assert!(energy_burst_bounds(&vec![25i16; 32000], 16000.0).is_empty());
    }

    #[test]
    fn visible_range_drops_trimmed_words() {
        let words = vec![
            test_word("alpha", 0.00, 0.20),
            test_word("bravo", 0.35, 0.55),
            test_word("wake", 1.30, 1.55),
            test_word("word", 1.65, 1.90),
        ];

        // A clamped start past the first words drops them from the transcript.
        assert_eq!(visible_range(&words, 0, 3, 0.60, 2.00), (2, 3));
        // A clamped end drops trailing words instead.
        assert_eq!(visible_range(&words, 0, 3, 0.00, 1.20), (0, 1));
        // No trimming needed: the full range stays.
        assert_eq!(visible_range(&words, 0, 3, 0.00, 2.00), (0, 3));
    }

    #[test]
    fn bulk_clip_hash_id_is_stable_across_reprocess() {
        let words = vec![test_word("all", 1.30, 1.55), test_word("set", 1.65, 1.90)];
        // Same recording identity + cut + words => same id, so re-syncing or
        // reprocessing the stored source produces identical slice ids.
        let a = bulk_clip_hash_id("rec-1", "2026-01-01", 5000, "positive", 1.2, 1.9, &words);
        let b = bulk_clip_hash_id("rec-1", "2026-01-01", 5000, "positive", 1.2, 1.9, &words);
        assert_eq!(a, b);
        // A different recording or cut yields a different id.
        let c = bulk_clip_hash_id("rec-2", "2026-01-01", 5000, "positive", 1.2, 1.9, &words);
        assert_ne!(a, c);
        let d = bulk_clip_hash_id("rec-1", "2026-01-01", 5000, "negative", 1.2, 1.9, &words);
        assert_ne!(a, d);
    }

    #[test]
    fn hard_negative_context_flags_near_miss_frames() {
        // A wake phrase spoken after an explicit near-miss cue must be treated as
        // a hard negative, not a positive.
        let words = vec![
            test_word("this", 0.0, 0.2),
            test_word("is", 0.25, 0.4),
            test_word("not", 0.45, 0.6),
            test_word("the", 0.65, 0.75),
            test_word("wake", 0.8, 1.0),
            test_word("phrase", 1.05, 1.3),
            test_word("all", 1.5, 1.7),
            test_word("set", 1.75, 2.0),
        ];
        // "all set" at indices 6..=7 sits inside a "not the wake phrase" frame.
        assert!(is_hard_negative_context(&words, 6, 7));

        // The same phrase with an ordinary lead-in is a clean positive.
        let clean = vec![
            test_word("please", 0.0, 0.3),
            test_word("say", 0.35, 0.6),
            test_word("all", 0.7, 0.9),
            test_word("set", 0.95, 1.2),
        ];
        assert!(!is_hard_negative_context(&clean, 2, 3));
    }

    #[test]
    fn background_chunks_cover_source_and_drop_short_tail() {
        // Too short to yield any usable window.
        assert!(background_chunk_bounds(0.5).is_empty());
        // Exact single window.
        assert_eq!(background_chunk_bounds(2.0), vec![(0.0, 2.0)]);
        // A 1.0s tail meets the minimum and is kept.
        assert_eq!(
            background_chunk_bounds(5.0),
            vec![(0.0, 2.0), (2.0, 4.0), (4.0, 5.0)]
        );
        // A 0.5s remnant is below the minimum and dropped.
        assert_eq!(background_chunk_bounds(4.5), vec![(0.0, 2.0), (2.0, 4.0)]);
    }

    fn test_word(word: &str, start: f64, end: f64) -> WhisperWord {
        WhisperWord {
            word: word.to_string(),
            start,
            end,
            probability: 1.0,
        }
    }
}
