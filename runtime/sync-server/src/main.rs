use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, Path as AxumPath, State},
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
    db: Arc<Mutex<Connection>>,
}

fn now_ms() -> i64 {
    Utc::now().timestamp_millis()
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

    let state = AppState {
        data_root: Arc::new(PathBuf::from(data_root)),
        incoming_root: Arc::new(PathBuf::from(incoming_root)),
        settings: Arc::new(Mutex::new(load_settings(&settings_path))),
        settings_path: Arc::new(settings_path),
        whisper_url: Arc::new(whisper_url),
        db: Arc::new(Mutex::new(db)),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/settings", get(get_settings).post(update_settings))
        .route("/projects", get(projects))
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
        .route(
            "/review/:slug/:category/:file_name",
            get(review_audio).delete(delete_review_clip),
        )
        .layer(DefaultBodyLimit::max(512 * 1024 * 1024))
        .with_state(state);

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
    // Cross-wake-word reuse: every other project's negatives, plus their
    // positives (reused as hard negatives), are available to this project.
    let total_negatives: i64 = rows.iter().map(|r| r.negative_count).sum();
    let total_positives: i64 = rows.iter().map(|r| r.positive_count).sum();
    let projects = rows
        .into_iter()
        .map(|row| {
            let pooled = (total_negatives - row.negative_count)
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
        if recording.script.trim().is_empty() {
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

/// Align and slice one already-materialized source WAV. Shared by the upload
/// path (source from the bundle) and the reprocess path (source is the stored
/// bulk_source WAV), so both cut slices identically.
async fn align_one_recording(
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
    summary.recordings += 1;
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

    // A wake phrase spoken inside a near-miss context ("...not the wake phrase
    // X...") must never train as a positive, but it is a valuable hard negative:
    // the true phrase in an explicitly-negative frame. Split the aligned phrase
    // occurrences into positives (kept) and hard negatives (filed under the
    // negative category, tagged distinctly) instead of discarding the latter.
    let (positive_ranges, hard_negative_ranges): (Vec<_>, Vec<_>) =
        phrase_ranges(&words, &phrase_words)
            .into_iter()
            .partition(|(first, last)| !is_hard_negative_context(&words, *first, *last));
    if positive_ranges.is_empty() {
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
            let dest = dest_root.join("negative").join(&file_name);
            if write_wav_slice(source, &dest, start, end)? {
                slice_rows.push(build_slice_row(
                    &clip_id,
                    "hard_negative",
                    "negative",
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
                "negative",
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
            let dest = dest_root.join("negative").join(&file_name);
            if write_wav_slice(source, &dest, start, end)? {
                slice_rows.push(build_slice_row(
                    &clip_id,
                    "negative",
                    "negative",
                    &dest,
                    &file_name,
                    start,
                    end,
                    slice_words,
                ));
                summary.negatives += 1;
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
    if !matches!(category, "positive" | "negative" | "background") {
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
        "negative" | "hard_negative" | "false_positive" => Some("negative"),
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
        assert_eq!(category_for_label("hard_negative"), Some("negative"));
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
