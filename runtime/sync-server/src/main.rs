use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, Path as AxumPath, State},
    http::{header, HeaderMap},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    env,
    fs,
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex},
};

mod align;
mod audio;
mod bundle;
mod constants;
mod db;
mod docker;
mod error;
mod projects;
mod score;
mod settings;
mod slicing;
mod state;
mod synth;
mod train;
mod util;
mod whisper;

use align::{
    align_background_recordings, align_bulk_recordings, align_one_recording, align_test_recordings,
    alignment_summary, review_clip_path, AlignmentSummary,
};
use bundle::{extract_zip, import_bundle, read_manifest, validate_manifest, validate_wavs};
use constants::BACKGROUND_SCRIPT_MARKER;
use error::{db_error, AppError};
use projects::{create_project, projects};
use score::{model_test_grades, score_all_test_takes, score_recording};
use settings::{get_settings, load_settings, update_settings};
use state::AppState;
use synth::{
    delete_synth, generate_synth, generate_synth_status, synthetic_audio, synthetic_samples,
};
use train::{
    cancel_training, delete_queue_entry, list_model_runs, list_models, start_training, training_log,
    training_queue, training_scheduler, training_status,
};
use util::{is_safe_recording_id, is_safe_slug, safe_filename, validation_summary};
use whisper::WhisperWord;

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

#[derive(Debug, Serialize)]
struct DeleteRecordingResponse {
    status: &'static str,
    deleted: bool,
    removed_files: usize,
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

