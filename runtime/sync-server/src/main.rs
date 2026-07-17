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
// Whisper word timestamps drift from the true audio, so cutting exactly at
// word.start/word.end clips onsets and (worst of all) chops the tail-aligned
// wake phrase. Nudge each cut outward, bounded by the neighboring words, to keep
// slices honest to their transcript. The positive tail is padded hardest because
// a positive that lost its wake phrase is the most damaging error.
const CUT_LEAD_PADDING_SECONDS: f64 = 0.08;
const POSITIVE_TAIL_PADDING_SECONDS: f64 = 0.28;
const NEGATIVE_TAIL_PADDING_SECONDS: f64 = 0.10;
const NEGATIVE_TARGET_SECONDS: f64 = 1.5;

#[derive(Clone)]
struct AppState {
    data_root: Arc<PathBuf>,
    incoming_root: Arc<PathBuf>,
    settings_path: Arc<PathBuf>,
    settings: Arc<Mutex<ServerSettings>>,
    db: Arc<Mutex<Connection>>,
}

fn now_ms() -> i64 {
    Utc::now().timestamp_millis()
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct ServerSettings {
    sync_server_url: Option<String>,
    whisper_server_url: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SettingsRequest {
    sync_server_url: Option<String>,
    whisper_server_url: Option<String>,
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

#[derive(Debug, Deserialize)]
struct Manifest {
    schema_version: u64,
    wake_word: WakeWord,
    clips: Vec<Clip>,
    #[serde(default)]
    bulk_recordings: Vec<BulkRecording>,
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
    warnings: Vec<String>,
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

    let state = AppState {
        data_root: Arc::new(PathBuf::from(data_root)),
        incoming_root: Arc::new(PathBuf::from(incoming_root)),
        settings: Arc::new(Mutex::new(load_settings(&settings_path))),
        settings_path: Arc::new(settings_path),
        db: Arc::new(Mutex::new(db)),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/settings", get(get_settings).post(update_settings))
        .route("/projects", get(projects))
        .route("/sync", post(sync))
        .route("/bulk/:slug/recordings", get(bulk_recording_ids))
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
        whisper_server_url: clean_optional_url(request.whisper_server_url),
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
    let projects = rows
        .into_iter()
        .map(|row| ProjectSummary {
            id: row.external_id,
            slug: row.slug,
            phrase: row.phrase,
            created_at_millis: row.created_at_ms,
            bulk_slice_count: row.bulk_slice_count as usize,
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
    let whisper_server_url = headers
        .get("x-whisper-server-url")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| {
            state
                .settings
                .lock()
                .expect("settings lock poisoned")
                .whisper_server_url
                .clone()
        });

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
        let alignment = align_bulk_recordings(
            &extract_root,
            &manifest,
            &state.data_root,
            &state.db,
            whisper_server_url.as_deref(),
        )
        .await?;
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
    let recording_ids = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::recording_ids(&conn, &slug).map_err(db_error)?
    };
    Ok(Json(BulkRecordingIdsResponse {
        status: "ok",
        wake_word_slug: slug.clone(),
        recording_ids,
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
        summary.recordings += 1;
        let source = match safe_join(bundle, &recording.file) {
            Ok(source) => source,
            Err(error) => {
                summary
                    .warnings
                    .push(format!("{}: {}", recording.id, error.message));
                continue;
            }
        };
        let whisper = match transcribe_with_words(whisper_url, &source, &recording.script).await {
            Ok(whisper) => whisper,
            Err(error) => {
                summary
                    .warnings
                    .push(format!("{}: {}", recording.id, error.message));
                continue;
            }
        };
        let words = whisper_words(&whisper);
        if words.is_empty() {
            summary.warnings.push(format!(
                "{}: Whisper returned no word timestamps",
                recording.id
            ));
            continue;
        }

        let phrase_words = normalized_words(&phrase);
        if phrase_words.is_empty() {
            summary.warnings.push(format!(
                "{}: wake phrase has no alignable words",
                recording.id
            ));
            continue;
        }

        let positive_ranges = phrase_ranges(&words, &phrase_words)
            .into_iter()
            .filter(|(first, last)| !is_hard_negative_context(&words, *first, *last))
            .collect::<Vec<_>>();
        if positive_ranges.is_empty() {
            summary.warnings.push(format!(
                "{}: no aligned wake phrase occurrences found in transcript {:?}",
                recording.id,
                whisper.text.trim()
            ));
        }

        // Persist the raw source recording, then remove any stale slice files
        // this recording produced on a previous pass before writing the new set.
        let source_wav = match store_bulk_source(&dest_root, &recording.id, &source) {
            Ok(path) => path,
            Err(error) => {
                summary
                    .warnings
                    .push(format!("{}: {}", recording.id, error.message));
                continue;
            }
        };
        let old_paths = {
            let conn = db.lock().expect("db lock poisoned");
            db::active_slice_paths(&conn, &recording.id).map_err(db_error)?
        };
        for old in old_paths {
            let old = PathBuf::from(old);
            if old.is_file() {
                let _ = fs::remove_file(old);
            }
        }

        let source_end = wav_duration_seconds(&source)
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
            let slice_words = &words[context_first..=*last];
            let clip_id = bulk_clip_hash_id(recording, "positive", start, end, slice_words);
            let file_name = format!("{}_{}.wav", safe_filename(&clip_id), safe_filename(&phrase));
            let dest = dest_root.join("positive").join(&file_name);
            if write_wav_slice(&source, &dest, start, end)? {
                slice_rows.push(build_slice_row(
                    &clip_id,
                    "positive",
                    &dest,
                    &file_name,
                    start,
                    end,
                    slice_words,
                ));
                summary.positives += 1;
            }
            occupied.push((start, end));
        }

        let negative_ranges = negative_ranges(&words, &occupied);
        for (_, _, word_start, word_end) in negative_ranges.iter() {
            let slice_words = &words[*word_start..=*word_end];
            let (start, end) = padded_bounds(
                &words,
                *word_start,
                *word_end,
                source_end,
                CUT_LEAD_PADDING_SECONDS,
                NEGATIVE_TAIL_PADDING_SECONDS,
                true,
            );
            let clip_id = bulk_clip_hash_id(recording, "negative", start, end, slice_words);
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
            if write_wav_slice(&source, &dest, start, end)? {
                slice_rows.push(build_slice_row(
                    &clip_id,
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
        let prompts = recording
            .script
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(ToString::to_string)
            .collect();

        let alignment = db::RecordingAlignment {
            recording_id: recording.id.clone(),
            project_slug: slug.clone(),
            phrase: phrase.clone(),
            external_id: external_id.clone(),
            script: recording.script.clone(),
            recorded_at: recording.recorded_at.clone(),
            duration_ms: recording.duration_ms as i64,
            source_wav: source_wav.to_string_lossy().to_string(),
            bundle: None,
            transcript_text: whisper.text.trim().to_string(),
            whisper_url: Some(whisper_url.to_string()),
            words: word_rows,
            prompts,
            slices: slice_rows,
        };
        {
            let mut conn = db.lock().expect("db lock poisoned");
            db::store_recording_alignment(&mut conn, &alignment, now_ms()).map_err(db_error)?;
        }
    }

    Ok(summary)
}

fn build_slice_row(
    clip_id: &str,
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
        label: category.to_string(),
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
    script: &str,
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
    let form = reqwest::multipart::Form::new()
        .part("file", part)
        .text("response_format", "verbose_json")
        .text("temperature", "0.0")
        .text("no_context", "true")
        .text("word_timestamps", "true")
        .text("prompt", script.to_string());
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
    recording: &BulkRecording,
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
        "recording_id": recording.id,
        "recording_file": recording.file,
        "recorded_at": recording.recorded_at,
        "recording_duration_ms": recording.duration_ms,
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

fn store_bulk_source(
    dest_root: &Path,
    recording_id: &str,
    source: &Path,
) -> Result<PathBuf, AppError> {
    let dir = dest_root.join("bulk_source");
    fs::create_dir_all(&dir)?;
    let dest = dir.join(format!("{}.wav", safe_filename(recording_id)));
    fs::copy(source, &dest)?;
    Ok(dest)
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
        "Aligned {} bulk recordings into {} positives and {} negatives",
        summary.recordings, summary.positives, summary.negatives
    );
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

    fn test_word(word: &str, start: f64, end: f64) -> WhisperWord {
        WhisperWord {
            word: word.to_string(),
            start,
            end,
            probability: 1.0,
        }
    }
}
