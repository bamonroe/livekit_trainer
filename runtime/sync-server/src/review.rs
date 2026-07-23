//! Review endpoints: inspect and prune what the server sliced.
//!
//! The Review page is server-driven: it lists a wake word's recordings and their
//! generated slices, shows each recording's word/cut alignment, streams the
//! source and per-slice audio, and lets the user delete a bad slice or an entire
//! recording (removing both the DB rows and the WAV files). These handlers back
//! that page.

use crate::align::review_clip_path;
use crate::db;
use crate::error::{db_error, AppError};
use crate::state::AppState;
use crate::util::{is_safe_recording_id, is_safe_slug, safe_filename};
use crate::whisper::WhisperWord;
use axum::{
    extract::{Path as AxumPath, State},
    http::header,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// The bulk-review reply: every slice for a wake word.
#[derive(Debug, Serialize)]
pub(crate) struct BulkReviewResponse {
    status: &'static str,
    wake_word_slug: String,
    clips: Vec<BulkReviewClip>,
}

/// The recording-ids reply, with per-recording source checksums so a device can
/// skip re-uploading takes the server already holds.
#[derive(Debug, Serialize)]
pub(crate) struct BulkRecordingIdsResponse {
    status: &'static str,
    wake_word_slug: String,
    recording_ids: Vec<String>,
    /// Each recording id paired with its source-WAV SHA-256 (null for legacy
    /// rows), so a device can skip re-uploading takes the server already holds.
    checksums: Vec<RecordingChecksum>,
}

/// One recording id paired with its source-WAV SHA-256.
#[derive(Debug, Serialize)]
pub(crate) struct RecordingChecksum {
    id: String,
    sha256: Option<String>,
}

/// One recording's detail row: kind, timing, per-category counts, and capture
/// provenance, for the Review page's recording list.
#[derive(Debug, Serialize)]
pub(crate) struct RecordingDetailItem {
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

/// The recording-details reply.
#[derive(Debug, Serialize)]
pub(crate) struct RecordingDetailsResponse {
    status: &'static str,
    wake_word_slug: String,
    recordings: Vec<RecordingDetailItem>,
}

/// One recording's source alignment: the transcript words and the cut spans.
#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct BulkAlignmentResponse {
    status: String,
    wake_word_slug: String,
    source_recording: String,
    script: String,
    words: Vec<WhisperWord>,
    cuts: Vec<BulkAlignmentCut>,
}

/// One cut span within a recording's alignment.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct BulkAlignmentCut {
    id: String,
    label: String,
    start_sec: f64,
    end_sec: f64,
}

/// One reviewable slice: its label, timing within the source, and stats.
#[derive(Debug, Serialize)]
pub(crate) struct BulkReviewClip {
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

/// The delete-slice reply.
#[derive(Debug, Serialize)]
pub(crate) struct DeleteReviewClipResponse {
    status: &'static str,
    deleted: bool,
}

/// The delete-recording reply, including how many files were removed.
#[derive(Debug, Serialize)]
pub(crate) struct DeleteRecordingResponse {
    status: &'static str,
    deleted: bool,
    removed_files: usize,
}

pub(crate) async fn delete_recording(
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

pub(crate) async fn bulk_review(
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

pub(crate) async fn bulk_recording_ids(
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

pub(crate) async fn bulk_recording_details(
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

pub(crate) async fn bulk_alignment(
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

pub(crate) async fn bulk_source_audio(
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

pub(crate) async fn review_audio(
    State(state): State<AppState>,
    AxumPath((slug, category, file_name)): AxumPath<(String, String, String)>,
) -> Result<impl IntoResponse, AppError> {
    let path = review_clip_path(&state.data_root, &slug, &category, &file_name)?;
    let bytes = fs::read(path)?;
    Ok(([(header::CONTENT_TYPE, "audio/wav")], bytes))
}

pub(crate) async fn delete_review_clip(
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

