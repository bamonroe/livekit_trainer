//! Reprocess endpoints: re-slice stored takes without a fresh upload.
//!
//! When the slicing logic changes, the operator can re-run alignment over the
//! already-stored source WAVs for a whole project or a single recording. This
//! cuts identically to the sync path (it routes through `align_one_recording`),
//! but reads the source from disk and keeps existing provenance, so no device
//! round-trip is needed.

use crate::align::{align_one_recording, alignment_summary, AlignmentSummary};
use crate::constants::BACKGROUND_SCRIPT_MARKER;
use crate::db;
use crate::error::{db_error, AppError};
use crate::state::AppState;
use crate::sync::resolve_whisper_url;
use crate::util::{is_safe_recording_id, is_safe_slug, safe_filename};
use axum::{
    extract::{Path as AxumPath, State},
    http::HeaderMap,
    Json,
};
use serde::Serialize;

/// The reprocess reply: how many recordings were re-sliced and into what.
#[derive(Debug, Serialize)]
pub(crate) struct ReprocessResponse {
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

pub(crate) async fn reprocess_project(
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

pub(crate) async fn reprocess_recording(
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

