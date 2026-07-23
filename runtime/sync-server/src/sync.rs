//! The `/sync` upload endpoint.
//!
//! A device uploads one zip bundle per sync. This handler archives it, extracts
//! and validates it, imports the pre-cut clips, and runs the three alignment
//! passes (bulk, background, test) so every long take is sliced into training
//! data — then returns a summary the app shows the user. `resolve_whisper_url`
//! picks the Whisper server (request header over env default) and is shared with
//! the reprocess path.

use crate::align::{
    align_background_recordings, align_bulk_recordings, align_test_recordings, alignment_summary,
};
use crate::bundle::{
    extract_zip, import_bundle, read_manifest, validate_manifest, validate_wavs,
};
use crate::error::AppError;
use crate::state::AppState;
use crate::util::validation_summary;
use axum::{body::Bytes, extract::State, http::HeaderMap, Json};
use chrono::Utc;
use serde::Serialize;
use std::env;
use std::fs;

/// The `/sync` reply: what was imported/aligned plus any warnings.
#[derive(Debug, Serialize)]
pub(crate) struct SyncResponse {
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


pub(crate) async fn sync(
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
pub(crate) fn resolve_whisper_url(state: &AppState, headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-whisper-server-url")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| state.whisper_url.as_ref().clone())
}

