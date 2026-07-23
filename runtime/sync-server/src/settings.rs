//! Persisted server settings and their read/update handlers.
//!
//! The only server-side setting today is the canonical sync-server URL the app
//! stores so other devices agree on where to sync. It is persisted as JSON and
//! mirrored in `AppState.settings`; this module owns the type, its disk IO, and
//! the `GET`/`POST /settings` handlers.

use crate::error::AppError;
use crate::state::AppState;
use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// The persisted server settings.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct ServerSettings {
    pub(crate) sync_server_url: Option<String>,
}

/// The `POST /settings` request body.
#[derive(Debug, Deserialize)]
pub(crate) struct SettingsRequest {
    pub(crate) sync_server_url: Option<String>,
}

/// The `GET`/`POST /settings` reply: a status tag and the current settings.
#[derive(Debug, Serialize)]
pub(crate) struct SettingsResponse {
    pub(crate) status: &'static str,
    pub(crate) settings: ServerSettings,
}

/// Load settings from disk, defaulting to empty when the file is absent or
/// unparseable so the server always starts.
pub(crate) fn load_settings(path: &Path) -> ServerSettings {
    fs::read_to_string(path)
        .ok()
        .and_then(|contents| serde_json::from_str(&contents).ok())
        .unwrap_or_default()
}

/// Persist settings as pretty JSON, creating the parent directory if needed.
pub(crate) fn save_settings(path: &Path, settings: &ServerSettings) -> Result<(), AppError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let contents = serde_json::to_string_pretty(settings)
        .map_err(|error| AppError::internal(format!("serialize settings: {error}")))?;
    fs::write(path, contents)?;
    Ok(())
}

/// Trim an optional URL to `None` when it is missing or blank.
pub(crate) fn clean_optional_url(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// `GET /settings`: the current server settings.
pub(crate) async fn get_settings(State(state): State<AppState>) -> Json<SettingsResponse> {
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

/// `POST /settings`: replace the server settings and persist them.
pub(crate) async fn update_settings(
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
