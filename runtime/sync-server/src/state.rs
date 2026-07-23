//! Shared application state and time helpers.
//!
//! `AppState` is the handle every axum handler receives: the on-disk roots, the
//! resolved service URLs, the trainer configuration, the SQLite connection, and
//! the in-memory synth-job map. It is cloned per request (all fields are `Arc`),
//! so cloning is cheap and the inner state is shared. The small epoch-time
//! helpers live here too since state and timestamps are used together everywhere.

use crate::ServerSettings;
use chrono::Utc;
use rusqlite::Connection;
use serde::Serialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Per-request application state. Cheap to clone (every field is an `Arc`); the
/// underlying data is shared across all handlers.
#[derive(Clone)]
pub(crate) struct AppState {
    /// Root under which each wake word's `data/real/<slug>` tree lives.
    pub(crate) data_root: Arc<PathBuf>,
    /// Directory where uploaded bundle zips are archived before extraction.
    pub(crate) incoming_root: Arc<PathBuf>,
    /// Path of the persisted server-settings JSON.
    pub(crate) settings_path: Arc<PathBuf>,
    /// The live, mutable server settings.
    pub(crate) settings: Arc<Mutex<ServerSettings>>,
    // Whisper server URL from the WHISPER_SERVER_URL environment variable. This
    // is the source of truth for where audio is transcribed; the app no longer
    // configures it.
    pub(crate) whisper_url: Arc<Option<String>>,
    // Wake-word scorer service URL from SCORER_SERVER_URL. Optional: only the
    // model-test scoring endpoint needs it; the rest of the server works without.
    pub(crate) scorer_url: Arc<Option<String>>,
    // Trained-model directory (MODELS_DIR), mounted read-only. Used only to
    // fingerprint the .onnx so cached score curves invalidate when a model is
    // retrained. May not exist; then the fingerprint is a sentinel.
    pub(crate) models_root: Arc<PathBuf>,
    // Absolute path of the repo on the *host*. Training launches a sibling
    // trainer container over the mounted docker socket, and its `-v` bind mount
    // source must be a host path, not a path inside this container. None until
    // HOST_REPO_ROOT is set; the training endpoints refuse to run without it.
    pub(crate) host_repo_root: Arc<Option<String>>,
    // Trainer image tag launched for a full training run (TRAINER_IMAGE).
    pub(crate) trainer_image: Arc<String>,
    // Pass `--gpus all` to the trainer container (TRAINER_USE_GPU != "0").
    pub(crate) trainer_gpu: Arc<bool>,
    /// The shared SQLite connection.
    pub(crate) db: Arc<Mutex<Connection>>,
    // Serializes training-queue dispatch so the enqueue-triggered kick and the
    // periodic scheduler never launch the same queued job twice.
    pub(crate) dispatch_lock: Arc<tokio::sync::Mutex<()>>,
    // In-flight F5 synthetic-positive generation jobs, keyed by slug, so the app
    // can kick one off and poll its progress without a second job being launched
    // for the same wake word.
    pub(crate) synth_jobs: Arc<Mutex<HashMap<String, SynthJob>>>,
}

/// State of one F5 synthetic-positive generation run for a wake word.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct SynthJob {
    pub(crate) running: bool,
    pub(crate) requested: usize,
    pub(crate) wrote: usize,
    pub(crate) error: Option<String>,
}

/// Current wall-clock time in epoch milliseconds.
pub(crate) fn now_ms() -> i64 {
    Utc::now().timestamp_millis()
}

/// Parse an RFC 3339 / ISO 8601 timestamp (as written by train_job.sh) to epoch
/// milliseconds, or None if it does not parse.
pub(crate) fn rfc3339_ms(value: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|dt| dt.timestamp_millis())
}
