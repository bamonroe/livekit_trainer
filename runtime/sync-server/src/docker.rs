//! Thin wrappers around the `docker` CLI.
//!
//! The sync-server shells out to `docker` (over the mounted host socket) to
//! launch trainer containers, drive the resident F5-TTS generator, and inspect
//! container liveness. Centralizing the argv construction here keeps every call
//! site argv-only — nothing is ever shell-interpolated — and gives the training
//! and synth modules a single place to run and error-check a docker command.

use crate::error::AppError;
use std::env;

/// The container running the resident F5-TTS model (speech_services stack).
pub(crate) fn f5_container() -> String {
    env::var("F5_CONTAINER").unwrap_or_else(|_| "speech-f5tts".to_string())
}

/// The container running the resident Zonos (Zyphra) TTS model (speech_services
/// stack). A second voice-clone synthetic-positive source alongside F5.
pub(crate) fn zonos_container() -> String {
    env::var("ZONOS_CONTAINER").unwrap_or_else(|_| "speech-zonos".to_string())
}

/// The container running the resident Kokoro TTS FastAPI (speech_services stack).
/// Used to synthesize impostor negatives — the wake phrase in female Kokoro
/// voices — for the speaker-discriminative negative pool.
pub(crate) fn kokoro_container() -> String {
    env::var("KOKORO_CONTAINER").unwrap_or_else(|_| "speech-kokoro".to_string())
}

/// The by-convention trainer container name for a wake word.
pub(crate) fn training_container_name(slug: &str) -> String {
    format!("lkww-train-{slug}")
}

/// The resident TTS containers the sync-server drives for synthetic data. Each
/// pins a model on the GPU (~10 GiB combined: F5 ~2.5, Zonos ~5.4, Kokoro ~2.3),
/// none of which is needed once the pre-launch synth buckets are filled, so they
/// are stopped to free VRAM for a trainer run and restarted afterward.
pub(crate) fn speech_service_containers() -> Vec<String> {
    vec![f5_container(), zonos_container(), kokoro_container()]
}

/// Stop the resident speech-synthesis containers to free their GPU memory before
/// a trainer run. Best-effort: a container already stopped or missing is not an
/// error, and a short stop timeout keeps this from blocking training for long.
pub(crate) async fn stop_speech_services() {
    for name in speech_service_containers() {
        let _ = run_docker(vec!["stop".into(), "-t".into(), "5".into(), name]).await;
    }
}

/// Start the resident speech-synthesis containers back up — after a trainer run,
/// or before synth generation needs them. Best-effort and idempotent: `docker
/// start` on an already-running container is a harmless no-op.
pub(crate) async fn start_speech_services() {
    for name in speech_service_containers() {
        let _ = run_docker(vec!["start".into(), name]).await;
    }
}

/// Append a `-e KEY=VALUE` pair to a docker argv.
pub(crate) fn push_env(args: &mut Vec<String>, key: &str, value: String) {
    args.push("-e".into());
    args.push(format!("{key}={value}"));
}

/// Run `docker` with the given args off the async executor, returning its output.
pub(crate) async fn run_docker(args: Vec<String>) -> Result<std::process::Output, AppError> {
    tokio::task::spawn_blocking(move || std::process::Command::new("docker").args(&args).output())
        .await
        .map_err(|e| AppError::internal(format!("docker task join failed: {e}")))?
        .map_err(|e| AppError::internal(format!("failed to run docker: {e}")))
}

/// Run `docker` and fail with its stderr if the command exits non-zero.
pub(crate) async fn docker_ok(args: Vec<String>) -> Result<std::process::Output, AppError> {
    let output = run_docker(args).await?;
    if !output.status.success() {
        return Err(AppError::internal(format!(
            "docker command failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(output)
}

/// Is a container by this name currently running? Uses `docker inspect`; a
/// missing container (the `--rm` run already exited) reports false.
pub(crate) async fn container_running(name: &str) -> bool {
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
