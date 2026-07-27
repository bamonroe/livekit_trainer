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

/// The by-convention trainer container name for a wake word.
pub(crate) fn training_container_name(slug: &str) -> String {
    format!("lkww-train-{slug}")
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
