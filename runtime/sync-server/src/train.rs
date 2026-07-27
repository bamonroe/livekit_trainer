//! Full-training pipeline: request validation, queueing, container launch, and
//! progress reporting.
//!
//! Training is asynchronous and serialized: a request is validated into a
//! `ValidatedTrain`, stored on a queue, and one trainer container is launched at
//! a time by the dispatcher/scheduler. This module owns the request/validated
//! types, the queue endpoints, the docker launch (every knob passed as an
//! argv `-e` pair), the per-step log-tail progress parser, and the status/log/
//! model-listing endpoints. F5 positives are topped up (via synth) just before
//! the container starts.

use crate::db;
use crate::docker::{container_running, push_env, run_docker, training_container_name};
use crate::error::{db_error, AppError};
use crate::score::read_model_manifest;
use crate::state::{now_ms, rfc3339_ms, AppState};
use crate::synth::{ensure_f5_positives, ensure_zonos_positives};
use crate::util::{is_safe_slug, parse_query};
use axum::{
    extract::{Path as AxumPath, RawQuery, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs;
use std::path::PathBuf;

/// Hyperparameters for a full training run. All optional; each falls back to the
/// same default as scripts/generate_config.py when omitted.
#[derive(Debug, Deserialize)]
pub(crate) struct TrainRequest {
    steps: Option<u32>,
    model_size: Option<String>,
    target_fp_per_hour: Option<f64>,
    n_samples: Option<u32>,
    n_samples_val: Option<u32>,
    personal: Option<bool>,
    positive_boost: Option<u32>,
    // How many F5 voice-cloned positives to generate (in the sync-server, before
    // the trainer launches) and fold into training. 0/omitted = no F5 clips.
    f5_count: Option<u32>,
    // How many Zonos voice-cloned positives to generate (same pre-launch path as
    // F5, from the resident speech-zonos container). 0/omitted = no Zonos clips.
    zonos_count: Option<u32>,
    positive_per_batch: Option<u32>,
    real_samples_dir: Option<String>,
    // "start" | "end". Selects the leading-context recipe (see
    // generate_config.py --token-type). Defaults to "end" when omitted.
    token_type: Option<String>,
    // Realistic-positive compositing knobs (augment_realistic.py). All optional;
    // omitted values fall back to the trainer's sidecar defaults.
    realistic: Option<bool>,
    lead_probability: Option<f64>,
    real_lead_fraction: Option<f64>,
    synthetic_lead: Option<bool>,
    max_lead_ms: Option<u32>,
    lead_gap_min_ms: Option<u32>,
    lead_gap_max_ms: Option<u32>,
    margin_min_ms: Option<u32>,
    margin_max_ms: Option<u32>,
    snr_min_db: Option<f64>,
    snr_max_db: Option<f64>,
    background_augment: Option<bool>,
    voice_peak: Option<f64>,
}

/// Allowlisted trainer model sizes. generate_config.py writes this straight into
/// the YAML, so reject anything unexpected up front.
fn valid_model_size(size: &str) -> bool {
    matches!(size, "tiny" | "small" | "medium" | "large")
}

/// A real-samples dir is a plain relative path passed to the assembler/config.
fn valid_real_samples_dir(dir: &str) -> bool {
    !dir.is_empty()
        && dir.len() <= 128
        && dir
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '/' | '_' | '-'))
        && !dir.contains("..")
}

/// Fully-resolved, validated training hyperparameters. This is what gets stored
/// in the queue (as JSON) and rebuilt into the trainer container's argv at
/// dispatch, so a job launched minutes after it was enqueued still runs with the
/// exact settings the user chose.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ValidatedTrain {
    pub(crate) steps: u32,
    pub(crate) model_size: String,
    target_fp_per_hour: f64,
    pub(crate) personal: bool,
    positive_boost: u32,
    pub(crate) f5_count: u32,
    pub(crate) zonos_count: u32,
    n_samples: Option<u32>,
    n_samples_val: Option<u32>,
    positive_per_batch: Option<u32>,
    real_samples_dir: String,
    token_type: String,
    // Realistic-positive compositing (see augment_realistic.py). Fully resolved
    // and clamped so the trainer container can render the sidecar verbatim.
    realistic: bool,
    lead_probability: f64,
    real_lead_fraction: f64,
    synthetic_lead: bool,
    max_lead_ms: u32,
    lead_gap_min_ms: u32,
    lead_gap_max_ms: u32,
    margin_min_ms: u32,
    margin_max_ms: u32,
    snr_min_db: f64,
    snr_max_db: f64,
    background_augment: bool,
    voice_peak: f64,
}

/// Validate + default a raw train request into a `ValidatedTrain`, rejecting bad
/// input at enqueue time so the queue only ever holds runnable jobs.
fn validate_train(request: TrainRequest) -> Result<ValidatedTrain, AppError> {
    let steps = request.steps.unwrap_or(50_000).clamp(100, 500_000);
    let model_size = request.model_size.unwrap_or_else(|| "medium".to_string());
    if !valid_model_size(&model_size) {
        return Err(AppError::bad_request(format!(
            "invalid model_size: {model_size}"
        )));
    }
    let target_fp_per_hour = request.target_fp_per_hour.unwrap_or(0.2);
    if !(target_fp_per_hour.is_finite() && (0.0..=100.0).contains(&target_fp_per_hour)) {
        return Err(AppError::bad_request("invalid target_fp_per_hour"));
    }
    let real_samples_dir = request
        .real_samples_dir
        .unwrap_or_else(|| "./data/train".to_string());
    if !valid_real_samples_dir(&real_samples_dir) {
        return Err(AppError::bad_request("invalid real_samples_dir"));
    }
    let token_type = request.token_type.unwrap_or_else(|| "end".to_string());
    if !matches!(token_type.as_str(), "start" | "end") {
        return Err(AppError::bad_request(
            "invalid token_type (expected start|end)",
        ));
    }

    // --- realistic-positive compositing knobs --------------------------------
    // Clamp fractions to [0,1], millisecond spans to a sane window, and dB to a
    // reasonable range; keep each range's min <= max so the trainer's randint /
    // uniform never sees an inverted interval.
    let clamp01 = |v: f64| v.clamp(0.0, 1.0);
    let realistic = request.realistic.unwrap_or(true);
    let lead_probability = clamp01(request.lead_probability.unwrap_or(0.75));
    let real_lead_fraction = clamp01(request.real_lead_fraction.unwrap_or(0.6));
    let voice_peak = request.voice_peak.unwrap_or(0.7).clamp(0.05, 1.0);
    let max_lead_ms = request.max_lead_ms.unwrap_or(900).min(2000);
    let mut lead_gap_min_ms = request.lead_gap_min_ms.unwrap_or(40).min(2000);
    let mut lead_gap_max_ms = request.lead_gap_max_ms.unwrap_or(300).min(2000);
    if lead_gap_min_ms > lead_gap_max_ms {
        std::mem::swap(&mut lead_gap_min_ms, &mut lead_gap_max_ms);
    }
    let mut margin_min_ms = request.margin_min_ms.unwrap_or(100).min(2000);
    let mut margin_max_ms = request.margin_max_ms.unwrap_or(700).min(2000);
    if margin_min_ms > margin_max_ms {
        std::mem::swap(&mut margin_min_ms, &mut margin_max_ms);
    }
    let mut snr_min_db = request.snr_min_db.unwrap_or(0.0).clamp(-10.0, 60.0);
    let mut snr_max_db = request.snr_max_db.unwrap_or(18.0).clamp(-10.0, 60.0);
    if !(snr_min_db.is_finite() && snr_max_db.is_finite()) {
        return Err(AppError::bad_request("invalid snr range"));
    }
    if snr_min_db > snr_max_db {
        std::mem::swap(&mut snr_min_db, &mut snr_max_db);
    }

    Ok(ValidatedTrain {
        steps,
        model_size,
        target_fp_per_hour,
        personal: request.personal.unwrap_or(false),
        positive_boost: request.positive_boost.unwrap_or(1).clamp(1, 50),
        f5_count: request.f5_count.unwrap_or(0).min(50_000),
        zonos_count: request.zonos_count.unwrap_or(0).min(50_000),
        n_samples: request.n_samples,
        n_samples_val: request.n_samples_val,
        positive_per_batch: request.positive_per_batch,
        real_samples_dir,
        token_type,
        realistic,
        lead_probability,
        real_lead_fraction,
        synthetic_lead: request.synthetic_lead.unwrap_or(true),
        max_lead_ms,
        lead_gap_min_ms,
        lead_gap_max_ms,
        margin_min_ms,
        margin_max_ms,
        snr_min_db,
        snr_max_db,
        background_augment: request.background_augment.unwrap_or(true),
        voice_peak,
    })
}

/// Enqueue a full training run. The training scheduler runs one trainer
/// container at a time and dispatches queued jobs oldest-first, so pressing
/// "start" several times (across one or many wake words) lines them up instead
/// of failing. No recorded clips are required: the trainer synthesizes positives
/// from the phrase, so a brand-new word can still train.
pub(crate) async fn start_training(
    State(state): State<AppState>,
    AxumPath(slug): AxumPath<String>,
    Json(request): Json<TrainRequest>,
) -> Result<Json<Value>, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request("invalid slug"));
    }
    if state.host_repo_root.is_none() {
        return Err(AppError::internal(
            "training disabled: HOST_REPO_ROOT not set on the sync-server",
        ));
    }

    // The wake word must exist so the dispatcher can supply its target phrase.
    let phrase = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::project_phrase(&conn, &slug).map_err(db_error)?
    };
    let Some((phrase, _external_id)) = phrase else {
        return Err(AppError {
            status: StatusCode::NOT_FOUND,
            message: format!("unknown wake word: {slug}"),
        });
    };

    let validated = validate_train(request)?;
    let params_json = serde_json::to_string(&validated)
        .map_err(|e| AppError::internal(format!("serialize train params: {e}")))?;

    let (id, position) = {
        let conn = state.db.lock().expect("db lock poisoned");
        let id = db::enqueue_training(&conn, &slug, &params_json, now_ms()).map_err(db_error)?;
        let position = db::queue_position(&conn, id).map_err(db_error)?.unwrap_or(0);
        (id, position)
    };

    // Nudge the scheduler so a job enqueued into an idle pipeline starts right
    // away instead of waiting for the next poll tick.
    let dispatch_state = state.clone();
    tokio::spawn(async move {
        dispatch_training(&dispatch_state).await;
    });

    Ok(Json(json!({
        "status": if position <= 1 { "started" } else { "queued" },
        "slug": slug,
        "phrase": phrase,
        "queue_id": id,
        "position": position,
        "steps": validated.steps,
        "model_size": validated.model_size,
        "personal": validated.personal,
        "token_type": validated.token_type,
    })))
}

/// Build the full `docker run ...` argv for a trainer container. Pure (no IO):
/// every hyperparameter is turned into a `-e KEY=VALUE` argv pair — never
/// shell-interpolated — so this is both injection-safe and unit-testable.
fn build_train_argv(
    host_repo_root: &str,
    name: &str,
    gpu: bool,
    trainer_image: &str,
    slug: &str,
    phrase: &str,
    vt: &ValidatedTrain,
) -> Vec<String> {
    let mut args: Vec<String> = vec![
        "run".into(),
        "-d".into(),
        "--rm".into(),
        "--name".into(),
        name.to_string(),
    ];
    if gpu {
        args.push("--gpus".into());
        args.push("all".into());
    }
    args.push("-v".into());
    args.push(format!("{host_repo_root}:/work"));
    // Mount the realistic-positive augment module over the installed one (same
    // mechanism as train_ctxfix.sh), so a training run composites positives as
    // one clear voice on a background bed without needing a trainer-image
    // rebuild. train_job.sh writes the sidecar this reads (AUG_REALISTIC_CONFIG).
    if vt.realistic {
        args.push("-v".into());
        args.push(format!(
            "{host_repo_root}/trainer/patches/augment_realistic.py:\
/opt/conda/lib/python3.11/site-packages/livekit/wakeword/data/augment.py:ro"
        ));
    }
    push_env(&mut args, "SLUG", slug.to_string());
    push_env(&mut args, "PHRASE", phrase.to_string());
    push_env(&mut args, "STEPS", vt.steps.to_string());
    push_env(&mut args, "MODEL_SIZE", vt.model_size.clone());
    push_env(&mut args, "TARGET_FP_PER_HOUR", vt.target_fp_per_hour.to_string());
    push_env(
        &mut args,
        "PERSONAL",
        if vt.personal { "1".into() } else { "0".into() },
    );
    push_env(&mut args, "POSITIVE_BOOST", vt.positive_boost.to_string());
    push_env(&mut args, "F5_COUNT", vt.f5_count.to_string());
    push_env(&mut args, "ZONOS_COUNT", vt.zonos_count.to_string());
    if let Some(n) = vt.n_samples {
        push_env(&mut args, "N_SAMPLES", n.to_string());
    }
    if let Some(n) = vt.n_samples_val {
        push_env(&mut args, "N_SAMPLES_VAL", n.to_string());
    }
    if let Some(n) = vt.positive_per_batch {
        push_env(&mut args, "POSITIVE_PER_BATCH", n.to_string());
    }
    push_env(&mut args, "REAL_SAMPLES_DIR", vt.real_samples_dir.clone());
    push_env(&mut args, "TOKEN_TYPE", vt.token_type.clone());
    // Realistic-positive compositing knobs -> train_job.sh sidecar.
    push_env(
        &mut args,
        "REALISTIC",
        if vt.realistic { "1".into() } else { "0".into() },
    );
    push_env(&mut args, "LEAD_PROBABILITY", vt.lead_probability.to_string());
    push_env(&mut args, "REAL_LEAD_FRACTION", vt.real_lead_fraction.to_string());
    push_env(
        &mut args,
        "SYNTHETIC_LEAD",
        if vt.synthetic_lead { "1".into() } else { "0".into() },
    );
    push_env(&mut args, "MAX_LEAD_MS", vt.max_lead_ms.to_string());
    push_env(&mut args, "LEAD_GAP_MIN_MS", vt.lead_gap_min_ms.to_string());
    push_env(&mut args, "LEAD_GAP_MAX_MS", vt.lead_gap_max_ms.to_string());
    push_env(&mut args, "MARGIN_MIN_MS", vt.margin_min_ms.to_string());
    push_env(&mut args, "MARGIN_MAX_MS", vt.margin_max_ms.to_string());
    push_env(&mut args, "SNR_MIN_DB", vt.snr_min_db.to_string());
    push_env(&mut args, "SNR_MAX_DB", vt.snr_max_db.to_string());
    push_env(
        &mut args,
        "BACKGROUND_AUGMENT",
        if vt.background_augment { "1".into() } else { "0".into() },
    );
    push_env(&mut args, "VOICE_PEAK", vt.voice_peak.to_string());
    // The job stamps its own image tag into the model manifest for provenance.
    push_env(&mut args, "TRAINER_IMAGE", trainer_image.to_string());
    args.push(trainer_image.to_string());
    args.push("bash".into());
    args.push("-lc".into());
    args.push("bash /work/trainer/scripts/train_job.sh".into());
    args
}

/// Launch the trainer container for one validated job and return its container
/// name. Every hyperparameter travels as a `-e KEY=VALUE` pair (argv, never
/// shell-interpolated), so nothing here can inject shell.
async fn launch_train_container(
    state: &AppState,
    slug: &str,
    phrase: &str,
    vt: &ValidatedTrain,
) -> Result<String, AppError> {
    let Some(host_repo_root) = (*state.host_repo_root).clone() else {
        return Err(AppError::internal(
            "training disabled: HOST_REPO_ROOT not set on the sync-server",
        ));
    };
    let name = training_container_name(slug);
    // Clear any dead container left behind by a crash so the name is free.
    let _ = run_docker(vec!["rm".into(), "-f".into(), name.clone()]).await;

    let args = build_train_argv(
        &host_repo_root,
        &name,
        *state.trainer_gpu,
        &state.trainer_image,
        slug,
        phrase,
        vt,
    );

    // F5 voice-cloned positives are generated HERE, in the sync-server, before
    // the trainer launches: only this container can reach the speech-f5tts
    // service (the trainer container has no docker access). This tops the synth
    // bucket up to vt.f5_count; the trainer then pools them into positives. The
    // f5gen phase is written into train_status.json so the app's training screen
    // shows it. Non-fatal — training proceeds on whatever clips exist.
    ensure_f5_positives(state, slug, phrase, vt).await;

    // Zonos voice-cloned positives are generated the same way, from the resident
    // speech-zonos container, as an independent third positive source. Runs after
    // F5 so the two pre-launch top-ups never contend for the same job-map key.
    ensure_zonos_positives(state, slug, phrase, vt).await;

    let output = run_docker(args).await?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(AppError::internal(format!(
            "docker run failed: {}",
            stderr.trim()
        )));
    }
    Ok(name)
}

/// Render a queue row for the API, folding in its live position.
fn queue_entry_json(entry: &db::QueueEntry, position: Option<i64>) -> Value {
    let params: Value =
        serde_json::from_str(&entry.params_json).unwrap_or_else(|_| json!({}));
    json!({
        "id": entry.id,
        "slug": entry.slug,
        "state": entry.state,
        "position": position,
        "container_name": entry.container_name,
        "enqueued_at_ms": entry.enqueued_at_ms,
        "started_at_ms": entry.started_at_ms,
        "finished_at_ms": entry.finished_at_ms,
        "params": params,
    })
}

/// The live training pipeline: every queued or running job, oldest first.
pub(crate) async fn training_queue(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let entries = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::active_queue(&conn).map_err(db_error)?
    };
    // Position: running jobs are 0, queued jobs count 1..N in id order.
    let mut queued_rank = 0i64;
    let jobs: Vec<Value> = entries
        .iter()
        .map(|e| {
            let position = if e.state == "running" {
                Some(0)
            } else {
                queued_rank += 1;
                Some(queued_rank)
            };
            queue_entry_json(e, position)
        })
        .collect();
    Ok(Json(json!({ "status": "ok", "jobs": jobs })))
}

/// Cancel every queued/running job for a wake word: mark the rows canceled and
/// kill any live trainer container. The killed container's `--rm` cleanup plus
/// the status endpoint's reconciliation report it as stopped.
pub(crate) async fn cancel_training(
    State(state): State<AppState>,
    AxumPath(slug): AxumPath<String>,
) -> Result<Json<Value>, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request("invalid slug"));
    }
    let canceled = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::cancel_slug_jobs(&conn, &slug, now_ms()).map_err(db_error)?
    };
    // Kill any container that was actually running for these jobs.
    for entry in &canceled {
        if let Some(name) = &entry.container_name {
            let _ = run_docker(vec!["rm".into(), "-f".into(), name.clone()]).await;
        }
    }
    // Also kill the by-convention container name in case a run started without a
    // recorded container_name yet.
    let _ = run_docker(vec![
        "rm".into(),
        "-f".into(),
        training_container_name(&slug),
    ])
    .await;

    Ok(Json(json!({
        "status": "canceled",
        "slug": slug,
        "canceled": canceled.len(),
    })))
}

/// Cancel a single queue entry by id (a queued job, or the running one).
pub(crate) async fn delete_queue_entry(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<i64>,
) -> Result<Json<Value>, AppError> {
    let entry = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::cancel_queue_entry(&conn, id, now_ms()).map_err(db_error)?
    };
    let Some(entry) = entry else {
        return Err(AppError {
            status: StatusCode::NOT_FOUND,
            message: format!("no active queue entry {id}"),
        });
    };
    if let Some(name) = &entry.container_name {
        let _ = run_docker(vec!["rm".into(), "-f".into(), name.clone()]).await;
    } else if entry.state == "running" {
        let _ = run_docker(vec![
            "rm".into(),
            "-f".into(),
            training_container_name(&entry.slug),
        ])
        .await;
    }
    Ok(Json(json!({
        "status": "canceled",
        "id": id,
        "slug": entry.slug,
    })))
}

/// Advance the training pipeline once: reconcile any running job against its
/// live container, then launch the next queued job if the pipeline is idle.
/// Cheap and idempotent, so it is safe to call on a timer and after every
/// enqueue.
async fn dispatch_training(state: &AppState) {
    // One dispatch at a time: the enqueue kick and the periodic scheduler share
    // this, so a queued job is never launched by both at once.
    let _guard = state.dispatch_lock.lock().await;

    // 1. Reconcile: a job marked running whose container has exited is finished.
    let running = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::running_queue(&conn).unwrap_or_default()
    };
    let mut pipeline_busy = false;
    for entry in running {
        let name = entry
            .container_name
            .clone()
            .unwrap_or_else(|| training_container_name(&entry.slug));
        if container_running(&name).await {
            pipeline_busy = true;
        } else {
            // The trainer container is gone; the real outcome lives in the
            // per-slug train_status.json. Retire the queue row either way.
            let conn = state.db.lock().expect("db lock poisoned");
            let _ = db::finish_queue_entry(&conn, entry.id, "done", now_ms());
        }
    }
    if pipeline_busy {
        return;
    }

    // 2. Dispatch: launch the oldest queued job, if any.
    let next = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::next_queued(&conn).unwrap_or_default()
    };
    let Some(entry) = next else { return };

    // The project must still exist to supply its phrase; otherwise drop the job.
    let phrase = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::project_phrase(&conn, &entry.slug).ok().flatten()
    };
    let Some((phrase, _)) = phrase else {
        let conn = state.db.lock().expect("db lock poisoned");
        let _ = db::finish_queue_entry(&conn, entry.id, "failed", now_ms());
        return;
    };
    let Ok(vt) = serde_json::from_str::<ValidatedTrain>(&entry.params_json) else {
        let conn = state.db.lock().expect("db lock poisoned");
        let _ = db::finish_queue_entry(&conn, entry.id, "failed", now_ms());
        return;
    };

    match launch_train_container(state, &entry.slug, &phrase, &vt).await {
        Ok(name) => {
            let conn = state.db.lock().expect("db lock poisoned");
            let _ = db::mark_queue_running(&conn, entry.id, &name, now_ms());
        }
        Err(err) => {
            eprintln!("training dispatch failed for {}: {}", entry.slug, err.message);
            let conn = state.db.lock().expect("db lock poisoned");
            let _ = db::finish_queue_entry(&conn, entry.id, "failed", now_ms());
        }
    }
}

/// Background loop that keeps the training pipeline moving: reconcile finished
/// runs and dispatch the next queued job every few seconds.
pub(crate) async fn training_scheduler(state: AppState) {
    loop {
        dispatch_training(&state).await;
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    }
}

/// Read the training status JSON for a wake word, reconciled with whether the
/// trainer container is still alive.
/// The fixed trainer pipeline. `livekit-wakeword run` always executes these six
/// stages in order, each with a recognizable tqdm sub-progress bar in the log.
/// This is the template that lets us report per-step percentages.
const TRAIN_STEPS: [&str; 6] = [
    "Generate synthetic data",
    "Augment clips",
    "Extract features",
    "Train classifier",
    "Export to ONNX",
    "Evaluate model",
];

/// Rough share of wall-clock each stage takes, so overall percent tracks time
/// rather than step count. Training dominates. Sums to 100.
const TRAIN_STEP_WEIGHTS: [u32; 6] = [10, 12, 10, 53, 5, 10];

/// Where each stage does its heavy compute, surfaced so the app can tag each
/// step "C" (CPU) or "G" (GPU). Verified against the installed trainer source:
/// synthesis (VITS/VoxCPM via get_device()->cuda), training, and evaluation run
/// on the GPU. Feature extraction runs on CPU — the mel/embedding ONNX sessions
/// are pinned to CPUExecutionProvider. Clip augmentation (parallelized sox
/// effects) and ONNX export are CPU.
const TRAIN_STEP_DEVICES: [&str; 6] = ["gpu", "cpu", "cpu", "gpu", "cpu", "gpu"];

/// Read only the last `max_bytes` of a file — the trainer log grows to many MB,
/// and the recent tail is all we need to locate the current stage.
fn read_tail_bytes(path: &std::path::Path, max_bytes: u64) -> Option<String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = fs::File::open(path).ok()?;
    let len = f.metadata().ok()?.len();
    f.seek(SeekFrom::Start(len.saturating_sub(max_bytes))).ok()?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).ok()?;
    Some(String::from_utf8_lossy(&buf).into_owned())
}

/// Extract the last `NN%` token from a tqdm-style line, clamped to 0..=100.
fn last_percent(line: &str) -> Option<u8> {
    let bytes = line.as_bytes();
    let mut found = None;
    for i in 0..bytes.len() {
        if bytes[i] == b'%' {
            let mut j = i;
            while j > 0 && bytes[j - 1].is_ascii_digit() {
                j -= 1;
            }
            if j < i {
                if let Ok(n) = line[j..i].parse::<u16>() {
                    found = Some(n.min(100) as u8);
                }
            }
        }
    }
    found
}

/// Map a single log line to a 1-based pipeline step, via either the explicit
/// "Step N/6" banner or the tqdm description unique to each stage.
fn line_step(line: &str) -> Option<usize> {
    if let Some(pos) = line.find("Step ") {
        let rest = &line[pos + 5..];
        if let Some(slash) = rest.find("/6") {
            if let Ok(n) = rest[..slash].trim().parse::<usize>() {
                if (1..=6).contains(&n) {
                    return Some(n);
                }
            }
        }
    }
    if line.contains("Synthesizing clips")
        || line.contains("VoxCPM clips")
        || line.contains("Background (")
    {
        return Some(1);
    }
    if line.contains("Augmenting ") {
        return Some(2);
    }
    if line.contains("Features ") {
        return Some(3);
    }
    if line.contains("Phase ") {
        return Some(4);
    }
    None
}

/// The tqdm description prefix of a progress line, e.g. "Augmenting positive r0".
fn tqdm_desc(line: &str) -> String {
    if let Some(bar) = line.find("%|") {
        let head = line[..bar]
            .trim_end()
            .trim_end_matches(|c: char| c.is_ascii_digit())
            .trim_end();
        if let Some(colon) = head.rfind(':') {
            return head[..colon].trim().to_string();
        }
        return head.trim().to_string();
    }
    String::new()
}

/// Scan the trainer log tail for the current pipeline position: returns the
/// 1-based active step (0 = pre-pipeline), its percent, and the active tqdm
/// label. A later line only advances the step forward; within a step it tracks
/// the newest percent and non-empty label.
fn scan_progress_lines(text: &str) -> (usize, u8, String) {
    // Split on both '\r' (tqdm redraws in place) and '\n'.
    let lines: Vec<&str> = text.split(|c| c == '\n' || c == '\r').collect();

    let mut cur_step = 0usize; // 0 = pre-pipeline (assemble / config / setup)
    let mut cur_percent = 0u8;
    let mut active_label = String::new();
    for &line in &lines {
        if let Some(step) = line_step(line) {
            if step > cur_step {
                cur_step = step;
                cur_percent = last_percent(line).unwrap_or(0);
                active_label = tqdm_desc(line);
            } else if step == cur_step {
                if let Some(p) = last_percent(line) {
                    cur_percent = p;
                }
                let d = tqdm_desc(line);
                if !d.is_empty() {
                    active_label = d;
                }
            }
        }
    }
    (cur_step, cur_percent, active_label)
}

/// The per-step template rows (name/state/percent/device). Steps before the
/// active one are done; the active one carries `cur_percent`; later ones pend.
/// When `done`, every step reads done at 100%.
fn progress_step_states(cur_step: usize, cur_percent: u8, done: bool) -> Vec<Value> {
    TRAIN_STEPS
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let idx = i + 1;
            let (st, percent) = if done || idx < cur_step {
                ("done", 100u8)
            } else if idx == cur_step {
                ("active", cur_percent)
            } else {
                ("pending", 0u8)
            };
            json!({ "name": name, "state": st, "percent": percent, "device": TRAIN_STEP_DEVICES[i] })
        })
        .collect()
}

/// Weighted overall percent: each completed stage contributes its full weight,
/// the active stage a fraction, later stages nothing. Weighted so time tracks
/// wall-clock (training dominates) rather than raw step count.
fn progress_overall_percent(cur_step: usize, cur_percent: u8, done: bool) -> u32 {
    let total: u32 = TRAIN_STEP_WEIGHTS.iter().sum();
    let mut acc = 0u32;
    for (i, w) in TRAIN_STEP_WEIGHTS.iter().enumerate() {
        let idx = i + 1;
        if done || idx < cur_step {
            acc += w;
        } else if idx == cur_step {
            acc += w * cur_percent as u32 / 100;
        }
    }
    (acc * 100 / total).min(100)
}

/// Parse the trainer log tail into a templated, per-step progress view.
fn parse_train_progress(text: &str, run_state: &str) -> Value {
    let (mut cur_step, mut cur_percent, mut active_label) = scan_progress_lines(text);
    let done = run_state == "succeeded";
    if done {
        cur_step = 6;
        cur_percent = 100;
    }
    if active_label.is_empty() && (1..=6).contains(&cur_step) {
        active_label = TRAIN_STEPS[cur_step - 1].to_string();
    }

    let steps = progress_step_states(cur_step, cur_percent, done);
    let overall = progress_overall_percent(cur_step, cur_percent, done);

    json!({
        "steps": steps,
        "overall_percent": overall,
        "active_step": cur_step,
        "active_label": active_label,
    })
}

/// Is this train_status.json `step` a sync-server pre-launch generation phase
/// (F5 or Zonos voice-clone top-up) rather than a trainer-container stage? These
/// phases are written by the sync-server before any trainer container exists, so
/// a "running" status with no live container is expected, not a crash.
fn is_pregen_phase(step: &str) -> bool {
    matches!(step, "f5gen" | "zonosgen")
}

/// Build the base training-status body from the persisted train_status.json (if
/// any) reconciled against whether the trainer container is currently alive. A
/// file marked "running" with no live container is treated as a crash and
/// downgraded to "stopped" — UNLESS it is the pre-launch f5gen phase, which the
/// sync-server writes while it clones voice positives, before any trainer
/// container exists. With no file at all, the body is "starting" if a container
/// is up (status not yet written) or "none".
fn reconcile_status_body(file_status: Option<Value>, running: bool, slug: &str) -> Value {
    match file_status {
        Some(mut status) => {
            let step = status.get("step").and_then(Value::as_str).unwrap_or("");
            if !running
                && status.get("state").and_then(Value::as_str) == Some("running")
                && !is_pregen_phase(step)
            {
                status["state"] = json!("stopped");
                status["message"] = json!("trainer container exited before completing");
            }
            status
        }
        None => {
            if running {
                json!({ "slug": slug, "state": "starting", "message": "container launching" })
            } else {
                json!({ "slug": slug, "state": "none", "message": "no training run found" })
            }
        }
    }
}

/// The run-duration in milliseconds from start/finish wall-clock stamps (already
/// parsed to epoch ms), or None when either is missing or the finish precedes the
/// start. Pure. Extracted from `training_status`'s terminal-run bookkeeping.
fn run_duration_ms(started_ms: Option<i64>, finished_ms: Option<i64>) -> Option<i64> {
    match (started_ms, finished_ms) {
        (Some(a), Some(b)) if b >= a => Some(b - a),
        _ => None,
    }
}

/// Projected total run time in ms from the benchmark average per step and the
/// configured step count. Pure. Extracted from `training_status`'s ETA overlay.
fn estimated_total_ms(avg_ms_per_step: f64, steps: i64) -> i64 {
    (avg_ms_per_step * steps as f64) as i64
}

/// Remaining run time in ms: the projected total minus elapsed, floored at zero
/// so a run that overshoots its benchmark never reports negative time left.
/// Pure. Extracted from `training_status`'s ETA overlay.
fn remaining_ms(estimated_total_ms: i64, elapsed_ms: i64) -> i64 {
    (estimated_total_ms - elapsed_ms).max(0)
}

pub(crate) async fn training_status(
    State(state): State<AppState>,
    AxumPath(slug): AxumPath<String>,
) -> Result<Json<Value>, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request("invalid slug"));
    }
    let name = training_container_name(&slug);
    let running = container_running(&name).await;

    let status_path = state
        .models_root
        .join(&slug)
        .join("train_status.json");
    let file_status: Option<Value> = fs::read_to_string(&status_path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok());

    let mut body = reconcile_status_body(file_status, running, &slug);
    body["container_running"] = json!(running);
    body["has_model"] = json!(state
        .models_root
        .join(&slug)
        .join(format!("{slug}.onnx"))
        .exists());

    // Templated per-step progress parsed from the trainer log tail.
    {
        let run_state = body.get("state").and_then(Value::as_str).unwrap_or("none");
        let step = body.get("step").and_then(Value::as_str).unwrap_or("");
        // Skip the six-step trainer progress during a pre-launch generation phase
        // (f5gen/zonosgen): the trainer hasn't run yet, so any train.log is stale
        // from a prior run. The app renders that phase from the status body instead.
        if matches!(run_state, "running" | "starting" | "succeeded") && !is_pregen_phase(step) {
            let log_path = state.models_root.join(&slug).join("train.log");
            if let Some(text) = read_tail_bytes(&log_path, 1_048_576) {
                body["progress"] = parse_train_progress(&text, run_state);
            }
        }
    }

    // Pull the fields we need for benchmarking and ETA out of the status body.
    let run_state = body
        .get("state")
        .and_then(Value::as_str)
        .unwrap_or("none")
        .to_string();
    let steps = body.get("steps").and_then(Value::as_i64).unwrap_or(0);
    let model_size = body
        .get("model_size")
        .and_then(Value::as_str)
        .map(str::to_string);
    let personal = body
        .get("personal")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let started_at = body
        .get("started_at")
        .and_then(Value::as_str)
        .map(str::to_string);
    let updated_at = body
        .get("updated_at")
        .and_then(Value::as_str)
        .map(str::to_string);
    let started_ms = started_at.as_deref().and_then(rfc3339_ms);

    // On a terminal state, compute the run duration and persist the benchmark.
    let terminal = matches!(run_state.as_str(), "succeeded" | "failed" | "stopped");
    if terminal {
        let finished_ms = updated_at.as_deref().and_then(rfc3339_ms);
        let duration_ms = run_duration_ms(started_ms, finished_ms);
        if let Some(d) = duration_ms {
            body["duration_ms"] = json!(d);
        }
        if let Some(started) = started_at.as_deref() {
            if steps > 0 {
                let conn = state.db.lock().expect("db lock poisoned");
                let _ = db::record_training_run(
                    &conn,
                    &slug,
                    steps,
                    model_size.as_deref(),
                    personal,
                    started,
                    updated_at.as_deref(),
                    duration_ms,
                    &run_state,
                );
            }
        }

        // A succeeded run leaves a versioned, checksummed artifact folder and a
        // manifest under output/<slug>/runs/<run_id>/. Fold that into the models
        // table so every trained model is retained and queryable, and flag this
        // freshly finished run as the current live model. Idempotent on the poll.
        if run_state == "succeeded" {
            if let Some(run_id) = body.get("run_id").and_then(Value::as_str) {
                if let Some(model) =
                    read_model_manifest(&state.models_root, &slug, run_id, &body)
                {
                    let conn = state.db.lock().expect("db lock poisoned");
                    let _ = db::record_model(&conn, &model, now_ms(), true);
                }
            }
        }
    }

    // Estimate total/remaining time from the benchmark history.
    let avg = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::avg_ms_per_step(&conn, model_size.as_deref()).map_err(db_error)?
    };
    if let (Some((avg_ms_per_step, based_on)), true) = (avg, steps > 0) {
        let estimated_total = estimated_total_ms(avg_ms_per_step, steps);
        body["avg_ms_per_step"] = json!(avg_ms_per_step);
        body["based_on_runs"] = json!(based_on);
        body["estimated_total_ms"] = json!(estimated_total);
        if matches!(run_state.as_str(), "running" | "starting") {
            if let Some(started) = started_ms {
                let elapsed = (now_ms() - started).max(0);
                body["elapsed_ms"] = json!(elapsed);
                body["remaining_ms"] = json!(remaining_ms(estimated_total, elapsed));
            }
        }
    }

    // Overlay the queue: a pending job (not yet launched) reports as "queued"
    // with its position, even though the trainer has written no status file yet.
    // Done last so the terminal-run bookkeeping above still ran for any prior run.
    {
        let entry = {
            let conn = state.db.lock().expect("db lock poisoned");
            db::active_entry_for_slug(&conn, &slug).map_err(db_error)?
        };
        if let Some(entry) = entry {
            body["queue_id"] = json!(entry.id);
            // The pre-launch f5gen/zonosgen phase is a queued entry with no
            // container yet, but it is actively generating — don't overwrite it
            // with "queued".
            let pregen = is_pregen_phase(body.get("step").and_then(Value::as_str).unwrap_or(""));
            if entry.state == "queued" && !running && !pregen {
                let position = {
                    let conn = state.db.lock().expect("db lock poisoned");
                    db::queue_position(&conn, entry.id).ok().flatten()
                };
                body["state"] = json!("queued");
                body["message"] = json!("waiting in the training queue");
                body["queue_position"] = json!(position);
            }
        }
    }

    Ok(Json(body))
}

/// Tail the training log for a wake word. `?tail=N` bounds the returned lines.
pub(crate) async fn training_log(
    State(state): State<AppState>,
    AxumPath(slug): AxumPath<String>,
    RawQuery(query): RawQuery,
) -> Result<Response, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request("invalid slug"));
    }
    let params = parse_query(query.as_deref());
    let tail = params
        .get("tail")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(200)
        .clamp(1, 5000);

    let log_path = state.models_root.join(&slug).join("train.log");
    let text = fs::read_to_string(&log_path).unwrap_or_default();
    let body = if text.is_empty() {
        "(no log yet)".to_string()
    } else {
        let lines: Vec<&str> = text.lines().collect();
        let start = lines.len().saturating_sub(tail);
        lines[start..].join("\n")
    };
    Ok(([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], body).into_response())
}

/// List trained models found under the output directory, with any metrics.
pub(crate) async fn list_models(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let mut models = Vec::new();
    if let Ok(entries) = fs::read_dir(&*state.models_root) {
        let mut dirs: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_dir())
            .collect();
        dirs.sort();
        for dir in dirs {
            let Some(slug) = dir.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if !is_safe_slug(slug) {
                continue;
            }
            let onnx = dir.join(format!("{slug}.onnx"));
            if !onnx.exists() {
                continue;
            }
            let metrics: Option<Value> = fs::read_to_string(dir.join(format!("{slug}_metrics.json")))
                .ok()
                .and_then(|text| serde_json::from_str(&text).ok());
            let modified_ms = fs::metadata(&onnx)
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_millis() as i64);
            models.push(json!({
                "slug": slug,
                "modified_ms": modified_ms,
                "metrics": metrics,
            }));
        }
    }
    Ok(Json(json!({ "status": "ok", "models": models })))
}

/// Full per-run training provenance from the `models` table: every knob, the
/// real vs synthetic data counts, the code revision, checksums, scores, and the
/// complete manifest, one entry per trained model, newest first. This is the
/// accounting ledger for comparing models and knowing exactly what produced each.
pub(crate) async fn list_model_runs(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let rows = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::list_model_records(&conn).map_err(db_error)?
    };
    let runs: Vec<Value> = rows
        .iter()
        .filter_map(|s| serde_json::from_str(s).ok())
        .collect();
    Ok(Json(json!({ "status": "ok", "runs": runs })))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fully-resolved ValidatedTrain with distinctive values for argv checks.
    fn sample_vt() -> ValidatedTrain {
        ValidatedTrain {
            steps: 1234,
            model_size: "medium".to_string(),
            target_fp_per_hour: 0.2,
            personal: true,
            positive_boost: 3,
            f5_count: 7,
            zonos_count: 9,
            n_samples: Some(500),
            n_samples_val: None,
            positive_per_batch: None,
            real_samples_dir: "./data/train".to_string(),
            token_type: "end".to_string(),
            realistic: true,
            lead_probability: 0.75,
            real_lead_fraction: 0.6,
            synthetic_lead: true,
            max_lead_ms: 900,
            lead_gap_min_ms: 40,
            lead_gap_max_ms: 300,
            margin_min_ms: 100,
            margin_max_ms: 700,
            snr_min_db: 0.0,
            snr_max_db: 18.0,
            background_augment: true,
            voice_peak: 0.7,
        }
    }

    fn env_value<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
        // Args are ["-e", "KEY=VALUE", ...]; find the pair for `key`.
        args.iter()
            .zip(args.iter().skip(1))
            .find_map(|(a, b)| (a == "-e" && b.starts_with(&format!("{key}="))).then(|| &b[key.len() + 1..]))
    }

    #[test]
    fn valid_model_size_allowlist() {
        for ok in ["tiny", "small", "medium", "large"] {
            assert!(valid_model_size(ok));
        }
        assert!(!valid_model_size("MEDIUM"));
        assert!(!valid_model_size("huge"));
        assert!(!valid_model_size(""));
    }

    #[test]
    fn valid_real_samples_dir_rejects_traversal_and_bad_chars() {
        assert!(valid_real_samples_dir("./data/train"));
        assert!(valid_real_samples_dir("data/train-01_v2"));
        assert!(!valid_real_samples_dir(""));
        assert!(!valid_real_samples_dir("../etc"));
        assert!(!valid_real_samples_dir("data/train;rm"));
        assert!(!valid_real_samples_dir(&"a".repeat(129)));
    }

    #[test]
    fn last_percent_takes_final_token_clamped() {
        assert_eq!(last_percent("Augmenting: 42%|####  |"), Some(42));
        // The last percent token on the line wins.
        assert_eq!(last_percent("a 10% then 87%"), Some(87));
        assert_eq!(last_percent("over 250% impossible"), Some(100));
        assert_eq!(last_percent("no percent here"), None);
        assert_eq!(last_percent("% leading nothing"), None);
    }

    #[test]
    fn line_step_from_banner_and_tqdm_desc() {
        assert_eq!(line_step("=== Step 4/6: Train classifier"), Some(4));
        assert_eq!(line_step("Step 9/6 bogus"), None); // out of range
        assert_eq!(line_step("Synthesizing clips 3%"), Some(1));
        assert_eq!(line_step("VoxCPM clips"), Some(1));
        assert_eq!(line_step("Augmenting positive r0 5%"), Some(2));
        assert_eq!(line_step("Features 1/10"), Some(3));
        assert_eq!(line_step("Phase 2 training"), Some(4));
        assert_eq!(line_step("nothing relevant"), None);
    }

    #[test]
    fn tqdm_desc_strips_bar_and_counts() {
        assert_eq!(tqdm_desc("Augmenting positive r0: 42%|### | 3/7"), "Augmenting positive r0");
        assert_eq!(tqdm_desc("Features 12%|"), "Features");
        assert_eq!(tqdm_desc("no bar here"), "");
    }

    #[test]
    fn scan_progress_lines_advances_only_forward() {
        let log = "Step 1/6\nSynthesizing clips: 50%|## |\nStep 4/6\nPhase 1: 30%|# |\nStep 2/6 late\n";
        let (step, percent, label) = scan_progress_lines(log);
        // Reached step 4 and never regresses to the later "Step 2/6" line.
        assert_eq!(step, 4);
        assert_eq!(percent, 30);
        assert_eq!(label, "Phase 1");
    }

    #[test]
    fn progress_overall_percent_weights_stages() {
        // Nothing started.
        assert_eq!(progress_overall_percent(0, 0, false), 0);
        // done overrides everything to 100.
        assert_eq!(progress_overall_percent(3, 40, true), 100);
        // Steps 1..3 complete (10+12+10=32 weight of 100), step 4 half of 53 -> 26.
        // acc = 32 + 26 = 58.
        assert_eq!(progress_overall_percent(4, 50, false), 58);
    }

    #[test]
    fn parse_train_progress_marks_done_when_succeeded() {
        let value = parse_train_progress("", "succeeded");
        assert_eq!(value["overall_percent"], json!(100));
        assert_eq!(value["active_step"], json!(6));
        let steps = value["steps"].as_array().unwrap();
        assert_eq!(steps.len(), 6);
        assert!(steps.iter().all(|s| s["state"] == json!("done")));
    }

    #[test]
    fn reconcile_status_body_downgrades_dead_running() {
        // Running file, no live container, not f5gen -> stopped.
        let body = reconcile_status_body(
            Some(json!({ "state": "running", "step": "train" })),
            false,
            "slug",
        );
        assert_eq!(body["state"], json!("stopped"));

        // f5gen phase is preserved even without a container.
        let body = reconcile_status_body(
            Some(json!({ "state": "running", "step": "f5gen" })),
            false,
            "slug",
        );
        assert_eq!(body["state"], json!("running"));

        // No file, container up -> starting; container down -> none.
        assert_eq!(
            reconcile_status_body(None, true, "w")["state"],
            json!("starting")
        );
        assert_eq!(
            reconcile_status_body(None, false, "w")["state"],
            json!("none")
        );
    }

    #[test]
    fn build_train_argv_encodes_knobs_as_env_pairs() {
        let vt = sample_vt();
        let args = build_train_argv(
            "/host/repo",
            "trainer-slug",
            true,
            "trainer:latest",
            "slug",
            "hey computer",
            &vt,
        );
        // GPU flag present, work mount present, image is the final positional arg
        // before the bash command.
        assert!(args.windows(2).any(|w| w == ["--gpus", "all"]));
        assert!(args.contains(&"/host/repo:/work".to_string()));
        assert_eq!(env_value(&args, "SLUG"), Some("slug"));
        assert_eq!(env_value(&args, "PHRASE"), Some("hey computer"));
        assert_eq!(env_value(&args, "STEPS"), Some("1234"));
        assert_eq!(env_value(&args, "PERSONAL"), Some("1"));
        assert_eq!(env_value(&args, "F5_COUNT"), Some("7"));
        assert_eq!(env_value(&args, "ZONOS_COUNT"), Some("9"));
        assert_eq!(env_value(&args, "N_SAMPLES"), Some("500"));
        // n_samples_val is None -> no env pair emitted.
        assert_eq!(env_value(&args, "N_SAMPLES_VAL"), None);
        assert_eq!(args.last().unwrap(), "bash /work/trainer/scripts/train_job.sh");
    }

    #[test]
    fn build_train_argv_omits_gpu_and_realistic_mount_when_disabled() {
        let mut vt = sample_vt();
        vt.realistic = false;
        let args = build_train_argv(
            "/host/repo", "n", false, "img", "slug", "phrase", &vt,
        );
        assert!(!args.iter().any(|a| a == "--gpus"));
        assert!(!args.iter().any(|a| a.contains("augment_realistic.py")));
        assert_eq!(env_value(&args, "REALISTIC"), Some("0"));
    }

    #[test]
    fn queue_entry_json_folds_position_and_params() {
        let entry = db::QueueEntry {
            id: 5,
            slug: "w".to_string(),
            params_json: "{\"steps\":100}".to_string(),
            state: "queued".to_string(),
            container_name: None,
            enqueued_at_ms: 111,
            started_at_ms: None,
            finished_at_ms: None,
        };
        let value = queue_entry_json(&entry, Some(2));
        assert_eq!(value["id"], json!(5));
        assert_eq!(value["position"], json!(2));
        assert_eq!(value["params"]["steps"], json!(100));

        // Invalid params JSON falls back to an empty object, never panics.
        let bad = db::QueueEntry {
            params_json: "not json".to_string(),
            ..entry
        };
        assert_eq!(queue_entry_json(&bad, None)["params"], json!({}));
    }

    #[test]
    fn read_tail_bytes_returns_only_the_tail() {
        let mut path = std::env::temp_dir();
        path.push(format!("lkww_tail_{}.txt", now_ms()));
        fs::write(&path, "0123456789ABCDEF").unwrap();
        let tail = read_tail_bytes(&path, 4).unwrap();
        assert_eq!(tail, "CDEF");
        // Requesting more than the file length returns the whole file.
        assert_eq!(read_tail_bytes(&path, 999).unwrap(), "0123456789ABCDEF");
        let _ = fs::remove_file(&path);
        // A missing file is None, not an error.
        assert!(read_tail_bytes(&path, 4).is_none());
    }

    #[test]
    fn run_duration_ms_needs_ordered_stamps() {
        assert_eq!(run_duration_ms(Some(100), Some(400)), Some(300));
        // Equal stamps are a zero-length run, not a rejection.
        assert_eq!(run_duration_ms(Some(400), Some(400)), Some(0));
        // Finish before start (clock skew) -> None.
        assert_eq!(run_duration_ms(Some(400), Some(100)), None);
        // A missing stamp on either side -> None.
        assert_eq!(run_duration_ms(None, Some(400)), None);
        assert_eq!(run_duration_ms(Some(100), None), None);
    }

    #[test]
    fn eta_math_projects_total_and_floors_remaining() {
        // Projected total is avg-per-step times the step count, truncated to i64.
        assert_eq!(estimated_total_ms(2.0, 50_000), 100_000);
        assert_eq!(estimated_total_ms(1.5, 3), 4); // 4.5 truncates toward zero
        // Remaining is total minus elapsed while there is time left...
        assert_eq!(remaining_ms(100_000, 40_000), 60_000);
        // ...and floors at zero once a run overshoots its benchmark.
        assert_eq!(remaining_ms(100_000, 140_000), 0);
    }

    #[test]
    fn validate_train_defaults_and_clamps() {
        let vt = validate_train(serde_json::from_value(json!({})).unwrap()).unwrap();
        assert_eq!(vt.steps, 50_000);
        assert_eq!(vt.model_size, "medium");
        assert_eq!(vt.token_type, "end");
        assert_eq!(vt.positive_boost, 1);

        // Steps clamp to the [100, 500_000] window; boost clamps to [1,50].
        let vt = validate_train(
            serde_json::from_value(json!({ "steps": 5, "positive_boost": 999 })).unwrap(),
        )
        .unwrap();
        assert_eq!(vt.steps, 100);
        assert_eq!(vt.positive_boost, 50);

        // Inverted snr/gap ranges are swapped, not rejected.
        let vt = validate_train(
            serde_json::from_value(json!({ "snr_min_db": 20.0, "snr_max_db": 5.0 })).unwrap(),
        )
        .unwrap();
        assert!(vt.snr_min_db <= vt.snr_max_db);

        // A bad model size is rejected.
        assert!(validate_train(
            serde_json::from_value(json!({ "model_size": "gigantic" })).unwrap()
        )
        .is_err());
    }
}
