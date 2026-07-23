//! F5-TTS synthetic-positive generation and review.
//!
//! The trainer's built-in TTS pool doesn't sound like the user, so a second
//! synthetic source clones the user's timbre with F5-TTS zero-shot voice
//! cloning, seeded by their real positive takes. Only the sync-server can reach
//! the resident F5 container, so this module drives generation over docker,
//! stages the reference clips, lands the 16 kHz output in the slug's synth
//! bucket, tracks in-flight jobs, and serves the Review page's sample/audio/
//! generate/status/delete endpoints. `ensure_f5_positives` is the train-time
//! hook that tops the bucket up before the trainer assembles it.

use crate::db;
use crate::docker::{docker_ok, f5_container, run_docker};
use crate::error::{db_error, AppError};
use crate::state::{now_ms, AppState, SynthJob};
use crate::train::ValidatedTrain;
use crate::util::{is_safe_slug, parse_query};
use axum::{
    extract::{Path as AxumPath, RawQuery, State},
    http::header,
    response::IntoResponse,
    Json,
};
use chrono::Utc;
use serde::Serialize;
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Directory holding the F5 voice-cloned positives for a slug. The trainer
/// writes these to `<repo>/data/synth_f5/<slug>/positive`; the container mounts
/// the repo `data/` at the parent of `data_root` (DATA_ROOT=/data/real →
/// /data), so the synth bucket is a sibling of `data_root`.
fn synth_positive_dir(data_root: &Path, slug: &str) -> PathBuf {
    data_root
        .parent()
        .unwrap_or(data_root)
        .join("synth_f5")
        .join(slug)
        .join("positive")
}

#[derive(Serialize)]
pub(crate) struct SyntheticSample {
    id: String,
    file_name: String,
    text: String,
}

#[derive(Serialize)]
pub(crate) struct SyntheticSamplesResponse {
    slug: String,
    phrase: String,
    total: usize,
    sampled: usize,
    samples: Vec<SyntheticSample>,
}

/// Return a representative sample of the F5 synthetic positives so the Review
/// page can spot-check them by ear. With potentially thousands of clips we take
/// an evenly-spaced stride across the sorted batch (start, middle, end), not the
/// first N — deterministic, no RNG. The clips are all the wake phrase, so each
/// sample carries the project phrase as its label text.
/// Delete every F5 synthetic positive for a wake word (the whole
/// data/synth_f5/<slug> tree). Synth clips are filesystem-only with no DB rows,
/// so this is a plain directory removal. Refuses while a generation run is in
/// flight so we don't yank the directory out from under it.
pub(crate) async fn delete_synth(
    State(state): State<AppState>,
    AxumPath(slug): AxumPath<String>,
) -> Result<Json<Value>, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request(format!("unsafe wake word slug: {slug}")));
    }
    {
        let jobs = state.synth_jobs.lock().expect("synth jobs lock poisoned");
        if jobs.get(&slug).map(|j| j.running).unwrap_or(false) {
            return Err(AppError::bad_request(
                "a generation run is in progress; wait for it to finish before deleting",
            ));
        }
    }
    // The synth bucket is data/synth_f5/<slug>/positive; remove its <slug> parent
    // so nothing for this wake word is left behind.
    let slug_dir = synth_positive_dir(&state.data_root, &slug)
        .parent()
        .map(Path::to_path_buf);
    let removed = match slug_dir {
        Some(dir) if dir.exists() => {
            let count = count_wavs(&synth_positive_dir(&state.data_root, &slug));
            fs::remove_dir_all(&dir)
                .map_err(|e| AppError::internal(format!("failed to delete synth dir: {e}")))?;
            count
        }
        _ => 0,
    };
    Ok(Json(json!({ "status": "ok", "slug": slug, "deleted": removed })))
}

pub(crate) async fn synthetic_samples(
    State(state): State<AppState>,
    AxumPath(slug): AxumPath<String>,
) -> Result<Json<SyntheticSamplesResponse>, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request(format!("unsafe wake word slug: {slug}")));
    }
    let dir = synth_positive_dir(&state.data_root, &slug);
    let mut files: Vec<String> = Vec::new();
    if dir.is_dir() {
        for entry in fs::read_dir(&dir)? {
            let name = entry?.file_name().to_string_lossy().to_string();
            if name.ends_with(".wav") {
                files.push(name);
            }
        }
    }
    files.sort();
    let total = files.len();

    const MAX_SAMPLES: usize = 24;
    let sampled: Vec<String> = if total <= MAX_SAMPLES {
        files
    } else {
        let stride = total as f64 / MAX_SAMPLES as f64;
        (0..MAX_SAMPLES)
            .map(|i| files[((i as f64) * stride) as usize].clone())
            .collect()
    };

    let phrase = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::project_phrase(&conn, &slug)
            .map_err(db_error)?
            .map(|(phrase, _)| phrase)
            .unwrap_or_default()
    };

    let samples = sampled
        .iter()
        .map(|name| SyntheticSample {
            id: name.trim_end_matches(".wav").to_string(),
            file_name: name.clone(),
            text: phrase.clone(),
        })
        .collect();

    Ok(Json(SyntheticSamplesResponse {
        slug,
        phrase,
        total,
        sampled: sampled.len(),
        samples,
    }))
}

/// Serve one F5 synthetic positive WAV by file name. Mirrors `review_audio`.
pub(crate) async fn synthetic_audio(
    State(state): State<AppState>,
    AxumPath((slug, file_name)): AxumPath<(String, String)>,
) -> Result<impl IntoResponse, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request(format!("unsafe wake word slug: {slug}")));
    }
    if file_name.contains('/')
        || file_name.contains('\\')
        || file_name.contains("..")
        || !file_name.ends_with(".wav")
    {
        return Err(AppError::bad_request(format!("unsafe file name: {file_name}")));
    }
    let path = synth_positive_dir(&state.data_root, &slug).join(&file_name);
    let bytes = fs::read(path)?;
    Ok(([(header::CONTENT_TYPE, "audio/wav")], bytes))
}

/// Does this directory hold at least one `.wav`?
fn dir_has_wav(dir: &Path) -> bool {
    fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .any(|e| e.file_name().to_string_lossy().ends_with(".wav"))
        })
        .unwrap_or(false)
}

/// How many `*.wav` files a directory holds (0 if missing/unreadable).
fn count_wavs(dir: &Path) -> usize {
    fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_string_lossy().ends_with(".wav"))
                .count()
        })
        .unwrap_or(0)
}

/// Resolve the F5 cloning-reference directory for a slug: the user's real
/// positive clips, whose transcript is exactly the phrase and whose length is
/// F5-friendly. Enrollment has been retired as a reference — a long passage
/// starved and leaked its own tail text into the output ("warm gold"), so F5
/// now only ever clones from real positives.
fn resolve_synth_refs(data_root: &Path, slug: &str) -> Result<PathBuf, AppError> {
    let positive_dir = data_root.join(slug).join("positive");
    if dir_has_wav(&positive_dir) {
        Ok(positive_dir)
    } else {
        Err(AppError::bad_request(
            "no positive clips to clone from; record positive takes first",
        ))
    }
}

/// Drive one F5 batch for a slug: resolve its reference clips, run the resident
/// generator, and land the resampled clips in the synth bucket. Returns how many
/// clips this run produced. Shared by the manual `/synth/generate` endpoint and
/// the train-time pre-generation step.
async fn generate_synth_batch(
    state: &AppState,
    slug: &str,
    phrase: &str,
    count: usize,
) -> Result<usize, AppError> {
    let refs = resolve_synth_refs(&state.data_root, slug)?;
    let synth_dir = synth_positive_dir(&state.data_root, slug);
    let cp_refs = refs.to_string_lossy().to_string();
    let cp_out = synth_dir.to_string_lossy().to_string();
    let cp_gen_py = "/trainer/scripts/f5_gen_positives.py".to_string();
    run_synth_generation(
        slug, phrase, count, &refs, &cp_refs, &cp_gen_py, &cp_out, &synth_dir,
    )
    .await
}

/// Ensure the F5 synth bucket for `slug` holds `vt.f5_count` voice-cloned
/// positives before the trainer assembles them, generating a fresh batch when it
/// is short. Reuses an existing batch that is already large enough (so repeated
/// trains don't re-pay F5's cost); otherwise clears the bucket and regenerates
/// exactly `f5_count` (the generator names clips `f5_00000..`, so a clean bucket
/// yields exactly the count the user asked for with no stale leftovers). Writes
/// an `f5gen` phase into train_status.json so the app's training screen shows it.
/// Non-fatal: on any failure it logs, records the error, and returns whatever
/// clips exist so training still proceeds. Returns the final clip count.
pub(crate) async fn ensure_f5_positives(
    state: &AppState,
    slug: &str,
    phrase: &str,
    vt: &ValidatedTrain,
) -> usize {
    let target = vt.f5_count as usize;
    let synth_dir = synth_positive_dir(&state.data_root, slug);
    if target == 0 {
        return count_wavs(&synth_dir);
    }
    let existing = count_wavs(&synth_dir);
    if existing >= target {
        return existing;
    }
    // Can we clone at all? Resolve references before touching the bucket so a
    // slug with no positives keeps whatever clips it already has.
    if let Err(e) = resolve_synth_refs(&state.data_root, slug) {
        eprintln!("f5 pregen: no reference clips for {slug}: {}", e.message);
        return existing;
    }
    // Don't collide with a manual generation already running for this slug.
    {
        let mut jobs = state.synth_jobs.lock().expect("synth jobs lock poisoned");
        if jobs.get(slug).map(|j| j.running).unwrap_or(false) {
            eprintln!(
                "f5 pregen: a manual generation is already running for {slug}; \
                 training on the {existing} existing clips"
            );
            return existing;
        }
        jobs.insert(
            slug.to_string(),
            SynthJob {
                running: true,
                requested: target,
                wrote: existing,
                error: None,
            },
        );
    }
    write_f5_status(
        state,
        slug,
        phrase,
        vt,
        target,
        0,
        &format!("generating {target} voice-cloned positives (F5)"),
    );
    // Fresh batch: clear the bucket so the count is exactly `target` (no stale
    // clips from a previous, larger run linger).
    let _ = fs::remove_dir_all(&synth_dir);
    let result = generate_synth_batch(state, slug, phrase, target).await;
    let wrote = *result.as_ref().unwrap_or(&0);
    if let Err(e) = &result {
        eprintln!("f5 pregen failed for {slug}: {}", e.message);
    }
    {
        let mut jobs = state.synth_jobs.lock().expect("synth jobs lock poisoned");
        if let Some(j) = jobs.get_mut(slug) {
            j.running = false;
            j.wrote = wrote;
            j.error = result.as_ref().err().map(|e| e.message.clone());
        }
    }
    let msg = match &result {
        Ok(w) => format!("generated {w} voice-cloned positives (F5)"),
        Err(e) => format!(
            "F5 generation failed: {}; training on existing clips",
            e.message
        ),
    };
    write_f5_status(state, slug, phrase, vt, target, wrote, &msg);
    count_wavs(&synth_dir)
}

/// Write the pre-training `f5gen` phase into a slug's train_status.json. The
/// trainer overwrites this file with its own phases once it launches; until then
/// the app sees a `running` status stepped `f5gen`, with the F5 clip counts.
fn write_f5_status(
    state: &AppState,
    slug: &str,
    phrase: &str,
    vt: &ValidatedTrain,
    requested: usize,
    wrote: usize,
    message: &str,
) {
    let dir = state.models_root.join(slug);
    if fs::create_dir_all(&dir).is_err() {
        return;
    }
    let now = Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
    let body = json!({
        "slug": slug,
        "phrase": phrase,
        "state": "running",
        "step": "f5gen",
        "exit_code": 0,
        "message": message,
        "steps": vt.steps,
        "model_size": vt.model_size,
        "personal": vt.personal,
        "f5_requested": requested,
        "f5_wrote": wrote,
        "started_at": now,
        "updated_at": now,
    });
    let _ = fs::write(dir.join("train_status.json"), body.to_string());
}

#[derive(Serialize)]
pub(crate) struct GenerateSynthResponse {
    status: &'static str,
    slug: String,
    requested: usize,
}

/// Kick off an F5 voice-cloned positive batch for a wake word, seeded by the
/// user's real positive takes. Runs the resident F5 model inside the speech-f5tts
/// container over the mounted docker socket, then copies the 16 kHz clips into the
/// slug's synth bucket. Returns immediately; progress is polled via the status
/// endpoint.
pub(crate) async fn generate_synth(
    State(state): State<AppState>,
    AxumPath(slug): AxumPath<String>,
    RawQuery(query): RawQuery,
) -> Result<Json<GenerateSynthResponse>, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request(format!("unsafe wake word slug: {slug}")));
    }
    let params = parse_query(query.as_deref());
    let count: usize = params
        .get("count")
        .and_then(|v| v.parse().ok())
        .unwrap_or(60)
        .clamp(1, 1000);

    // Clone from the user's SHORT real positive clips: their transcript is exactly
    // the phrase and their length is F5-friendly. (F5 sizes the spoken output from
    // the reference's rate, so a long passage starves the short wake phrase and
    // leaks its own text.) Refuse early if there is nothing to clone from.
    let refs_server_dir = resolve_synth_refs(&state.data_root, &slug)?;

    let phrase = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::project_phrase(&conn, &slug)
            .map_err(db_error)?
            .map(|(phrase, _)| phrase)
            .unwrap_or_default()
    };
    if phrase.trim().is_empty() {
        return Err(AppError::bad_request(
            "unknown wake phrase for this slug",
        ));
    }

    {
        let mut jobs = state.synth_jobs.lock().expect("synth jobs lock poisoned");
        if jobs.get(&slug).map(|j| j.running).unwrap_or(false) {
            return Err(AppError::bad_request(
                "a generation run is already in progress for this wake word",
            ));
        }
        jobs.insert(
            slug.clone(),
            SynthJob {
                running: true,
                requested: count,
                wrote: 0,
                error: None,
            },
        );
    }

    // These paths feed `docker cp`, whose local side is resolved inside THIS
    // container, so they must be container paths (the repo `data/` is mounted at
    // /data, trainer scripts at /trainer), not host paths.
    let synth_server_dir = synth_positive_dir(&state.data_root, &slug);
    let cp_refs = refs_server_dir.to_string_lossy().to_string();
    let cp_gen_py = "/trainer/scripts/f5_gen_positives.py".to_string();
    let cp_out = synth_server_dir.to_string_lossy().to_string();

    let task_state = state.clone();
    let task_slug = slug.clone();
    let task_phrase = phrase.clone();
    tokio::spawn(async move {
        let result = run_synth_generation(
            &task_slug,
            &task_phrase,
            count,
            &refs_server_dir,
            &cp_refs,
            &cp_gen_py,
            &cp_out,
            &synth_server_dir,
        )
        .await;
        let mut jobs = task_state.synth_jobs.lock().expect("synth jobs lock poisoned");
        let entry = jobs.entry(task_slug).or_insert(SynthJob {
            running: false,
            requested: count,
            wrote: 0,
            error: None,
        });
        entry.running = false;
        match result {
            Ok(wrote) => {
                entry.wrote = wrote;
                entry.error = None;
            }
            Err(e) => entry.error = Some(e.message),
        }
    });

    Ok(Json(GenerateSynthResponse {
        status: "ok",
        slug,
        requested: count,
    }))
}

/// Drive one F5 batch end to end: stage ≤8 references into the F5 container, run
/// the resident generator (writing 16 kHz clips), copy them into the synth
/// bucket, and clean up. Returns how many clips this run produced. Every path
/// handed to `docker cp` is a path INSIDE this sync-server container, since the
/// docker CLI resolves the local side of a cp against its own filesystem — the
/// repo `data/` is mounted at /data and trainer scripts at /trainer.
#[allow(clippy::too_many_arguments)]
async fn run_synth_generation(
    slug: &str,
    phrase: &str,
    count: usize,
    refs_server_dir: &Path,
    cp_refs: &str,
    cp_gen_py: &str,
    cp_out: &str,
    synth_server_dir: &Path,
) -> Result<usize, AppError> {
    let container = f5_container();
    let stamp = now_ms();
    let scratch = format!("/tmp/f5gen_{slug}_{stamp}");
    let crefs = format!("{scratch}/refs");
    let cout = format!("{scratch}/out");

    docker_ok(vec![
        "exec".into(),
        container.clone(),
        "mkdir".into(),
        "-p".into(),
        crefs.clone(),
        cout.clone(),
    ])
    .await?;

    // Stage a small, clean subset of references (rotated inside the python).
    let mut names: Vec<String> = fs::read_dir(refs_server_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.ends_with(".wav"))
        .collect();
    names.sort();
    names.truncate(8);
    if names.is_empty() {
        return Err(AppError::internal("no reference wavs to stage"));
    }
    for name in &names {
        docker_ok(vec![
            "cp".into(),
            format!("{cp_refs}/{name}"),
            format!("{container}:{crefs}/{name}"),
        ])
        .await?;
        // An enrollment reference ships its exact passage in a sibling .txt; stage
        // it too so F5 gets the right ref_text.
        let sidecar = format!("{}.txt", name.trim_end_matches(".wav"));
        if refs_server_dir.join(&sidecar).is_file() {
            docker_ok(vec![
                "cp".into(),
                format!("{cp_refs}/{sidecar}"),
                format!("{container}:{crefs}/{sidecar}"),
            ])
            .await?;
        }
    }

    // Stage and run the resident-model generator, writing 16 kHz clips directly.
    docker_ok(vec![
        "cp".into(),
        cp_gen_py.to_string(),
        format!("{container}:{scratch}/gen.py"),
    ])
    .await?;
    docker_ok(vec![
        "exec".into(),
        container.clone(),
        "python3".into(),
        format!("{scratch}/gen.py"),
        "--refs-dir".into(),
        crefs.clone(),
        "--ref-text".into(),
        phrase.to_string(),
        "--gen-text".into(),
        phrase.to_string(),
        "--out-dir".into(),
        cout.clone(),
        "--count".into(),
        count.to_string(),
        "--out-sr".into(),
        "16000".into(),
        // Fidelity knobs (F5 defaults unless overridden in the environment).
        // Raise F5_NFE_STEP for a sharper, more faithful render (slower);
        // raise F5_CFG_STRENGTH to hew closer to the user's timbre.
        "--nfe-step".into(),
        env::var("F5_NFE_STEP").unwrap_or_else(|_| "32".to_string()),
        "--cfg-strength".into(),
        env::var("F5_CFG_STRENGTH").unwrap_or_else(|_| "2.0".to_string()),
    ])
    .await?;

    // Copy the finished clips into the synth bucket (dir must exist first).
    fs::create_dir_all(synth_server_dir)?;
    docker_ok(vec![
        "cp".into(),
        format!("{container}:{cout}/."),
        format!("{cp_out}/"),
    ])
    .await?;

    // Count what this run produced from the container's output dir.
    let counted = docker_ok(vec![
        "exec".into(),
        container.clone(),
        "sh".into(),
        "-c".into(),
        format!("ls -1 {cout}/*.wav 2>/dev/null | wc -l"),
    ])
    .await?;
    let wrote = String::from_utf8_lossy(&counted.stdout)
        .trim()
        .parse::<usize>()
        .unwrap_or(0);

    // Best-effort scratch cleanup.
    let _ = run_docker(vec![
        "exec".into(),
        container,
        "rm".into(),
        "-rf".into(),
        scratch,
    ])
    .await;

    Ok(wrote)
}

#[derive(Serialize)]
pub(crate) struct GenerateSynthStatusResponse {
    slug: String,
    running: bool,
    requested: usize,
    wrote: usize,
    error: Option<String>,
    idle: bool,
}

/// Report the state of a slug's F5 generation run for the app to poll.
pub(crate) async fn generate_synth_status(
    State(state): State<AppState>,
    AxumPath(slug): AxumPath<String>,
) -> Result<Json<GenerateSynthStatusResponse>, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request(format!("unsafe wake word slug: {slug}")));
    }
    let job = {
        let jobs = state.synth_jobs.lock().expect("synth jobs lock poisoned");
        jobs.get(&slug).cloned()
    };
    Ok(Json(match job {
        Some(j) => GenerateSynthStatusResponse {
            slug,
            running: j.running,
            requested: j.requested,
            wrote: j.wrote,
            error: j.error,
            idle: false,
        },
        None => GenerateSynthStatusResponse {
            slug,
            running: false,
            requested: 0,
            wrote: 0,
            error: None,
            idle: true,
        },
    }))
}
