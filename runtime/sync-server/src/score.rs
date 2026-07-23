//! Model-test scoring: replaying stored takes through the scorer service.
//!
//! Synthetic eval recall hides the streaming gap, so the honest diagnostic is to
//! run a real stored take through the trained model and check whether the wake
//! phrase actually fires where the transcript says it was spoken. This module
//! owns the scorer HTTP client, the rolling-curve → target/false-positive
//! reduction, curve/grade caching keyed on a model fingerprint, and the
//! model-run manifest reader used to record finished trainings.

use crate::constants::MAX_DRIFT_MS;
use crate::db;
use crate::error::{db_error, AppError};
use crate::state::{now_ms, AppState};
use crate::util::{is_safe_recording_id, is_safe_run_id, is_safe_slug, parse_query, safe_filename};
use crate::whisper::normalize_token;
use axum::{
    extract::{Path as AxumPath, RawQuery, State},
    http::HeaderMap,
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::Path;

/// Rolling wake-word detection over a stored bulk recording, with each true
/// target-phrase utterance (located from the current Whisper transcript) tagged
/// with the model's peak score in its window. The full curve is returned so a
/// client can re-threshold the detection/false-positive counts for free.
#[derive(Debug, Serialize)]
pub(crate) struct ScoreResponse {
    status: &'static str,
    wake_word_slug: String,
    source_recording: String,
    phrase: String,
    /// The archived run id scored, or null when the current model was used.
    run: Option<String>,
    mode: String,
    window_ms: u64,
    step_ms: u64,
    keep_ms: u64,
    duration_ms: f64,
    threshold: f64,
    times_ms: Vec<f64>,
    scores: Vec<f64>,
    targets: Vec<ScoreTarget>,
    /// Counts at `threshold`: targets that peaked at/above it, targets that did
    /// not, and above-threshold detections landing outside any target window.
    true_positives: usize,
    false_negatives: usize,
    false_positives: usize,
}

/// One occurrence of the trigger phrase in the transcript, with the model's
/// best score anywhere in the detection window aligned to the phrase tail.
#[derive(Debug, Serialize)]
pub(crate) struct ScoreTarget {
    text: String,
    start_ms: i64,
    end_ms: i64,
    peak_score: f64,
    peak_time_ms: f64,
    detected: bool,
}

/// One test take's stored grade against a model, as returned to the app. Mirrors
/// `db::ScoreGrade` with the recording id kept so the app can match it to a card.
#[derive(Debug, Serialize)]
pub(crate) struct ScoreGradeItem {
    recording_id: String,
    run: Option<String>,
    threshold: f64,
    peak_score: f64,
    max_score: f64,
    has_target: bool,
    target_count: i64,
    true_positives: i64,
    false_negatives: i64,
    false_positives: i64,
    detected: bool,
    scored_at_ms: i64,
}

impl From<db::ScoreGrade> for ScoreGradeItem {
    fn from(g: db::ScoreGrade) -> Self {
        Self {
            recording_id: g.recording_id,
            run: g.run_id,
            threshold: g.threshold,
            peak_score: g.peak_score,
            max_score: g.max_score,
            has_target: g.has_target,
            target_count: g.target_count,
            true_positives: g.true_positives,
            false_negatives: g.false_negatives,
            false_positives: g.false_positives,
            detected: g.detected,
            scored_at_ms: g.created_at_ms,
        }
    }
}

/// Model-wide totals across all graded test takes: how many takes exist, how
/// many are graded against this model, and the summed target/miss/false-positive
/// counts that headline the Model test view.
#[derive(Debug, Serialize)]
pub(crate) struct GradeTotals {
    test_takes: usize,
    graded: usize,
    targets: i64,
    true_positives: i64,
    false_negatives: i64,
    false_positives: i64,
    detections: usize,
}

/// The Model test view's grade payload: per-take grades plus the model totals,
/// for one model (`run` null = current) in one detection mode.
#[derive(Debug, Serialize)]
pub(crate) struct ModelGradesResponse {
    status: &'static str,
    wake_word_slug: String,
    run: Option<String>,
    model_fp: String,
    mode: String,
    totals: GradeTotals,
    grades: Vec<ScoreGradeItem>,
}

/// The scorer service's `/score` reply: parallel time/score arrays. The scorer
/// emits fractional milliseconds, so times are floats.
#[derive(Debug, Deserialize)]
pub(crate) struct ScorerCurve {
    #[serde(default)]
    duration_ms: f64,
    #[serde(default)]
    times_ms: Vec<f64>,
    #[serde(default)]
    scores: Vec<f64>,
}

/// Resolve the scorer service URL from the request header, falling back to the
/// SCORER_SERVER_URL environment variable captured at startup.
fn resolve_scorer_url(state: &AppState, headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-scorer-server-url")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| state.scorer_url.as_ref().clone())
}

/// Score a stored bulk recording against a trained model and overlay the
/// current Whisper transcript so each true target utterance is scored honestly.
/// This is the model-test diagnostic: synthetic eval recall hides the streaming
/// gap, so we replay real takes and check whether the phrase actually fires
/// mid-sentence. Query params: `mode` (full|reset), `step_ms`, `keep_ms`,
/// `threshold`.
pub(crate) async fn score_recording(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath((slug, recording_id)): AxumPath<(String, String)>,
    RawQuery(query): RawQuery,
) -> Result<Json<ScoreResponse>, AppError> {
    if !is_safe_slug(&slug) || !is_safe_recording_id(&recording_id) {
        return Err(AppError::bad_request("unsafe score path"));
    }
    let params = parse_query(query.as_deref());
    let mode = match params.get("mode").map(String::as_str) {
        Some("full") | None => "full",
        Some("reset") => "reset",
        Some(other) => return Err(AppError::bad_request(format!("unknown mode: {other}"))),
    };
    let step_ms = params
        .get("step_ms")
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v >= 1)
        .unwrap_or(if mode == "reset" { 40 } else { 10 });
    let keep_ms = params
        .get("keep_ms")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(700);
    let threshold = params
        .get("threshold")
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .unwrap_or(0.5);
    // Optional archived run id: score a specific past model version instead of
    // the mutable current model, so retraining never overwrites a run's scores.
    let run = match params.get("run").map(String::as_str) {
        Some(r) if is_safe_run_id(r) => Some(r.to_string()),
        Some(_) => return Err(AppError::bad_request("unsafe run id")),
        None => None,
    };

    let scorer_url = resolve_scorer_url(&state, &headers).ok_or_else(|| {
        AppError::bad_request("scorer not configured; set SCORER_SERVER_URL or x-scorer-server-url")
    })?;
    let force = params.get("nocache").map(String::as_str) == Some("1");

    let response = compute_score(
        &state,
        &scorer_url,
        &slug,
        &recording_id,
        mode,
        step_ms,
        keep_ms,
        threshold,
        run.as_deref(),
        force,
    )
    .await?;
    Ok(Json(response))
}

/// Score one stored recording against a model, cache its curve, persist the
/// derived grade (so the take's score on this model survives without re-running
/// the scorer), and return the full response. Shared by the single-recording
/// endpoint and the bulk "score all test takes" endpoint. `force` bypasses the
/// cached curve (the `nocache=1` path).
#[allow(clippy::too_many_arguments)]
async fn compute_score(
    state: &AppState,
    scorer_url: &str,
    slug: &str,
    recording_id: &str,
    mode: &str,
    step_ms: u64,
    keep_ms: u64,
    threshold: f64,
    run: Option<&str>,
    force: bool,
) -> Result<ScoreResponse, AppError> {
    let (phrase, words) = {
        let conn = state.db.lock().expect("db lock poisoned");
        let phrase = db::project_phrase(&conn, slug)
            .map_err(db_error)?
            .map(|(phrase, _external_id)| phrase)
            .unwrap_or_else(|| slug.replace('_', " "));
        let words = db::current_transcript_words(&conn, recording_id).map_err(db_error)?;
        (phrase, words)
    };

    // Fingerprint the trained model so a cached curve is reused only while the
    // model that produced it is unchanged; retraining flips the fingerprint and
    // the next request re-scores. An archived run is immutable, so its id alone
    // is a stable fingerprint and keeps each run's cached curves distinct from
    // the current model's and from each other.
    let model_fp = resolve_model_fp(&state.models_root, slug, run);

    let cached = if force {
        None
    } else {
        let conn = state.db.lock().expect("db lock poisoned");
        db::cached_score_curve(&conn, recording_id, mode, step_ms, keep_ms, &model_fp)
            .map_err(db_error)?
    };

    let curve = if let Some((duration_ms, times_ms, scores)) = cached {
        ScorerCurve { duration_ms, times_ms, scores }
    } else {
        let wav_path = state
            .data_root
            .join(slug)
            .join("bulk_source")
            .join(format!("{}.wav", safe_filename(recording_id)));
        let wav = fs::read(&wav_path)
            .map_err(|_| AppError::bad_request(format!("no source audio for {recording_id}")))?;

        let curve = run_scorer(scorer_url, wav, slug, run, mode, step_ms, keep_ms).await?;
        let conn = state.db.lock().expect("db lock poisoned");
        db::store_score_curve(
            &conn,
            recording_id,
            mode,
            step_ms,
            keep_ms,
            &model_fp,
            curve.duration_ms,
            &curve.times_ms,
            &curve.scores,
            now_ms(),
        )
        .map_err(db_error)?;
        curve
    };

    let targets = locate_targets(&phrase, &words, &curve, threshold);
    let true_positives = targets.iter().filter(|t| t.detected).count();
    let false_negatives = targets.len() - true_positives;
    let false_positives = count_false_positives(&curve, &targets, threshold);

    // A representative single score for the take: the peak over located target
    // phrases, or the whole-curve maximum when the take has no target phrase (so
    // a negative take still shows how hot the model got). Persist the grade.
    let max_score = curve.scores.iter().cloned().fold(0.0_f64, f64::max);
    let has_target = !targets.is_empty();
    let target_peak = targets.iter().map(|t| t.peak_score).fold(0.0_f64, f64::max);
    let peak_score = if has_target { target_peak } else { max_score };
    {
        let conn = state.db.lock().expect("db lock poisoned");
        db::store_score_grade(
            &conn,
            &model_fp,
            mode,
            &db::ScoreGrade {
                recording_id: recording_id.to_string(),
                run_id: run.map(str::to_string),
                threshold,
                peak_score,
                max_score,
                has_target,
                target_count: targets.len() as i64,
                true_positives: true_positives as i64,
                false_negatives: false_negatives as i64,
                false_positives: false_positives as i64,
                detected: peak_score >= threshold,
                created_at_ms: now_ms(),
            },
        )
        .map_err(db_error)?;
    }

    Ok(ScoreResponse {
        status: "ok",
        wake_word_slug: slug.to_string(),
        source_recording: recording_id.to_string(),
        phrase,
        run: run.map(str::to_string),
        mode: mode.to_string(),
        window_ms: 2000,
        step_ms,
        keep_ms,
        duration_ms: curve.duration_ms,
        threshold,
        times_ms: curve.times_ms,
        scores: curve.scores,
        targets,
        true_positives,
        false_negatives,
        false_positives,
    })
}

/// Query params shared by the score endpoints: detection mode and its scan
/// params plus the threshold and optional archived run id. Parsed once so the
/// single and bulk endpoints agree on defaults.
struct ScoreParams {
    mode: &'static str,
    step_ms: u64,
    keep_ms: u64,
    threshold: f64,
    run: Option<String>,
}

fn parse_score_params(query: Option<&str>) -> Result<ScoreParams, AppError> {
    let params = parse_query(query);
    let mode = match params.get("mode").map(String::as_str) {
        Some("full") | None => "full",
        Some("reset") => "reset",
        Some(other) => return Err(AppError::bad_request(format!("unknown mode: {other}"))),
    };
    let step_ms = params
        .get("step_ms")
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v >= 1)
        .unwrap_or(if mode == "reset" { 40 } else { 10 });
    let keep_ms = params
        .get("keep_ms")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(700);
    let threshold = params
        .get("threshold")
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .unwrap_or(0.5);
    let run = match params.get("run").map(String::as_str) {
        Some(r) if is_safe_run_id(r) => Some(r.to_string()),
        Some(_) => return Err(AppError::bad_request("unsafe run id")),
        None => None,
    };
    Ok(ScoreParams { mode, step_ms, keep_ms, threshold, run })
}

/// Score every test take for a wake word against one model in a single request,
/// so the Model test view can grade the whole set without the app firing a
/// score per take. Runs the takes one at a time (the scorer is a single-GPU
/// service) and returns the resulting grades plus the model totals.
pub(crate) async fn score_all_test_takes(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(slug): AxumPath<String>,
    RawQuery(query): RawQuery,
) -> Result<Json<ModelGradesResponse>, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request("unsafe score path"));
    }
    let sp = parse_score_params(query.as_deref())?;
    let scorer_url = resolve_scorer_url(&state, &headers).ok_or_else(|| {
        AppError::bad_request("scorer not configured; set SCORER_SERVER_URL or x-scorer-server-url")
    })?;
    let ids = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::test_take_ids(&conn, &slug).map_err(db_error)?
    };
    for id in &ids {
        // A take with no source audio or transcript would fail; skip it so one
        // bad take can't sink the whole batch.
        if let Err(error) = compute_score(
            &state,
            &scorer_url,
            &slug,
            id,
            sp.mode,
            sp.step_ms,
            sp.keep_ms,
            sp.threshold,
            sp.run.as_deref(),
            false,
        )
        .await
        {
            eprintln!("score_all: skipping {id}: {}", error.message);
        }
    }
    grades_response(&state, &slug, sp.mode, sp.run.as_deref())
}

/// The stored grades for a model's test takes plus the model totals, read back
/// without re-running the scorer. Backs the Model test view's per-take cards and
/// its misses / false-positive statistics.
pub(crate) async fn model_test_grades(
    State(state): State<AppState>,
    AxumPath(slug): AxumPath<String>,
    RawQuery(query): RawQuery,
) -> Result<Json<ModelGradesResponse>, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request("unsafe score path"));
    }
    let sp = parse_score_params(query.as_deref())?;
    grades_response(&state, &slug, sp.mode, sp.run.as_deref())
}

/// Assemble the grades response: every stored grade for this model plus totals.
fn grades_response(
    state: &AppState,
    slug: &str,
    mode: &str,
    run: Option<&str>,
) -> Result<Json<ModelGradesResponse>, AppError> {
    let model_fp = resolve_model_fp(&state.models_root, slug, run);
    let (test_takes, grades) = {
        let conn = state.db.lock().expect("db lock poisoned");
        let ids = db::test_take_ids(&conn, slug).map_err(db_error)?;
        let grades = db::grades_for_model(&conn, slug, &model_fp, mode).map_err(db_error)?;
        (ids.len(), grades)
    };
    let grades: Vec<ScoreGradeItem> = grades.into_iter().map(ScoreGradeItem::from).collect();
    let totals = GradeTotals {
        test_takes,
        graded: grades.len(),
        targets: grades.iter().map(|g| g.target_count).sum(),
        true_positives: grades.iter().map(|g| g.true_positives).sum(),
        false_negatives: grades.iter().map(|g| g.false_negatives).sum(),
        false_positives: grades.iter().map(|g| g.false_positives).sum(),
        detections: grades.iter().filter(|g| g.detected).count(),
    };
    Ok(Json(ModelGradesResponse {
        status: "ok",
        wake_word_slug: slug.to_string(),
        run: run.map(str::to_string),
        model_fp,
        mode: mode.to_string(),
        totals,
        grades,
    }))
}

/// The cache-key fingerprint for the model being scored: an archived run's id is
/// itself immutable (so `run-<id>` is stable and distinct per run), while the
/// mutable current model is fingerprinted from its `.onnx` on disk. Shared by the
/// scoring and grade-listing paths so both key their caches identically.
fn resolve_model_fp(models_root: &Path, slug: &str, run: Option<&str>) -> String {
    match run {
        Some(r) => format!("run-{r}"),
        None => model_fingerprint(models_root, slug),
    }
}

/// Fingerprint the trained `.onnx` for a wake word so cached score curves stay
/// valid only while that exact model file is in place. Uses size + modified
/// time — cheap and flips on any retrain/export. Missing model or unreadable
/// metadata yields a stable sentinel so caching still works but never masks a
/// swapped model (a later present model changes the fingerprint).
fn model_fingerprint(models_root: &Path, slug: &str) -> String {
    let path = models_root.join(slug).join(format!("{slug}.onnx"));
    let Ok(meta) = fs::metadata(&path) else {
        return "nomodel".to_string();
    };
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{}-{}", meta.len(), mtime)
}

/// Read a finished run's manifest (and its metrics sidecar) into a `ModelRecord`,
/// merging in the fields already parsed from the training status body. Returns
/// None if the run id is unsafe or the manifest is missing/unreadable.
pub(crate) fn read_model_manifest(
    models_root: &Path,
    slug: &str,
    run_id: &str,
    status: &Value,
) -> Option<db::ModelRecord> {
    if !is_safe_slug(slug) || !is_safe_run_id(run_id) {
        return None;
    }
    let run_dir = models_root.join(slug).join("runs").join(run_id);
    let manifest_raw = fs::read_to_string(run_dir.join("manifest.json")).ok()?;
    let manifest: Value = serde_json::from_str(&manifest_raw).ok()?;
    let metrics = fs::read_to_string(run_dir.join(format!("{slug}_metrics.json"))).ok();
    let mstr = |k: &str| manifest.get(k).and_then(Value::as_str).map(str::to_string);
    let mi64 = |k: &str| manifest.get(k).and_then(Value::as_i64);
    let sstr = |k: &str| status.get(k).and_then(Value::as_str).map(str::to_string);
    // The manifest carries the full eval object; the metrics sidecar is stored
    // raw. Prefer manifest-recorded knobs (the effective, resolved values) and
    // fall back to the terminal status body for the few older fields.
    let eval = manifest
        .get("eval")
        .filter(|v| !v.is_null())
        .map(|v| v.to_string());
    Some(db::ModelRecord {
        slug: slug.to_string(),
        phrase: mstr("phrase").or_else(|| sstr("phrase")).unwrap_or_default(),
        run_id: run_id.to_string(),
        onnx_path: mstr("onnx_path").unwrap_or_else(|| format!("runs/{run_id}/{slug}.onnx")),
        onnx_sha256: mstr("onnx_sha256"),
        pt_sha256: mstr("pt_sha256"),
        steps: mi64("steps").or_else(|| status.get("steps").and_then(Value::as_i64)),
        model_size: mstr("model_size").or_else(|| sstr("model_size")),
        personal: manifest
            .get("personal")
            .and_then(Value::as_bool)
            .or_else(|| status.get("personal").and_then(Value::as_bool))
            .unwrap_or(false),
        git_commit: mstr("git_commit"),
        metrics,
        started_at: mstr("started_at").or_else(|| sstr("started_at")),
        finished_at: mstr("finished_at").or_else(|| sstr("updated_at")),
        model_type: mstr("model_type"),
        token_type: mstr("token_type"),
        positive_boost: mi64("positive_boost"),
        n_samples: mi64("n_samples"),
        n_samples_val: mi64("n_samples_val"),
        positive_per_batch: mi64("positive_per_batch"),
        target_fp_per_hour: manifest.get("target_fp_per_hour").and_then(Value::as_f64),
        context_fix: manifest.get("context_fix").and_then(Value::as_bool),
        real_positive: mi64("real_positive"),
        real_negative: mi64("real_negative"),
        real_background: mi64("real_background"),
        trainer_image: mstr("trainer_image"),
        onnx_bytes: mi64("onnx_bytes"),
        eval,
        params_json: Some(manifest_raw),
        synth_positive: mi64("synth_positive"),
        kokoro_positive: mi64("kokoro_positive"),
        total_positive_input: mi64("total_positive_input"),
    })
}

/// POST the WAV to the scorer service and parse its rolling-score curve.
async fn run_scorer(
    scorer_url: &str,
    wav: Vec<u8>,
    slug: &str,
    run: Option<&str>,
    mode: &str,
    step_ms: u64,
    keep_ms: u64,
) -> Result<ScorerCurve, AppError> {
    let part = reqwest::multipart::Part::bytes(wav)
        .file_name(format!("{slug}.wav"))
        .mime_str("audio/wav")
        .map_err(|error| AppError::internal(format!("prepare scorer upload: {error}")))?;
    let mut form = reqwest::multipart::Form::new()
        .part("file", part)
        .text("slug", slug.to_string())
        .text("mode", mode.to_string())
        .text("step_ms", step_ms.to_string())
        .text("keep_ms", keep_ms.to_string());
    if let Some(run) = run {
        form = form.text("run", run.to_string());
    }
    let endpoint = format!("{}/score", scorer_url.trim_end_matches('/'));
    let response = reqwest::Client::new()
        .post(endpoint)
        .multipart(form)
        .send()
        .await
        .map_err(|error| AppError::internal(format!("scorer request failed: {error}")))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| AppError::internal(format!("scorer response read failed: {error}")))?;
    if !status.is_success() {
        return Err(AppError::internal(format!(
            "scorer returned {status}: {}",
            body.trim()
        )));
    }
    serde_json::from_str(&body)
        .map_err(|error| AppError::internal(format!("scorer response JSON failed: {error}")))
}

/// Find every run of transcript words matching the trigger phrase and attach the
/// model's peak score in the detection window aligned to the phrase tail. The
/// model fires when the phrase ends the 2s window, so the peak sits near the
/// last word's end; we search a small band around it.
fn locate_targets(
    phrase: &str,
    words: &[db::WordRow],
    curve: &ScorerCurve,
    threshold: f64,
) -> Vec<ScoreTarget> {
    let tokens: Vec<String> = phrase
        .split_whitespace()
        .map(normalize_token)
        .filter(|t| !t.is_empty())
        .collect();
    if tokens.is_empty() || words.is_empty() {
        return Vec::new();
    }
    // First pass: locate every phrase occurrence and its Whisper start/end.
    let occ = phrase_occurrences(&tokens, words);
    // Second pass: score each occurrence at the model's peak within its drift
    // window (neighbor-clamped so dense repeats can't claim each other's firing).
    let ends: Vec<f64> = occ.iter().map(|(_, end, _)| *end as f64).collect();
    let windows = target_windows(&ends);
    occ.into_iter()
        .zip(windows)
        .map(|((start_ms, end_ms, text), (lo, hi))| {
            let (peak_time_ms, peak_score) = peak_in_window(curve, lo, hi);
            ScoreTarget {
                text,
                start_ms,
                end_ms,
                peak_score,
                peak_time_ms,
                detected: peak_score >= threshold,
            }
        })
        .collect()
}

/// Every non-overlapping run of transcript words whose normalized tokens equal
/// `tokens`, as `(start_ms, end_ms, joined_text)`. Matching consumes the phrase
/// (advances past it) so dense repeats yield one occurrence each. Pure, so the
/// occurrence-finding is testable without a scorer curve.
fn phrase_occurrences(tokens: &[String], words: &[db::WordRow]) -> Vec<(i64, i64, String)> {
    let normalized: Vec<String> = words.iter().map(|w| normalize_token(&w.word)).collect();
    let mut occ: Vec<(i64, i64, String)> = Vec::new();
    let mut i = 0;
    while i + tokens.len() <= normalized.len() {
        if normalized[i..i + tokens.len()] == tokens[..] {
            let start_ms = words[i].start_ms;
            let end_ms = words[i + tokens.len() - 1].end_ms;
            let text = words[i..i + tokens.len()]
                .iter()
                .map(|w| w.word.trim())
                .collect::<Vec<_>>()
                .join(" ");
            occ.push((start_ms, end_ms, text));
            i += tokens.len();
        } else {
            i += 1;
        }
    }
    occ
}

/// One detection search window per located phrase, in time order. Whisper word
/// timings drift up to ~1s from where the tail-aligned model actually fires, so
/// each window spans `end ± MAX_DRIFT_MS`. To stop a dense script (many "all
/// set" a couple seconds apart) from letting neighbors claim the same firing,
/// each window is clamped to the midpoints between adjacent phrase ends.
fn target_windows(ends: &[f64]) -> Vec<(f64, f64)> {
    ends.iter()
        .enumerate()
        .map(|(k, &e)| {
            let mut lo = e - MAX_DRIFT_MS;
            let mut hi = e + MAX_DRIFT_MS;
            if k > 0 {
                lo = lo.max((ends[k - 1] + e) / 2.0);
            }
            if k + 1 < ends.len() {
                hi = hi.min((e + ends[k + 1]) / 2.0);
            }
            (lo.max(0.0), hi)
        })
        .collect()
}

/// Highest score (and its time) among curve points within [lo_ms, hi_ms].
fn peak_in_window(curve: &ScorerCurve, lo_ms: f64, hi_ms: f64) -> (f64, f64) {
    let mut best = (lo_ms.max(0.0), 0.0_f64);
    for (t, s) in curve.times_ms.iter().zip(curve.scores.iter()) {
        if *t >= lo_ms && *t <= hi_ms && *s >= best.1 {
            best = (*t, *s);
        }
    }
    best
}

/// Count above-threshold detections that do not overlap any target window. A
/// contiguous run above threshold counts once. A detection whose time falls in
/// any target's search band is a true positive, not a false one.
fn count_false_positives(curve: &ScorerCurve, targets: &[ScoreTarget], threshold: f64) -> usize {
    let ends: Vec<f64> = targets.iter().map(|t| t.end_ms as f64).collect();
    let bands = target_windows(&ends);
    let mut count = 0;
    let mut in_run = false;
    for (t, s) in curve.times_ms.iter().zip(curve.scores.iter()) {
        let hot = *s >= threshold;
        let near_target = bands.iter().any(|(lo, hi)| *t >= *lo && *t <= *hi);
        if hot && !near_target {
            if !in_run {
                count += 1;
                in_run = true;
            }
        } else {
            in_run = false;
        }
    }
    count
}
