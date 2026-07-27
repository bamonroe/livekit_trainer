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
use crate::docker::{docker_ok, f5_container, kokoro_container, run_docker, zonos_container};
use crate::error::{db_error, AppError};
use crate::state::{now_ms, AppState, SynthJob};
use crate::train::ValidatedTrain;
use crate::util::{is_safe_slug, parse_query};
use crate::whisper::{normalized_words, transcribe_with_words, transcript_tail_has_phrase};
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

/// How many real reference clips to stage per F5 batch. The generator
/// concatenates a rotating window (`--concat-size`) of these into each priming
/// clip and rolls the window forward per set, so this caps how many distinct
/// real positives a batch can draw on (and how many small files we `docker cp`).
const STAGED_REFERENCE_CAP: usize = 48;

/// Which resident voice-clone TTS produced a batch of synthetic positives. Both
/// sources share the same pipeline — stage the user's real refs, run a resident
/// generator over docker, Whisper-gate the output — and differ only in their
/// bucket subdir, container, generator script, clip prefix, and status label. F5
/// is the default so every existing app call (which passes no `source`) is
/// unchanged; Zonos is the second source that adds explicit prosody levers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SynthSource {
    F5,
    Zonos,
    /// Impostor negatives: the wake phrase spoken by not-the-user, female Kokoro
    /// voices. Pooled into NEGATIVES (not positives) so a personal detector learns
    /// to reject the phrase in other people's voices. Has no user reference clips.
    Impostor,
}

impl SynthSource {
    /// Parse the `?source=` query value; anything but an explicit `zonos` /
    /// `impostor` (incl. the absent/default case) is F5, so existing callers keep
    /// F5 behavior.
    fn from_str(value: Option<&str>) -> SynthSource {
        match value {
            Some(v) if v.eq_ignore_ascii_case("zonos") => SynthSource::Zonos,
            Some(v) if v.eq_ignore_ascii_case("impostor") || v.eq_ignore_ascii_case("kokoro") => {
                SynthSource::Impostor
            }
            _ => SynthSource::F5,
        }
    }

    /// Which pooled category this source feeds: F5/Zonos clone the user so they
    /// are positives; Impostor is the phrase in other voices, so it is a negative.
    fn category(self) -> &'static str {
        match self {
            SynthSource::F5 | SynthSource::Zonos => "positive",
            SynthSource::Impostor => "negative",
        }
    }

    /// Whether this source clones the user's real reference clips. F5/Zonos do;
    /// the Kokoro impostor source uses its own named voices and stages no refs.
    fn has_refs(self) -> bool {
        matches!(self, SynthSource::F5 | SynthSource::Zonos)
    }

    /// Resolve the source from a raw query string's `source` param (default F5).
    fn from_query(query: Option<&str>) -> SynthSource {
        let params = parse_query(query);
        SynthSource::from_str(params.get("source").map(String::as_str))
    }

    /// The `data/<subdir>/<slug>/<category>` bucket subdir for this source.
    fn subdir(self) -> &'static str {
        match self {
            SynthSource::F5 => "synth_f5",
            SynthSource::Zonos => "synth_zonos",
            SynthSource::Impostor => "synth_impostor_neg",
        }
    }

    /// The resident generator container name for this source.
    fn container(self) -> String {
        match self {
            SynthSource::F5 => f5_container(),
            SynthSource::Zonos => zonos_container(),
            SynthSource::Impostor => kokoro_container(),
        }
    }

    /// The container path of the generator script (mounted at /trainer).
    fn gen_script(self) -> &'static str {
        match self {
            SynthSource::F5 => "/trainer/scripts/f5_gen_positives.py",
            SynthSource::Zonos => "/trainer/scripts/zonos_gen_positives.py",
            SynthSource::Impostor => "/trainer/scripts/kokoro_gen_negatives.py",
        }
    }

    /// Short lowercase tag used for scratch dirs, status field prefixes, and the
    /// synth-job map key. Matches the generator's clip filename prefix.
    fn tag(self) -> &'static str {
        match self {
            SynthSource::F5 => "f5",
            SynthSource::Zonos => "zonos",
            SynthSource::Impostor => "impostor",
        }
    }

    /// Human label for status/log messages.
    fn label(self) -> &'static str {
        match self {
            SynthSource::F5 => "F5",
            SynthSource::Zonos => "Zonos",
            SynthSource::Impostor => "Impostor (Kokoro)",
        }
    }

    /// The `train_status.json` pre-launch phase step for this source. The trainer
    /// module treats all of these as pre-launch generation phases (not a crash).
    fn status_step(self) -> &'static str {
        match self {
            SynthSource::F5 => "f5gen",
            SynthSource::Zonos => "zonosgen",
            SynthSource::Impostor => "impostorgen",
        }
    }

    /// The synth-job map key: `<tag>:<slug>` so F5 and Zonos runs for the same
    /// wake word never collide in the shared map.
    fn job_key(self, slug: &str) -> String {
        format!("{}:{slug}", self.tag())
    }
}

/// Directory holding a source's generated clips for a slug. The bucket lives at
/// `<repo>/data/<subdir>/<slug>/<category>`; the container mounts the repo `data/`
/// at the parent of `data_root` (DATA_ROOT=/data/real → /data), so the bucket is
/// a sibling of `data_root`. F5/Zonos land under `positive`, the Kokoro impostor
/// source under `negative`.
fn synth_bucket_dir(data_root: &Path, slug: &str, source: SynthSource) -> PathBuf {
    data_root
        .parent()
        .unwrap_or(data_root)
        .join(source.subdir())
        .join(slug)
        .join(source.category())
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
    RawQuery(query): RawQuery,
) -> Result<Json<Value>, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request(format!("unsafe wake word slug: {slug}")));
    }
    let source = SynthSource::from_query(query.as_deref());
    {
        let jobs = state.synth_jobs.lock().expect("synth jobs lock poisoned");
        if jobs.get(&source.job_key(&slug)).map(|j| j.running).unwrap_or(false) {
            return Err(AppError::bad_request(
                "a generation run is in progress; wait for it to finish before deleting",
            ));
        }
    }
    // The synth bucket is data/<subdir>/<slug>/positive; remove its <slug> parent
    // so nothing for this wake word is left behind.
    let slug_dir = synth_bucket_dir(&state.data_root, &slug, source)
        .parent()
        .map(Path::to_path_buf);
    let removed = match slug_dir {
        Some(dir) if dir.exists() => {
            let count = count_wavs(&synth_bucket_dir(&state.data_root, &slug, source));
            fs::remove_dir_all(&dir)
                .map_err(|e| AppError::internal(format!("failed to delete synth dir: {e}")))?;
            count
        }
        _ => 0,
    };
    Ok(Json(json!({ "status": "ok", "slug": slug, "source": source.tag(), "deleted": removed })))
}

pub(crate) async fn synthetic_samples(
    State(state): State<AppState>,
    AxumPath(slug): AxumPath<String>,
    RawQuery(query): RawQuery,
) -> Result<Json<SyntheticSamplesResponse>, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request(format!("unsafe wake word slug: {slug}")));
    }
    let source = SynthSource::from_query(query.as_deref());
    let dir = synth_bucket_dir(&state.data_root, &slug, source);
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
    let sampled = evenly_spaced_sample(files, MAX_SAMPLES);

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

/// Serve one synthetic positive WAV by file name. Mirrors `review_audio`.
pub(crate) async fn synthetic_audio(
    State(state): State<AppState>,
    AxumPath((slug, file_name)): AxumPath<(String, String)>,
    RawQuery(query): RawQuery,
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
    let source = SynthSource::from_query(query.as_deref());
    let path = synth_bucket_dir(&state.data_root, &slug, source).join(&file_name);
    let bytes = fs::read(path)?;
    Ok(([(header::CONTENT_TYPE, "audio/wav")], bytes))
}

/// Pick an evenly-spaced sample of at most `max` items from a sorted list: all of
/// them when the batch is small, otherwise a deterministic stride across it (no
/// RNG) so the sample spans start, middle, and end. Pure so the spread is
/// unit-testable. Extracted verbatim from `synthetic_samples`.
fn evenly_spaced_sample(files: Vec<String>, max: usize) -> Vec<String> {
    let total = files.len();
    if total <= max {
        files
    } else {
        let stride = total as f64 / max as f64;
        (0..max)
            .map(|i| files[((i as f64) * stride) as usize].clone())
            .collect()
    }
}

/// The reference WAV file names to stage for one F5 batch: every `.wav` in the
/// reference dir, sorted, truncated to the first `STAGED_REFERENCE_CAP`. Errors
/// when none exist. The generator concatenates a rotating window of these into
/// each priming clip, so the cap must comfortably exceed one window; it bounds
/// how many small files we `docker cp` per run. Extracted from
/// `run_synth_generation`'s staging step.
fn staged_reference_names(refs_server_dir: &Path) -> Result<Vec<String>, AppError> {
    let mut names: Vec<String> = fs::read_dir(refs_server_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.ends_with(".wav"))
        .collect();
    names.sort();
    names.truncate(STAGED_REFERENCE_CAP);
    if names.is_empty() {
        return Err(AppError::internal("no reference wavs to stage"));
    }
    Ok(names)
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

/// Drive one generation batch for a slug: resolve reference clips when the source
/// clones the user (F5/Zonos), run the resident generator, and land the resampled
/// clips in the source's bucket. Returns how many clips this run produced. Shared
/// by the manual `/synth/generate` endpoint and the train-time pre-generation
/// step. The Kokoro impostor source stages no refs — it uses its own voices.
async fn generate_synth_batch(
    state: &AppState,
    slug: &str,
    phrase: &str,
    count: usize,
    source: SynthSource,
) -> Result<usize, AppError> {
    let refs = if source.has_refs() {
        Some(resolve_synth_refs(&state.data_root, slug)?)
    } else {
        None
    };
    let synth_dir = synth_bucket_dir(&state.data_root, slug, source);
    run_synth_generation(
        state,
        slug,
        phrase,
        count,
        source,
        refs.as_deref(),
        &synth_dir,
        state.whisper_url.as_deref(),
    )
    .await
}

/// Ensure the F5 synth bucket holds `vt.f5_count` voice-cloned positives before
/// the trainer assembles them. Thin wrapper over `ensure_synth_positives`.
pub(crate) async fn ensure_f5_positives(
    state: &AppState,
    slug: &str,
    phrase: &str,
    vt: &ValidatedTrain,
) -> usize {
    ensure_synth_positives(state, slug, phrase, vt, vt.f5_count as usize, SynthSource::F5).await
}

/// Ensure the Zonos synth bucket holds `vt.zonos_count` voice-cloned positives
/// before the trainer assembles them. Thin wrapper over `ensure_synth_positives`.
pub(crate) async fn ensure_zonos_positives(
    state: &AppState,
    slug: &str,
    phrase: &str,
    vt: &ValidatedTrain,
) -> usize {
    ensure_synth_positives(state, slug, phrase, vt, vt.zonos_count as usize, SynthSource::Zonos)
        .await
}

/// Ensure the Kokoro impostor-negative bucket holds `vt.impostor_neg_count` clips
/// (the phrase in female voices) before the trainer assembles them. Thin wrapper
/// over `ensure_synth_positives`. No user references are needed.
pub(crate) async fn ensure_impostor_negatives(
    state: &AppState,
    slug: &str,
    phrase: &str,
    vt: &ValidatedTrain,
) -> usize {
    ensure_synth_positives(
        state,
        slug,
        phrase,
        vt,
        vt.impostor_neg_count as usize,
        SynthSource::Impostor,
    )
    .await
}

/// Ensure a source's synth bucket for `slug` holds `target` voice-cloned
/// positives before the trainer assembles them, generating a fresh batch when it
/// is short. Reuses an existing batch that is already large enough (so repeated
/// trains don't re-pay the generator's cost); otherwise clears the bucket and
/// regenerates exactly `target` (the generator names clips `<tag>_00000..`, so a
/// clean bucket yields exactly the count with no stale leftovers). Writes the
/// source's pre-launch phase into train_status.json so the app's training screen
/// shows it. Non-fatal: on any failure it logs, records the error, and returns
/// whatever clips exist so training still proceeds. Returns the final clip count.
/// F5 and Zonos runs for the same slug key the job map independently, so the two
/// pre-launch top-ups never collide.
pub(crate) async fn ensure_synth_positives(
    state: &AppState,
    slug: &str,
    phrase: &str,
    vt: &ValidatedTrain,
    target: usize,
    source: SynthSource,
) -> usize {
    let label = source.label();
    // What this source contributes, for human-readable status messages.
    let noun = if source.has_refs() {
        "voice-cloned positives"
    } else {
        "impostor negatives"
    };
    let synth_dir = synth_bucket_dir(&state.data_root, slug, source);
    if target == 0 {
        return count_wavs(&synth_dir);
    }
    let existing = count_wavs(&synth_dir);
    if existing >= target {
        return existing;
    }
    // Cloning sources need the user's real positives to clone from; resolve them
    // before touching the bucket so a slug with none keeps whatever it already
    // has. Reference-free sources (Kokoro impostor) skip this check.
    if source.has_refs() {
        if let Err(e) = resolve_synth_refs(&state.data_root, slug) {
            eprintln!("{label} pregen: no reference clips for {slug}: {}", e.message);
            return existing;
        }
    }
    // Don't collide with a manual generation already running for this slug/source.
    let key = source.job_key(slug);
    {
        let mut jobs = state.synth_jobs.lock().expect("synth jobs lock poisoned");
        if jobs.get(&key).map(|j| j.running).unwrap_or(false) {
            eprintln!(
                "{label} pregen: a manual generation is already running for {slug}; \
                 training on the {existing} existing clips"
            );
            return existing;
        }
        jobs.insert(
            key.clone(),
            SynthJob {
                running: true,
                requested: target,
                wrote: existing,
                error: None,
            },
        );
    }
    write_synth_status(
        state,
        slug,
        phrase,
        vt,
        source,
        target,
        0,
        &format!("generating {target} {noun} ({label})"),
    );
    // Fresh batch: clear the bucket so the count is exactly `target` (no stale
    // clips from a previous, larger run linger).
    let _ = fs::remove_dir_all(&synth_dir);
    let result = generate_synth_batch(state, slug, phrase, target, source).await;
    let wrote = *result.as_ref().unwrap_or(&0);
    if let Err(e) = &result {
        eprintln!("{label} pregen failed for {slug}: {}", e.message);
    }
    {
        let mut jobs = state.synth_jobs.lock().expect("synth jobs lock poisoned");
        if let Some(j) = jobs.get_mut(&key) {
            j.running = false;
            j.wrote = wrote;
            j.error = result.as_ref().err().map(|e| e.message.clone());
        }
    }
    let msg = match &result {
        Ok(w) => format!("generated {w} {noun} ({label})"),
        Err(e) => format!(
            "{label} generation failed: {}; training on existing clips",
            e.message
        ),
    };
    write_synth_status(state, slug, phrase, vt, source, target, wrote, &msg);
    count_wavs(&synth_dir)
}

/// Write the pre-training generation phase into a slug's train_status.json. The
/// trainer overwrites this file with its own phases once it launches; until then
/// the app sees a `running` status stepped with the source's phase (`f5gen` /
/// `zonosgen`), carrying the generated clip counts under `<tag>_requested` /
/// `<tag>_wrote`. F5 writes exactly the same `f5gen`/`f5_requested`/`f5_wrote`
/// fields it always has.
#[allow(clippy::too_many_arguments)]
fn write_synth_status(
    state: &AppState,
    slug: &str,
    phrase: &str,
    vt: &ValidatedTrain,
    source: SynthSource,
    requested: usize,
    wrote: usize,
    message: &str,
) {
    let dir = state.models_root.join(slug);
    if fs::create_dir_all(&dir).is_err() {
        return;
    }
    let now = Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
    let tag = source.tag();
    let body = json!({
        "slug": slug,
        "phrase": phrase,
        "state": "running",
        "step": source.status_step(),
        "exit_code": 0,
        "message": message,
        "steps": vt.steps,
        "model_size": vt.model_size,
        "personal": vt.personal,
        (format!("{tag}_requested")): requested,
        (format!("{tag}_wrote")): wrote,
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
    let source = SynthSource::from_str(params.get("source").map(String::as_str));
    let count: usize = params
        .get("count")
        .and_then(|v| v.parse().ok())
        .unwrap_or(60)
        .clamp(1, 1000);

    // Cloning sources (F5/Zonos) clone the user's SHORT real positive clips: their
    // transcript is exactly the phrase and their length is F5-friendly. (F5 sizes
    // the spoken output from the reference's rate, so a long passage starves the
    // short wake phrase and leaks its own text.) Refuse early if there is nothing
    // to clone from. The Kokoro impostor source stages no refs.
    let refs_server_dir = if source.has_refs() {
        Some(resolve_synth_refs(&state.data_root, &slug)?)
    } else {
        None
    };

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

    let key = source.job_key(&slug);
    {
        let mut jobs = state.synth_jobs.lock().expect("synth jobs lock poisoned");
        if jobs.get(&key).map(|j| j.running).unwrap_or(false) {
            return Err(AppError::bad_request(
                "a generation run is already in progress for this wake word",
            ));
        }
        jobs.insert(
            key.clone(),
            SynthJob {
                running: true,
                requested: count,
                wrote: 0,
                error: None,
            },
        );
    }

    let synth_server_dir = synth_bucket_dir(&state.data_root, &slug, source);

    let task_state = state.clone();
    let task_slug = slug.clone();
    let task_phrase = phrase.clone();
    let task_key = key.clone();
    tokio::spawn(async move {
        let result = run_synth_generation(
            &task_state,
            &task_slug,
            &task_phrase,
            count,
            source,
            refs_server_dir.as_deref(),
            &synth_server_dir,
            task_state.whisper_url.as_deref(),
        )
        .await;
        let mut jobs = task_state.synth_jobs.lock().expect("synth jobs lock poisoned");
        let entry = jobs.entry(task_key).or_insert(SynthJob {
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
    state: &AppState,
    slug: &str,
    phrase: &str,
    count: usize,
    source: SynthSource,
    refs_server_dir: Option<&Path>,
    synth_server_dir: &Path,
    whisper_url: Option<&str>,
) -> Result<usize, AppError> {
    let container = source.container();
    let tag = source.tag();
    let cp_out = synth_server_dir.to_string_lossy().to_string();
    let cp_gen_py = source.gen_script();
    let stamp = now_ms();
    let scratch = format!("/tmp/{tag}gen_{slug}_{stamp}");
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

    // Cloning sources (F5/Zonos) stage the user's real references; the Kokoro
    // impostor source has none (it uses its own named voices), so skip staging.
    let cnegs = format!("{scratch}/negs");
    let mut have_negs = false;
    if let Some(refs_server_dir) = refs_server_dir {
        let cp_refs = refs_server_dir.to_string_lossy().to_string();
        // Stage a small, clean subset of references (rotated inside the python).
        let names = staged_reference_names(refs_server_dir)?;
        for name in &names {
            docker_ok(vec![
                "cp".into(),
                format!("{cp_refs}/{name}"),
                format!("{container}:{crefs}/{name}"),
            ])
            .await?;
            // An enrollment reference ships its exact passage in a sibling .txt;
            // stage it too so F5 gets the right ref_text.
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

        // Zonos clones its accent best from ~16s of varied phonetics, so also
        // stage the user's real NEGATIVE takes (full sentences, sibling `negative`
        // dir) — the generator builds each priming clip half from these and half
        // from the short positives. Best-effort: no negatives → positives-only.
        if matches!(source, SynthSource::Zonos) {
            if let Some(neg_server_dir) = refs_server_dir.parent().map(|p| p.join("negative")) {
                if dir_has_wav(&neg_server_dir) {
                    docker_ok(vec!["exec".into(), container.clone(), "mkdir".into(),
                        "-p".into(), cnegs.clone()])
                    .await?;
                    let cp_negs = neg_server_dir.to_string_lossy().to_string();
                    for name in staged_reference_names(&neg_server_dir)? {
                        docker_ok(vec![
                            "cp".into(),
                            format!("{cp_negs}/{name}"),
                            format!("{container}:{cnegs}/{name}"),
                        ])
                        .await?;
                    }
                    have_negs = true;
                }
            }
        }
    }

    // Stage and run the resident-model generator, writing 16 kHz clips directly.
    docker_ok(vec![
        "cp".into(),
        cp_gen_py.to_string(),
        format!("{container}:{scratch}/gen.py"),
    ])
    .await?;
    // Common CLI shape: every generator accepts --gen-text/--out-dir/--count/
    // --out-sr. Cloning generators (F5/Zonos) additionally take the staged
    // --refs-dir/--ref-text; the Kokoro impostor generator does not.
    let mut gen_args = vec![
        "exec".into(),
        container.clone(),
        "python3".into(),
        format!("{scratch}/gen.py"),
        "--gen-text".into(),
        phrase.to_string(),
        "--out-dir".into(),
        cout.clone(),
        "--count".into(),
        count.to_string(),
        "--out-sr".into(),
        "16000".into(),
    ];
    if source.has_refs() {
        gen_args.extend([
            "--refs-dir".into(),
            crefs.clone(),
            "--ref-text".into(),
            phrase.to_string(),
        ]);
    }
    match source {
        SynthSource::F5 => gen_args.extend([
            // Fidelity knobs (F5 defaults unless overridden in the environment).
            // Raise F5_NFE_STEP for a sharper, more faithful render (slower);
            // raise F5_CFG_STRENGTH to hew closer to the user's timbre.
            "--nfe-step".into(),
            env::var("F5_NFE_STEP").unwrap_or_else(|_| "32".to_string()),
            "--cfg-strength".into(),
            env::var("F5_CFG_STRENGTH").unwrap_or_else(|_| "2.0".to_string()),
            // Batched-priming rotation: concatenate this many real refs into one
            // priming clip, render this many seeds off it, then rotate the window.
            "--concat-size".into(),
            env::var("F5_CONCAT_SIZE").unwrap_or_else(|_| "5".to_string()),
            "--seeds-per-set".into(),
            env::var("F5_SEEDS_PER_SET").unwrap_or_else(|_| "5".to_string()),
        ]),
        // Zonos carries its own sane prosody-lever defaults (speaking_rate,
        // pitch_std, emotion jitter). Point it at the staged negatives when we
        // have them so priming spans varied phonetics and holds the user's accent.
        SynthSource::Zonos => {
            if have_negs {
                gen_args.extend(["--neg-refs-dir".into(), cnegs.clone()]);
            }
        }
        // Kokoro impostor carries its own female-voice pool and prosody jitter
        // defaults, so it runs on the common args alone.
        SynthSource::Impostor => {}
    }
    // Run generation while polling the container's out dir so the app can show a
    // live wave-file count. `docker exec ls` runs concurrently with the generator
    // process; each tick updates this source's SynthJob, which both the Review
    // and training status endpoints read back.
    let job_key = source.job_key(slug);
    let gen_fut = docker_ok(gen_args);
    tokio::pin!(gen_fut);
    loop {
        tokio::select! {
            res = &mut gen_fut => { res?; break; }
            _ = tokio::time::sleep(std::time::Duration::from_secs(2)) => {
                if let Ok(out) = run_docker(vec![
                    "exec".into(), container.clone(), "sh".into(), "-c".into(),
                    format!("ls -1 {cout}/*.wav 2>/dev/null | wc -l"),
                ]).await {
                    if let Ok(n) = String::from_utf8_lossy(&out.stdout).trim().parse::<usize>() {
                        let mut jobs = state.synth_jobs.lock().expect("synth jobs lock poisoned");
                        if let Some(j) = jobs.get_mut(&job_key) {
                            j.wrote = n;
                        }
                    }
                }
            }
        }
    }

    // Copy the finished clips into the synth bucket (dir must exist first).
    fs::create_dir_all(synth_server_dir)?;
    docker_ok(vec![
        "cp".into(),
        format!("{container}:{cout}/."),
        format!("{cp_out}/"),
    ])
    .await?;

    // List exactly the clips this run produced (basenames), so the Whisper gate
    // and the count only touch this batch, not older clips already in the bucket.
    let listed = docker_ok(vec![
        "exec".into(),
        container.clone(),
        "sh".into(),
        "-c".into(),
        format!("cd {cout} && ls -1 *.wav 2>/dev/null"),
    ])
    .await?;
    let names: Vec<String> = String::from_utf8_lossy(&listed.stdout)
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();

    // Whisper-verify the fresh clips: F5 zero-shot cloning occasionally
    // hallucinates filler ("um", "okay") or a quote instead of the wake phrase,
    // and such a clip is a poisoned positive. Drop any whose transcript doesn't
    // end in the phrase before they can enter the training pool.
    let wrote = match whisper_url {
        Some(url) => gate_synth_clips(url, phrase, synth_server_dir, &names).await,
        None => {
            eprintln!("{tag} gate: no Whisper URL configured; skipping verification");
            names.len()
        }
    };

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

/// Fraction of a batch that may fail the phrase check before the gate assumes the
/// wake word is non-lexical (Whisper hears no words) and keeps the whole batch
/// rather than gutting it. "beep beep" and similar sound-based tokens transcribe
/// to nothing, so a near-total failure means "don't trust Whisper here", not
/// "every clip is bad".
const GATE_MAX_REJECT_FRACTION: f64 = 0.8;

/// Whisper-verify freshly generated F5 clips in `dir` (named by `names`), deleting
/// any whose transcript doesn't end in the wake phrase. Returns how many clips
/// survive. Best-effort and fail-open: a Whisper error on a clip keeps it (better
/// than dropping good data on a flaky call), and if nearly the whole batch fails
/// the check the wake word is presumed non-lexical and the batch is kept intact.
///
/// Whisper is deliberately called with NO phrase prompt: priming it with the
/// expected text would bias it toward "hearing" the phrase in a hallucinated
/// clip, defeating the gate.
async fn gate_synth_clips(
    whisper_url: &str,
    phrase: &str,
    dir: &Path,
    names: &[String],
) -> usize {
    let phrase_words = normalized_words(phrase);
    if phrase_words.is_empty() || names.is_empty() {
        return names.len();
    }
    // Pass 1: collect verdicts without deleting, so a mostly-failing batch (a
    // non-lexical wake word) can be spared wholesale below.
    let mut fails: Vec<(String, String)> = Vec::new();
    let mut checked = 0usize;
    for name in names {
        let path = dir.join(name);
        if !path.exists() {
            continue;
        }
        match transcribe_with_words(whisper_url, &path, None).await {
            Ok(resp) => {
                checked += 1;
                if !transcript_tail_has_phrase(&resp.text, &phrase_words) {
                    fails.push((name.clone(), resp.text.trim().to_string()));
                }
            }
            Err(e) => eprintln!("f5 gate: whisper failed on {name}: {}; keeping", e.message),
        }
    }
    if checked == 0 {
        return names.len();
    }
    if (fails.len() as f64 / checked as f64) > GATE_MAX_REJECT_FRACTION {
        eprintln!(
            "f5 gate: {}/{checked} clips lacked \"{phrase}\"; assuming non-lexical wake word, keeping all",
            fails.len()
        );
        return names.len();
    }
    // Pass 2: delete the confirmed hallucinations.
    let mut rejected = 0usize;
    for (name, heard) in &fails {
        if fs::remove_file(dir.join(name)).is_ok() {
            rejected += 1;
            eprintln!("f5 gate: rejected {name}: heard {heard:?}");
        }
    }
    eprintln!(
        "f5 gate: kept {}/{} clips ({rejected} rejected as not \"{phrase}\")",
        names.len() - rejected,
        names.len()
    );
    names.len() - rejected
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
    RawQuery(query): RawQuery,
) -> Result<Json<GenerateSynthStatusResponse>, AppError> {
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request(format!("unsafe wake word slug: {slug}")));
    }
    let source = SynthSource::from_query(query.as_deref());
    let job = {
        let jobs = state.synth_jobs.lock().expect("synth jobs lock poisoned");
        jobs.get(&source.job_key(&slug)).cloned()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(dir: &Path, name: &str) {
        fs::write(dir.join(name), b"x").expect("write test file");
    }

    #[test]
    fn evenly_spaced_sample_returns_all_when_small() {
        let files = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert_eq!(evenly_spaced_sample(files.clone(), 24), files);
    }

    #[test]
    fn evenly_spaced_sample_strides_across_large_batch() {
        let files: Vec<String> = (0..100).map(|i| format!("{i:03}")).collect();
        let sampled = evenly_spaced_sample(files, 4);
        // Stride 25: indices 0, 25, 50, 75.
        assert_eq!(sampled, vec!["000", "025", "050", "075"]);
    }

    #[test]
    fn synth_bucket_dir_is_sibling_of_data_root() {
        let dir = synth_bucket_dir(Path::new("/data/real"), "all_set", SynthSource::F5);
        assert_eq!(dir, PathBuf::from("/data/synth_f5/all_set/positive"));
        // Zonos lands in its own sibling bucket, keyed by the source subdir.
        let zdir = synth_bucket_dir(Path::new("/data/real"), "all_set", SynthSource::Zonos);
        assert_eq!(zdir, PathBuf::from("/data/synth_zonos/all_set/positive"));
    }

    #[test]
    fn synth_source_from_query_defaults_to_f5() {
        assert_eq!(SynthSource::from_query(None), SynthSource::F5);
        assert_eq!(SynthSource::from_query(Some("count=5")), SynthSource::F5);
        assert_eq!(SynthSource::from_query(Some("source=f5")), SynthSource::F5);
        assert_eq!(SynthSource::from_query(Some("source=ZONOS")), SynthSource::Zonos);
        assert_eq!(SynthSource::from_query(Some("source=zonos&count=3")), SynthSource::Zonos);
        // An unknown source falls back to F5 rather than erroring.
        assert_eq!(SynthSource::from_query(Some("source=bogus")), SynthSource::F5);
    }

    #[test]
    fn synth_source_job_keys_never_collide() {
        assert_eq!(SynthSource::F5.job_key("w"), "f5:w");
        assert_eq!(SynthSource::Zonos.job_key("w"), "zonos:w");
        assert_ne!(SynthSource::F5.job_key("w"), SynthSource::Zonos.job_key("w"));
    }

    #[test]
    fn dir_has_wav_and_count_wavs_over_temp_dir() {
        let base = env::temp_dir().join(format!("synth_test_{}", now_ms()));
        fs::create_dir_all(&base).expect("mkdir");
        // A missing dir counts as zero and no wav.
        let missing = base.join("missing");
        assert!(!dir_has_wav(&missing));
        assert_eq!(count_wavs(&missing), 0);
        // Non-wav files are ignored.
        touch(&base, "notes.txt");
        assert!(!dir_has_wav(&base));
        assert_eq!(count_wavs(&base), 0);
        // Wavs are counted.
        touch(&base, "a.wav");
        touch(&base, "b.wav");
        assert!(dir_has_wav(&base));
        assert_eq!(count_wavs(&base), 2);
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn staged_reference_names_sorts_truncates_and_requires_some() {
        let base = env::temp_dir().join(format!("synth_refs_{}", now_ms()));
        fs::create_dir_all(&base).expect("mkdir");
        // No wavs -> error.
        assert!(staged_reference_names(&base).is_err());
        // More wavs than the cap (plus a txt) -> the first CAP by sort order,
        // wavs only.
        for i in 0..(STAGED_REFERENCE_CAP + 5) {
            touch(&base, &format!("ref_{i:03}.wav"));
        }
        touch(&base, "ref_000.txt");
        let names = staged_reference_names(&base).expect("names");
        assert_eq!(names.len(), STAGED_REFERENCE_CAP);
        assert_eq!(names[0], "ref_000.wav");
        assert!(names.iter().all(|n| n.ends_with(".wav")));
        let _ = fs::remove_dir_all(&base);
    }
}
