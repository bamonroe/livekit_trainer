//! Alignment engine: turning long takes into training slices on disk.
//!
//! This is where a stored source WAV becomes positive/negative/hard-negative/
//! background clips. It composes the pure slicing math with Whisper
//! transcription, WAV IO, and the database: per take kind it routes to the right
//! strategy (word-timestamp slicing, the energy fallback for non-lexical
//! positives, fixed-window background chunking, or store-whole for test/
//! enrollment takes), verifies each positive actually contains the wake phrase,
//! and persists the recording alignment. `AlignmentSummary` accumulates the
//! per-take counts and warnings the sync/reprocess handlers report back.

use crate::audio::{build_slice_row, wav_duration_seconds, write_wav_slice};
use crate::bundle::{capture_from_extra, store_bulk_source, Manifest};
use crate::constants::{
    BACKGROUND_MIN_CHUNK_SECONDS, BACKGROUND_SCRIPT_MARKER, CUT_LEAD_PADDING_SECONDS,
    ENERGY_POSITIVE_SCRIPT_MARKER, NEGATIVE_TAIL_PADDING_SECONDS, POSITIVE_TAIL_PADDING_SECONDS,
};
use crate::db;
use crate::error::{db_error, AppError};
use crate::slicing::{
    background_chunk_bounds, bulk_clip_hash_id, clamp_slice_span, energy_burst_bounds,
    is_hard_negative_context, negative_ranges, padded_bounds, phrase_ranges,
    positive_context_first, visible_range,
};
use crate::state::now_ms;
use crate::util::{is_safe_slug, safe_filename, safe_join};
use crate::whisper::{
    normalized_words, transcribe_with_words, transcript_tail_has_phrase, whisper_words,
};
use hound::{SampleFormat, WavReader};
use rusqlite::Connection;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Debug, Default)]
pub(crate) struct AlignmentSummary {
    pub(crate) recordings: usize,
    pub(crate) positives: usize,
    pub(crate) negatives: usize,
    pub(crate) hard_negatives: usize,
    pub(crate) background: usize,
    pub(crate) dropped_positives: usize,
    pub(crate) warnings: Vec<String>,
}

impl AlignmentSummary {
    /// Fold another summary's counts and warnings into this one. Used to combine
    /// the bulk-alignment and background-slicing passes over a single upload.
    pub(crate) fn absorb(&mut self, other: AlignmentSummary) {
        self.recordings += other.recordings;
        self.positives += other.positives;
        self.negatives += other.negatives;
        self.hard_negatives += other.hard_negatives;
        self.background += other.background;
        self.dropped_positives += other.dropped_positives;
        self.warnings.extend(other.warnings);
    }
}
pub(crate) async fn align_bulk_recordings(
    bundle: &Path,
    manifest: &Manifest,
    data_root: &Path,
    db: &Mutex<Connection>,
    whisper_url: Option<&str>,
) -> Result<AlignmentSummary, AppError> {
    let mut summary = AlignmentSummary::default();
    if manifest.bulk_recordings.is_empty() {
        return Ok(summary);
    }
    let Some(whisper_url) = whisper_url.map(str::trim).filter(|value| !value.is_empty()) else {
        summary
            .warnings
            .push("bulk recordings present but no Whisper server URL configured".to_string());
        return Ok(summary);
    };

    let slug = manifest.wake_word.slug.clone();
    let phrase = manifest.wake_word.phrase();
    let external_id = manifest
        .wake_word
        .extra
        .get("id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string);
    let dest_root = data_root.join(&slug);

    for recording in &manifest.bulk_recordings {
        let source = match safe_join(bundle, &recording.file) {
            Ok(source) => source,
            Err(error) => {
                summary.recordings += 1;
                summary
                    .warnings
                    .push(format!("{}: {}", recording.id, error.message));
                continue;
            }
        };
        align_one_recording(
            &recording.id,
            &recording.script,
            &recording.kind,
            &recording.recorded_at,
            recording.duration_ms,
            &source,
            &slug,
            &phrase,
            external_id.as_deref(),
            &dest_root,
            db,
            whisper_url,
            &capture_from_extra(&recording.extra),
            &mut summary,
        )
        .await?;
    }

    Ok(summary)
}

/// Slice every background take in the bundle into fixed-length background clips.
/// Independent of Whisper, so it runs even when no Whisper URL is configured.
pub(crate) async fn align_background_recordings(
    bundle: &Path,
    manifest: &Manifest,
    data_root: &Path,
    db: &Mutex<Connection>,
) -> Result<AlignmentSummary, AppError> {
    let mut summary = AlignmentSummary::default();
    if manifest.background_recordings.is_empty() {
        return Ok(summary);
    }

    let slug = manifest.wake_word.slug.clone();
    let phrase = manifest.wake_word.phrase();
    let external_id = manifest
        .wake_word
        .extra
        .get("id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string);
    let dest_root = data_root.join(&slug);

    for recording in &manifest.background_recordings {
        let source = match safe_join(bundle, &recording.file) {
            Ok(source) => source,
            Err(error) => {
                summary.recordings += 1;
                summary
                    .warnings
                    .push(format!("{}: {}", recording.id, error.message));
                continue;
            }
        };
        summary.recordings += 1;
        slice_background_recording(
            &recording.id,
            &recording.recorded_at,
            recording.duration_ms,
            &source,
            &slug,
            &phrase,
            external_id.as_deref(),
            &dest_root,
            db,
            &capture_from_extra(&recording.extra),
            &mut summary,
        )?;
    }

    Ok(summary)
}

/// Transcribe every test take in the bundle for word timings, but cut no
/// training slices. Requires Whisper, since scoring needs to locate the wake
/// phrase inside the take; without it the take is stored source-only.
pub(crate) async fn align_test_recordings(
    bundle: &Path,
    manifest: &Manifest,
    data_root: &Path,
    db: &Mutex<Connection>,
    whisper_url: Option<&str>,
) -> Result<AlignmentSummary, AppError> {
    let mut summary = AlignmentSummary::default();
    if manifest.test_recordings.is_empty() {
        return Ok(summary);
    }
    let Some(whisper_url) = whisper_url.map(str::trim).filter(|value| !value.is_empty()) else {
        summary
            .warnings
            .push("test recordings present but no Whisper server URL configured".to_string());
        return Ok(summary);
    };

    let slug = manifest.wake_word.slug.clone();
    let phrase = manifest.wake_word.phrase();
    let external_id = manifest
        .wake_word
        .extra
        .get("id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string);
    let dest_root = data_root.join(&slug);

    for recording in &manifest.test_recordings {
        let source = match safe_join(bundle, &recording.file) {
            Ok(source) => source,
            Err(error) => {
                summary.recordings += 1;
                summary
                    .warnings
                    .push(format!("{}: {}", recording.id, error.message));
                continue;
            }
        };
        // Route through align_one_recording so the shared `test_` branch handles
        // it identically to a reprocess pass.
        align_one_recording(
            &recording.id,
            &recording.script,
            &recording.kind,
            &recording.recorded_at,
            recording.duration_ms,
            &source,
            &slug,
            &phrase,
            external_id.as_deref(),
            &dest_root,
            db,
            whisper_url,
            &capture_from_extra(&recording.extra),
            &mut summary,
        )
        .await?;
    }

    Ok(summary)
}

/// Store a test take: transcribe it so scoring can locate the wake phrase,
/// persist the source WAV and transcript, and record ZERO slices. A test take
/// must never contribute training clips, so this path writes nothing under the
/// positive/negative/background directories.
async fn align_test_one(
    recording_id: &str,
    script: &str,
    recorded_at: &str,
    duration_ms: u64,
    source: &Path,
    slug: &str,
    phrase: &str,
    external_id: Option<&str>,
    dest_root: &Path,
    db: &Mutex<Connection>,
    whisper_url: &str,
    capture: &db::CaptureMeta,
    summary: &mut AlignmentSummary,
) -> Result<(), AppError> {
    let whisper = match transcribe_with_words(whisper_url, source, None).await {
        Ok(whisper) => whisper,
        Err(error) => {
            summary
                .warnings
                .push(format!("{}: {}", recording_id, error.message));
            return Ok(());
        }
    };
    let words = whisper_words(&whisper);

    // Persist the raw source recording, then drop any slice files a prior pass
    // left behind (e.g. if this id was ever mis-processed as a bulk take).
    let (source_wav, source_sha256) = match store_bulk_source(dest_root, recording_id, source) {
        Ok(stored) => stored,
        Err(error) => {
            summary
                .warnings
                .push(format!("{}: {}", recording_id, error.message));
            return Ok(());
        }
    };
    let old_paths = {
        let conn = db.lock().expect("db lock poisoned");
        db::active_slice_paths(&conn, recording_id).map_err(db_error)?
    };
    for old in old_paths {
        let old = PathBuf::from(old);
        if old.is_file() {
            let _ = fs::remove_file(old);
        }
    }

    let word_rows = words
        .iter()
        .map(|word| db::WordRow {
            word: word.word.trim().to_string(),
            start_ms: (word.start * 1000.0).round() as i64,
            end_ms: (word.end * 1000.0).round() as i64,
            probability: word.probability,
        })
        .collect();
    let prompts = script
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToString::to_string)
        .collect();

    let alignment = db::RecordingAlignment {
        recording_id: recording_id.to_string(),
        project_slug: slug.to_string(),
        phrase: phrase.to_string(),
        external_id: external_id.map(ToString::to_string),
        script: script.to_string(),
        kind: "test".to_string(),
        recorded_at: recorded_at.to_string(),
        duration_ms: duration_ms as i64,
        source_wav: source_wav.to_string_lossy().to_string(),
        source_sha256: Some(source_sha256),
        bundle: None,
        transcript_text: whisper.text.trim().to_string(),
        whisper_url: Some(whisper_url.to_string()),
        words: word_rows,
        prompts,
        // Intentionally empty: test takes contribute no training data.
        slices: Vec::new(),
        capture: capture.clone(),
    };
    {
        let mut conn = db.lock().expect("db lock poisoned");
        db::store_recording_alignment(&mut conn, &alignment, now_ms()).map_err(db_error)?;
    }
    if words.is_empty() {
        summary.warnings.push(format!(
            "{}: test take stored but Whisper returned no words; scoring can still run",
            recording_id
        ));
    }
    Ok(())
}

/// Store an enrollment take whole. Enrollment reads are the voice-cloning
/// reference for F5-TTS — one clean take of the user reading a fixed passage —
/// so they must never be sliced into training clips and never touch Whisper: the
/// passage text is already known (it is the recording's `script`), so there is
/// nothing to align. This mirrors `align_test_one` (persist the raw source, drop
/// any stale slices, write an alignment with an empty slice list) but skips
/// transcription entirely. Reprocess re-enters here off the persisted `kind`.
fn store_enrollment_whole(
    recording_id: &str,
    script: &str,
    recorded_at: &str,
    duration_ms: u64,
    source: &Path,
    slug: &str,
    phrase: &str,
    external_id: Option<&str>,
    dest_root: &Path,
    db: &Mutex<Connection>,
    capture: &db::CaptureMeta,
    summary: &mut AlignmentSummary,
) -> Result<(), AppError> {
    let (source_wav, source_sha256) = match store_bulk_source(dest_root, recording_id, source) {
        Ok(stored) => stored,
        Err(error) => {
            summary
                .warnings
                .push(format!("{}: {}", recording_id, error.message));
            return Ok(());
        }
    };
    let old_paths = {
        let conn = db.lock().expect("db lock poisoned");
        db::active_slice_paths(&conn, recording_id).map_err(db_error)?
    };
    for old in old_paths {
        let old = PathBuf::from(old);
        if old.is_file() {
            let _ = fs::remove_file(old);
        }
    }

    let prompts = script
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToString::to_string)
        .collect();

    let alignment = db::RecordingAlignment {
        recording_id: recording_id.to_string(),
        project_slug: slug.to_string(),
        phrase: phrase.to_string(),
        external_id: external_id.map(ToString::to_string),
        script: script.to_string(),
        kind: "enrollment".to_string(),
        recorded_at: recorded_at.to_string(),
        duration_ms: duration_ms as i64,
        source_wav: source_wav.to_string_lossy().to_string(),
        source_sha256: Some(source_sha256),
        bundle: None,
        // The passage text is known; store it as the transcript without Whisper.
        transcript_text: script.trim().to_string(),
        whisper_url: None,
        words: Vec::new(),
        prompts,
        // Intentionally empty: enrollment reads are the F5 reference, not training data.
        slices: Vec::new(),
        capture: capture.clone(),
    };
    {
        let mut conn = db.lock().expect("db lock poisoned");
        db::store_recording_alignment(&mut conn, &alignment, now_ms()).map_err(db_error)?;
    }

    // Also drop the take into a stable `enrollment/` bucket with a sidecar
    // transcript, so the F5 generator can find the reference audio + its exact
    // text without querying the DB. The passage is known, so the .txt is precise.
    if let Err(error) = write_enrollment_reference(dest_root, recording_id, &source_wav, script) {
        summary
            .warnings
            .push(format!("{}: {}", recording_id, error.message));
    }
    Ok(())
}

/// Copy an enrollment take into `<dest_root>/enrollment/<id>.wav` and write its
/// transcript to `<id>.txt` beside it, giving the F5 generator a self-describing
/// reference (audio + exact ref_text) with no DB lookup.
fn write_enrollment_reference(
    dest_root: &Path,
    recording_id: &str,
    source_wav: &Path,
    script: &str,
) -> Result<(), AppError> {
    let dir = dest_root.join("enrollment");
    fs::create_dir_all(&dir)?;
    let safe = safe_filename(recording_id);
    fs::copy(source_wav, dir.join(format!("{safe}.wav")))?;
    fs::write(dir.join(format!("{safe}.txt")), script.trim())?;
    Ok(())
}

/// Align and slice one already-materialized source WAV. Shared by the upload
/// path (source from the bundle) and the reprocess path (source is the stored
/// bulk_source WAV), so both cut slices identically.
pub(crate) async fn align_one_recording(
    recording_id: &str,
    script: &str,
    kind: &str,
    recorded_at: &str,
    duration_ms: u64,
    source: &Path,
    slug: &str,
    phrase: &str,
    external_id: Option<&str>,
    dest_root: &Path,
    db: &Mutex<Connection>,
    whisper_url: &str,
    capture: &db::CaptureMeta,
    summary: &mut AlignmentSummary,
) -> Result<(), AppError> {
    summary.recordings += 1;
    if recording_id.starts_with("test_") {
        // Test takes exist only to score a trained model. Transcribe them for
        // word timings but cut zero training slices, so they can never enter the
        // positive/negative/background pool. This branch also fires on reprocess,
        // since it keys off the persisted `test_` id prefix.
        return align_test_one(
            recording_id,
            script,
            recorded_at,
            duration_ms,
            source,
            slug,
            phrase,
            external_id,
            dest_root,
            db,
            whisper_url,
            capture,
            summary,
        )
        .await;
    }
    if kind.trim() == "enrollment" {
        // Enrollment reads are the F5 voice-cloning reference: store the whole
        // take, cut zero slices, skip Whisper. Keyed off the persisted kind so
        // reprocess re-enters here too.
        return store_enrollment_whole(
            recording_id,
            script,
            recorded_at,
            duration_ms,
            source,
            slug,
            phrase,
            external_id,
            dest_root,
            db,
            capture,
            summary,
        );
    }
    if script == BACKGROUND_SCRIPT_MARKER {
        // Background takes carry no speech to align; chop into fixed windows and
        // skip Whisper entirely. This branch also fires on reprocess, since the
        // marker is persisted in the recording's script column.
        return slice_background_recording(
            recording_id,
            recorded_at,
            duration_ms,
            source,
            slug,
            phrase,
            external_id,
            dest_root,
            db,
            capture,
            summary,
        );
    }
    if kind.trim() == "positive" && script == ENERGY_POSITIVE_SCRIPT_MARKER {
        // The project marked this wake word as a non-lexical sound, so slice this
        // positive take by burst energy and skip Whisper entirely — a single
        // burst Whisper happened to catch must not collapse the whole take to one
        // positive. Reprocess re-enters here because the marker is persisted.
        return slice_positive_by_energy(
            recording_id,
            script,
            recorded_at,
            duration_ms,
            source,
            slug,
            phrase,
            external_id,
            dest_root,
            db,
            "",
            whisper_url,
            capture,
            summary,
        );
    }
    let whisper = match transcribe_with_words(whisper_url, source, None).await {
        Ok(whisper) => whisper,
        Err(error) => {
            summary
                .warnings
                .push(format!("{}: {}", recording_id, error.message));
            return Ok(());
        }
    };
    let words = whisper_words(&whisper);
    if words.is_empty() {
        // Non-lexical positive take (e.g. a fast "beep beep") — Whisper heard no
        // words, so fall back to energy/VAD burst slicing for positives only.
        // Negatives and hard negatives with no words carry no useful training
        // data, so they keep the plain warning-and-skip behavior.
        if kind.trim() == "positive" {
            return slice_positive_by_energy(
                recording_id,
                script,
                recorded_at,
                duration_ms,
                source,
                slug,
                phrase,
                external_id,
                dest_root,
                db,
                whisper.text.trim(),
                whisper_url,
                capture,
                summary,
            );
        }
        summary.warnings.push(format!(
            "{}: Whisper returned no word timestamps",
            recording_id
        ));
        return Ok(());
    }

    let phrase_words = normalized_words(phrase);
    if phrase_words.is_empty() {
        summary.warnings.push(format!(
            "{}: wake phrase has no alignable words",
            recording_id
        ));
        return Ok(());
    }

    // Route by take kind. Each straight recording knows what it is, so the server
    // no longer has to guess positives vs negatives from one mixed script.
    //  - `positive`: the wake phrase said many times; every aligned occurrence is
    //    a positive, no near-miss frames to split out.
    //  - `negative` / `hard_negative`: ordinary speech or near-miss phrases with
    //    no true wake phrase; the whole read is chopped by the negative pass and
    //    filed under the matching label.
    //  - `mixed` (legacy, or older app builds that send no kind): the original
    //    behavior — split aligned occurrences into positives and the
    //    near-miss-framed hard negatives inferred from surrounding cue words.
    let take_kind = kind.trim();
    let expects_positive = !matches!(take_kind, "negative" | "hard_negative");
    let (positive_ranges, hard_negative_ranges): (Vec<_>, Vec<_>) = match take_kind {
        "positive" => (phrase_ranges(&words, &phrase_words), Vec::new()),
        "negative" | "hard_negative" => (Vec::new(), Vec::new()),
        _ => phrase_ranges(&words, &phrase_words)
            .into_iter()
            .partition(|(first, last)| !is_hard_negative_context(&words, *first, *last)),
    };
    // Whether to run the generic negative pass, and the label its slices carry.
    // A token take skips negatives entirely so silent gaps never become clips.
    let (run_negatives, negative_label): (bool, &str) = match take_kind {
        "positive" => (false, "negative"),
        "hard_negative" => (true, "hard_negative"),
        _ => (true, "negative"),
    };
    if expects_positive && positive_ranges.is_empty() {
        summary.warnings.push(format!(
            "{}: no aligned wake phrase occurrences found in transcript {:?}",
            recording_id,
            whisper.text.trim()
        ));
    }

    // Persist the raw source recording, then remove any stale slice files
    // this recording produced on a previous pass before writing the new set.
    let (source_wav, source_sha256) = match store_bulk_source(dest_root, recording_id, source) {
        Ok(stored) => stored,
        Err(error) => {
            summary
                .warnings
                .push(format!("{}: {}", recording_id, error.message));
            return Ok(());
        }
    };
    let old_paths = {
        let conn = db.lock().expect("db lock poisoned");
        db::active_slice_paths(&conn, recording_id).map_err(db_error)?
    };
    for old in old_paths {
        let old = PathBuf::from(old);
        if old.is_file() {
            let _ = fs::remove_file(old);
        }
    }

    let source_end = wav_duration_seconds(source)
        .unwrap_or_else(|_| words.last().map(|word| word.end).unwrap_or(0.0));

        let mut occupied = Vec::new();
        let mut slice_rows = Vec::new();
        for (first, last) in positive_ranges.iter() {
            let context_first = positive_context_first(&words, *first, *last);
            let (start, end) = padded_bounds(
                &words,
                context_first,
                *last,
                source_end,
                CUT_LEAD_PADDING_SECONDS,
                POSITIVE_TAIL_PADDING_SECONDS,
                false,
            );
            let (start, end) = clamp_slice_span(start, end, true);
            let (visible_first, visible_last) =
                visible_range(&words, context_first, *last, start, end);
            let slice_words = &words[visible_first..=visible_last];
            let clip_id = bulk_clip_hash_id(
                recording_id,
                recorded_at,
                duration_ms,
                "positive",
                start,
                end,
                slice_words,
            );
            let file_name = format!("{}_{}.wav", safe_filename(&clip_id), safe_filename(phrase));
            let dest = dest_root.join("positive").join(&file_name);
            if write_wav_slice(source, &dest, start, end)? {
                // Verify the cut audio actually contains the wake phrase, judged
                // on its own with no script prompt. Whisper word timings are
                // unstable, so a phrase the alignment placed here may not really
                // be in the slice; drop it rather than poison training.
                let heard = transcribe_with_words(whisper_url, &dest, None)
                    .await
                    .ok()
                    .map(|response| transcript_tail_has_phrase(&response.text, &phrase_words));
                match heard {
                    Some(false) => {
                        let _ = fs::remove_file(&dest);
                        summary.dropped_positives += 1;
                        summary.warnings.push(format!(
                            "{}: dropped positive at {:.2}-{:.2}s; wake phrase not heard in slice",
                            recording_id, start, end
                        ));
                    }
                    // Kept when the phrase is heard, or when verification itself
                    // failed (a transient Whisper error should not discard data).
                    _ => {
                        slice_rows.push(build_slice_row(
                            &clip_id,
                            "positive",
                            "positive",
                            &dest,
                            &file_name,
                            start,
                            end,
                            slice_words,
                        ));
                        summary.positives += 1;
                    }
                }
            }
            occupied.push((start, end));
        }

        // Hard negatives: the wake phrase captured in a near-miss frame. Cut with
        // the same generous bounds as a positive so the whole phrase is present,
        // but file it under the negative category with a distinct label, and mark
        // it occupied so the generic negative pass does not re-cut the same words.
        for (first, last) in hard_negative_ranges.iter() {
            let context_first = positive_context_first(&words, *first, *last);
            let (start, end) = padded_bounds(
                &words,
                context_first,
                *last,
                source_end,
                CUT_LEAD_PADDING_SECONDS,
                POSITIVE_TAIL_PADDING_SECONDS,
                false,
            );
            let (start, end) = clamp_slice_span(start, end, true);
            let (visible_first, visible_last) =
                visible_range(&words, context_first, *last, start, end);
            let slice_words = &words[visible_first..=visible_last];
            let clip_id = bulk_clip_hash_id(
                recording_id,
                recorded_at,
                duration_ms,
                "hard_negative",
                start,
                end,
                slice_words,
            );
            let phrase_text = slice_words
                .iter()
                .map(|word| word.word.trim())
                .collect::<Vec<_>>()
                .join(" ");
            let file_name = format!(
                "{}_{}.wav",
                safe_filename(&clip_id),
                safe_filename(&phrase_text)
            );
            // Hard negatives live in their own on-disk category so the pooling
            // assembler keeps them scoped to this project instead of borrowing
            // them into every other wake word's negative pool.
            let dest = dest_root.join("hard_negative").join(&file_name);
            if write_wav_slice(source, &dest, start, end)? {
                slice_rows.push(build_slice_row(
                    &clip_id,
                    "hard_negative",
                    "hard_negative",
                    &dest,
                    &file_name,
                    start,
                    end,
                    slice_words,
                ));
                summary.hard_negatives += 1;
            }
            occupied.push((start, end));
        }

        // Generic negative pass: every stretch of speech not already claimed by a
        // positive or hard-negative cut becomes a negative clip. For a `negative`
        // or `hard_negative` take `occupied` is empty, so this chops the whole
        // read; `negative_label` files it under the matching label. Skipped for a
        // token take so its silent gaps never turn into stray clips.
        if run_negatives {
            let negative_ranges = negative_ranges(&words, &occupied);
            for (_, _, word_start, word_end) in negative_ranges.iter() {
                let (start, end) = padded_bounds(
                    &words,
                    *word_start,
                    *word_end,
                    source_end,
                    CUT_LEAD_PADDING_SECONDS,
                    NEGATIVE_TAIL_PADDING_SECONDS,
                    true,
                );
                let (start, end) = clamp_slice_span(start, end, false);
                let (visible_first, visible_last) =
                    visible_range(&words, *word_start, *word_end, start, end);
                let slice_words = &words[visible_first..=visible_last];
                let clip_id = bulk_clip_hash_id(
                    recording_id,
                    recorded_at,
                    duration_ms,
                    negative_label,
                    start,
                    end,
                    slice_words,
                );
                let phrase_text = slice_words
                    .iter()
                    .map(|word| word.word.trim())
                    .collect::<Vec<_>>()
                    .join(" ");
                let file_name = format!(
                    "{}_{}.wav",
                    safe_filename(&clip_id),
                    safe_filename(&phrase_text)
                );
                // A hard-negative take files its whole read under the hard_negative
                // category (project-scoped); everything else is a pooled negative.
                let neg_category = if negative_label == "hard_negative" {
                    "hard_negative"
                } else {
                    "negative"
                };
                let dest = dest_root.join(neg_category).join(&file_name);
                if write_wav_slice(source, &dest, start, end)? {
                    slice_rows.push(build_slice_row(
                        &clip_id,
                        negative_label,
                        neg_category,
                        &dest,
                        &file_name,
                        start,
                        end,
                        slice_words,
                    ));
                    if negative_label == "hard_negative" {
                        summary.hard_negatives += 1;
                    } else {
                        summary.negatives += 1;
                    }
                }
            }
        }

        let word_rows = words
            .iter()
            .map(|word| db::WordRow {
                word: word.word.trim().to_string(),
                start_ms: (word.start * 1000.0).round() as i64,
                end_ms: (word.end * 1000.0).round() as i64,
                probability: word.probability,
            })
            .collect();
        let prompts = script
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(ToString::to_string)
            .collect();

        let alignment = db::RecordingAlignment {
            recording_id: recording_id.to_string(),
            project_slug: slug.to_string(),
            phrase: phrase.to_string(),
            external_id: external_id.map(ToString::to_string),
            script: script.to_string(),
            // Persist the resolved kind so reprocess slices this take the same way
            // without a manifest; empty (older app) normalizes to legacy mixed.
            kind: if take_kind.is_empty() { "mixed".to_string() } else { take_kind.to_string() },
            recorded_at: recorded_at.to_string(),
            duration_ms: duration_ms as i64,
            source_wav: source_wav.to_string_lossy().to_string(),
            source_sha256: Some(source_sha256),
            bundle: None,
            transcript_text: whisper.text.trim().to_string(),
            whisper_url: Some(whisper_url.to_string()),
            words: word_rows,
            prompts,
            slices: slice_rows,
            capture: capture.clone(),
        };
        {
            let mut conn = db.lock().expect("db lock poisoned");
            db::store_recording_alignment(&mut conn, &alignment, now_ms()).map_err(db_error)?;
        }

    Ok(())
}

/// Slice one ambient background take into fixed-length background clips without
/// transcription. Shared by the upload path and reprocess (which re-enters via
/// `align_one_recording` on the background script marker), so both chunk a stored
/// source WAV identically. Persists a recording row carrying the marker script so
/// later reprocesses keep treating it as background.
fn slice_background_recording(
    recording_id: &str,
    recorded_at: &str,
    duration_ms: u64,
    source: &Path,
    slug: &str,
    phrase: &str,
    external_id: Option<&str>,
    dest_root: &Path,
    db: &Mutex<Connection>,
    capture: &db::CaptureMeta,
    summary: &mut AlignmentSummary,
) -> Result<(), AppError> {
    let (source_wav, source_sha256) = match store_bulk_source(dest_root, recording_id, source) {
        Ok(stored) => stored,
        Err(error) => {
            summary
                .warnings
                .push(format!("{}: {}", recording_id, error.message));
            return Ok(());
        }
    };
    // Remove any slice files a previous pass produced before writing the new set.
    let old_paths = {
        let conn = db.lock().expect("db lock poisoned");
        db::active_slice_paths(&conn, recording_id).map_err(db_error)?
    };
    for old in old_paths {
        let old = PathBuf::from(old);
        if old.is_file() {
            let _ = fs::remove_file(old);
        }
    }

    let total = wav_duration_seconds(source).unwrap_or(0.0);
    if total < BACKGROUND_MIN_CHUNK_SECONDS {
        summary.warnings.push(format!(
            "{}: background recording too short to slice ({:.2}s)",
            recording_id, total
        ));
    }

    let mut slice_rows = Vec::new();
    for (start, end) in background_chunk_bounds(total) {
        let clip_id = bulk_clip_hash_id(
            recording_id,
            recorded_at,
            duration_ms,
            "background",
            start,
            end,
            &[],
        );
        let file_name = format!("{}_background.wav", safe_filename(&clip_id));
        let dest = dest_root.join("background").join(&file_name);
        if write_wav_slice(source, &dest, start, end)? {
            slice_rows.push(build_slice_row(
                &clip_id,
                "background",
                "background",
                &dest,
                &file_name,
                start,
                end,
                &[],
            ));
            summary.background += 1;
        }
    }

    let alignment = db::RecordingAlignment {
        recording_id: recording_id.to_string(),
        project_slug: slug.to_string(),
        phrase: phrase.to_string(),
        external_id: external_id.map(ToString::to_string),
        script: BACKGROUND_SCRIPT_MARKER.to_string(),
        kind: "background".to_string(),
        recorded_at: recorded_at.to_string(),
        duration_ms: duration_ms as i64,
        source_wav: source_wav.to_string_lossy().to_string(),
        source_sha256: Some(source_sha256),
        bundle: None,
        transcript_text: String::new(),
        whisper_url: None,
        words: Vec::new(),
        prompts: Vec::new(),
        slices: slice_rows,
        capture: capture.clone(),
    };
    {
        let mut conn = db.lock().expect("db lock poisoned");
        db::store_recording_alignment(&mut conn, &alignment, now_ms()).map_err(db_error)?;
    }

    Ok(())
}

/// Slice a non-lexical positive take into positive clips by energy bursts, used
/// when Whisper returned no words. Mirrors `slice_background_recording`: stores
/// the source WAV, clears stale slices, writes one positive per detected burst,
/// and persists the alignment (with the resolved `positive` kind so reprocess
/// re-enters here). No per-slice Whisper verification — there are no words to
/// confirm against, which is the whole reason this path exists.
#[allow(clippy::too_many_arguments)]
fn slice_positive_by_energy(
    recording_id: &str,
    script: &str,
    recorded_at: &str,
    duration_ms: u64,
    source: &Path,
    slug: &str,
    phrase: &str,
    external_id: Option<&str>,
    dest_root: &Path,
    db: &Mutex<Connection>,
    transcript_text: &str,
    whisper_url: &str,
    capture: &db::CaptureMeta,
    summary: &mut AlignmentSummary,
) -> Result<(), AppError> {
    let (source_wav, source_sha256) = match store_bulk_source(dest_root, recording_id, source) {
        Ok(stored) => stored,
        Err(error) => {
            summary
                .warnings
                .push(format!("{}: {}", recording_id, error.message));
            return Ok(());
        }
    };
    let old_paths = {
        let conn = db.lock().expect("db lock poisoned");
        db::active_slice_paths(&conn, recording_id).map_err(db_error)?
    };
    for old in old_paths {
        let old = PathBuf::from(old);
        if old.is_file() {
            let _ = fs::remove_file(old);
        }
    }

    // Read the whole take as mono 16-bit PCM to compute the energy envelope.
    let mut reader = WavReader::open(source)
        .map_err(|error| AppError::bad_request(format!("cannot read WAV for slicing: {error}")))?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_format != SampleFormat::Int || spec.bits_per_sample != 16 {
        return Err(AppError::bad_request(
            "energy slicing requires mono 16-bit PCM WAV",
        ));
    }
    let samples: Vec<i16> = reader
        .samples::<i16>()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| AppError::bad_request(format!("bad WAV sample while slicing: {error}")))?;
    let bursts = energy_burst_bounds(&samples, spec.sample_rate as f64);
    if bursts.is_empty() {
        summary.warnings.push(format!(
            "{}: no sound bursts found for energy-sliced positive take",
            recording_id
        ));
    }

    let mut slice_rows = Vec::new();
    for (start, end) in bursts {
        let (start, end) = clamp_slice_span(start, end, true);
        let clip_id = bulk_clip_hash_id(
            recording_id,
            recorded_at,
            duration_ms,
            "positive",
            start,
            end,
            &[],
        );
        let file_name = format!("{}_{}.wav", safe_filename(&clip_id), safe_filename(phrase));
        let dest = dest_root.join("positive").join(&file_name);
        if write_wav_slice(source, &dest, start, end)? {
            slice_rows.push(build_slice_row(
                &clip_id,
                "positive",
                "positive",
                &dest,
                &file_name,
                start,
                end,
                &[],
            ));
            summary.positives += 1;
        }
    }

    // The energy marker is an internal routing sentinel, not a read-aloud prompt,
    // so keep it out of the displayed prompt list — but leave it in the persisted
    // `script` below so reprocess re-enters the energy path.
    let prompts = script
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && *line != ENERGY_POSITIVE_SCRIPT_MARKER)
        .map(ToString::to_string)
        .collect();
    let alignment = db::RecordingAlignment {
        recording_id: recording_id.to_string(),
        project_slug: slug.to_string(),
        phrase: phrase.to_string(),
        external_id: external_id.map(ToString::to_string),
        script: script.to_string(),
        kind: "positive".to_string(),
        recorded_at: recorded_at.to_string(),
        duration_ms: duration_ms as i64,
        source_wav: source_wav.to_string_lossy().to_string(),
        source_sha256: Some(source_sha256),
        bundle: None,
        transcript_text: transcript_text.to_string(),
        whisper_url: Some(whisper_url.to_string()),
        words: Vec::new(),
        prompts,
        slices: slice_rows,
        capture: capture.clone(),
    };
    {
        let mut conn = db.lock().expect("db lock poisoned");
        db::store_recording_alignment(&mut conn, &alignment, now_ms()).map_err(db_error)?;
    }
    Ok(())
}

pub(crate) fn alignment_summary(summary: &AlignmentSummary) -> String {
    if summary.recordings == 0 {
        if !summary.warnings.is_empty() {
            let mut output = "No bulk recordings were aligned".to_string();
            output.push_str("\nWarnings:");
            for warning in &summary.warnings {
                output.push_str("\n- ");
                output.push_str(warning);
            }
            return output;
        }
        return "No bulk recordings".to_string();
    }
    let mut output = format!(
        "Aligned {} recordings into {} positives and {} negatives",
        summary.recordings, summary.positives, summary.negatives
    );
    if summary.hard_negatives > 0 {
        output.push_str(&format!(
            " (incl. {} hard negatives)",
            summary.hard_negatives
        ));
    }
    if summary.background > 0 {
        output.push_str(&format!(", plus {} background clips", summary.background));
    }
    if summary.dropped_positives > 0 {
        output.push_str(&format!(
            " ({} positives dropped: wake phrase not heard)",
            summary.dropped_positives
        ));
    }
    if !summary.warnings.is_empty() {
        output.push_str("\nWarnings:");
        for warning in &summary.warnings {
            output.push_str("\n- ");
            output.push_str(warning);
        }
    }
    output
}

pub(crate) fn review_clip_path(
    data_root: &Path,
    slug: &str,
    category: &str,
    file_name: &str,
) -> Result<PathBuf, AppError> {
    if !is_safe_slug(slug) {
        return Err(AppError::bad_request(format!(
            "unsafe wake word slug: {slug}"
        )));
    }
    if !matches!(category, "positive" | "negative" | "hard_negative" | "background") {
        return Err(AppError::bad_request(format!(
            "unsafe category: {category}"
        )));
    }
    if file_name.contains('/')
        || file_name.contains('\\')
        || file_name.contains("..")
        || !file_name.ends_with(".wav")
    {
        return Err(AppError::bad_request(format!(
            "unsafe file name: {file_name}"
        )));
    }
    Ok(data_root.join(slug).join(category).join(file_name))
}

