//! Upload-bundle schema, validation, and import.
//!
//! A sync upload is a zip carrying a `manifest.json` plus WAVs. This module owns
//! the manifest types, unpacks and validates the archive (path-safety, schema,
//! WAV format sanity), imports the pre-cut short clips into `data/real`, and
//! stores each long take's source WAV. The alignment code consumes the parsed
//! manifest to slice the bulk/background/test takes.

use crate::audio::file_sha256;
use crate::db;
use crate::error::AppError;
use crate::state::now_ms;
use crate::util::{category_for_label, is_safe_slug, safe_filename, safe_join};
use hound::WavReader;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::{self, File};
use std::io::{self, Cursor};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use zip::ZipArchive;

/// The parsed `manifest.json` describing one upload.
#[derive(Debug, Deserialize)]
pub(crate) struct Manifest {
    pub(crate) schema_version: u64,
    pub(crate) wake_word: WakeWord,
    pub(crate) clips: Vec<Clip>,
    #[serde(default)]
    pub(crate) bulk_recordings: Vec<BulkRecording>,
    #[serde(default)]
    pub(crate) background_recordings: Vec<BackgroundRecording>,
    /// Test-only takes. Transcribed for word timings so the model can be scored
    /// against them, but never sliced into training data. Recording ids carry the
    /// `test_` prefix so every downstream path can keep them out of the pool.
    #[serde(default)]
    pub(crate) test_recordings: Vec<BulkRecording>,
}

/// The wake word an upload belongs to: a stable slug plus any extra fields
/// (phrase, external id) carried opaquely in `extra`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct WakeWord {
    pub(crate) slug: String,
    #[serde(flatten)]
    pub(crate) extra: Value,
}

impl WakeWord {
    /// The spoken wake phrase, falling back to the slug when none is carried.
    pub(crate) fn phrase(&self) -> String {
        self.extra
            .get("phrase")
            .and_then(Value::as_str)
            .unwrap_or(&self.slug)
            .to_string()
    }
}

/// One pre-cut short clip in an upload.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct Clip {
    pub(crate) id: String,
    pub(crate) file: String,
    pub(crate) label: String,
    #[serde(default)]
    pub(crate) spoken_phrase: String,
    #[serde(flatten)]
    pub(crate) extra: Value,
}

/// A long scripted take (positive/negative/hard_negative/mixed) to be
/// transcribed and sliced server-side.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct BulkRecording {
    pub(crate) id: String,
    pub(crate) file: String,
    pub(crate) script: String,
    /// How to slice this take: `positive`, `negative`, `hard_negative`, or (for
    /// takes from older app builds) empty, which the server treats as `mixed`.
    #[serde(default)]
    pub(crate) kind: String,
    #[serde(default)]
    pub(crate) recorded_at: String,
    #[serde(default)]
    pub(crate) duration_ms: u64,
    #[serde(flatten)]
    pub(crate) extra: Value,
}

/// A long ambient/background noise take. Unlike a bulk recording it is not
/// transcribed; the server slices it into fixed-length background clips.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct BackgroundRecording {
    pub(crate) id: String,
    pub(crate) file: String,
    #[serde(default)]
    pub(crate) recorded_at: String,
    #[serde(default)]
    pub(crate) duration_ms: u64,
    #[serde(flatten)]
    pub(crate) extra: Value,
}

/// Extract a zip's entries into `dest`, rejecting any path that escapes it.
pub(crate) fn extract_zip(bytes: &[u8], dest: &Path) -> Result<(), AppError> {
    let reader = Cursor::new(bytes);
    let mut archive = ZipArchive::new(reader)
        .map_err(|error| AppError::bad_request(format!("invalid zip: {error}")))?;

    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .map_err(|error| AppError::bad_request(format!("invalid zip entry: {error}")))?;
        let name = entry.name();
        let output = safe_join(dest, name)?;
        if entry.is_dir() {
            fs::create_dir_all(&output)?;
            continue;
        }
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = File::create(output)?;
        io::copy(&mut entry, &mut file)?;
    }
    Ok(())
}

/// Read and parse `manifest.json` from an extracted bundle.
pub(crate) fn read_manifest(bundle: &Path) -> Result<Manifest, AppError> {
    let manifest_path = bundle.join("manifest.json");
    let contents = fs::read_to_string(&manifest_path)
        .map_err(|error| AppError::bad_request(format!("missing manifest: {error}")))?;
    serde_json::from_str(&contents)
        .map_err(|error| AppError::bad_request(format!("invalid manifest JSON: {error}")))
}

/// Validate the manifest's schema version, slug safety, and every clip/recording
/// entry's required fields before anything is imported.
pub(crate) fn validate_manifest(manifest: &Manifest) -> Result<(), AppError> {
    if manifest.schema_version != 1 {
        return Err(AppError::bad_request(format!(
            "unsupported schema_version: {}",
            manifest.schema_version
        )));
    }
    if !is_safe_slug(&manifest.wake_word.slug) {
        return Err(AppError::bad_request(format!(
            "unsafe wake_word.slug: {}",
            manifest.wake_word.slug
        )));
    }
    for (index, clip) in manifest.clips.iter().enumerate() {
        if clip.id.is_empty() {
            return Err(AppError::bad_request(format!("clip {index} missing id")));
        }
        if clip.file.is_empty() {
            return Err(AppError::bad_request(format!("clip {index} missing file")));
        }
        if category_for_label(&clip.label).is_none() {
            return Err(AppError::bad_request(format!(
                "clip {index} has unknown label {}",
                clip.label
            )));
        }
    }
    for (index, recording) in manifest.bulk_recordings.iter().enumerate() {
        if recording.id.is_empty() {
            return Err(AppError::bad_request(format!(
                "bulk recording {index} missing id"
            )));
        }
        if recording.file.is_empty() {
            return Err(AppError::bad_request(format!(
                "bulk recording {index} missing file"
            )));
        }
        // Only a mixed/legacy take needs a script to align against. A token
        // (`positive`) or plain `negative` take is a straight repeated read with
        // no prompt, so an empty script is expected; a `hard_negative` take does
        // carry its near-miss prompt but is not required to.
        let scripted_kind = matches!(recording.kind.trim(), "" | "mixed");
        if scripted_kind && recording.script.trim().is_empty() {
            return Err(AppError::bad_request(format!(
                "bulk recording {index} missing script"
            )));
        }
    }
    for (index, recording) in manifest.background_recordings.iter().enumerate() {
        if recording.id.is_empty() {
            return Err(AppError::bad_request(format!(
                "background recording {index} missing id"
            )));
        }
        if recording.file.is_empty() {
            return Err(AppError::bad_request(format!(
                "background recording {index} missing file"
            )));
        }
    }
    for (index, recording) in manifest.test_recordings.iter().enumerate() {
        if recording.id.is_empty() {
            return Err(AppError::bad_request(format!(
                "test recording {index} missing id"
            )));
        }
        if !recording.id.starts_with("test_") {
            return Err(AppError::bad_request(format!(
                "test recording {index} id {} must start with test_",
                recording.id
            )));
        }
        if recording.file.is_empty() {
            return Err(AppError::bad_request(format!(
                "test recording {index} missing file"
            )));
        }
        // A test take is a free-form spoken take scored against a model; it needs
        // no script, so an empty one is fine.
    }
    Ok(())
}

/// Sanity-check every WAV referenced by the manifest and return non-fatal
/// warnings (wrong channels/rate/bit-depth, clipping, silence, over-long clips).
pub(crate) fn validate_wavs(bundle: &Path, manifest: &Manifest) -> Result<Vec<String>, AppError> {
    let mut warnings = Vec::new();
    for clip in &manifest.clips {
        let path = safe_join(bundle, &clip.file)?;
        let mut reader = WavReader::open(&path).map_err(|error| {
            AppError::bad_request(format!("{}: cannot read WAV: {error}", clip.id))
        })?;
        let spec = reader.spec();
        let duration = reader.duration() as f64 / spec.sample_rate as f64;
        warnings.extend(spec_warnings(
            &clip.id,
            "",
            spec.channels,
            spec.bits_per_sample,
            spec.sample_rate,
        ));
        if duration <= 0.0 {
            warnings.push(format!("{}: zero duration", clip.id));
        }
        if duration > 5.0 {
            warnings.push(format!("{}: long clip {:.2}s", clip.id, duration));
        }

        let mut peak = 0i32;
        let mut total = 0f64;
        let mut count = 0f64;
        for sample in reader.samples::<i16>() {
            let sample = sample
                .map_err(|error| AppError::bad_request(format!("bad WAV sample: {error}")))?;
            let sample_i32 = i32::from(sample);
            peak = peak.max(sample_i32.abs());
            total += f64::from(sample) * f64::from(sample);
            count += 1.0;
        }
        let rms = if count > 0.0 {
            (total / count).sqrt()
        } else {
            0.0
        };
        warnings.extend(amplitude_warnings(&clip.id, peak, rms, duration));
    }
    for recording in &manifest.bulk_recordings {
        let path = safe_join(bundle, &recording.file)?;
        let reader = WavReader::open(&path).map_err(|error| {
            AppError::bad_request(format!("{}: cannot read bulk WAV: {error}", recording.id))
        })?;
        let spec = reader.spec();
        warnings.extend(spec_warnings(
            &recording.id,
            " bulk recording",
            spec.channels,
            spec.bits_per_sample,
            spec.sample_rate,
        ));
    }
    for recording in &manifest.background_recordings {
        let path = safe_join(bundle, &recording.file)?;
        let reader = WavReader::open(&path).map_err(|error| {
            AppError::bad_request(format!(
                "{}: cannot read background WAV: {error}",
                recording.id
            ))
        })?;
        let spec = reader.spec();
        warnings.extend(spec_warnings(
            &recording.id,
            " background recording",
            spec.channels,
            spec.bits_per_sample,
            spec.sample_rate,
        ));
    }
    for recording in &manifest.test_recordings {
        let path = safe_join(bundle, &recording.file)?;
        let reader = WavReader::open(&path).map_err(|error| {
            AppError::bad_request(format!("{}: cannot read test WAV: {error}", recording.id))
        })?;
        let spec = reader.spec();
        warnings.extend(spec_warnings(
            &recording.id,
            " test recording",
            spec.channels,
            spec.bits_per_sample,
            spec.sample_rate,
        ));
    }
    Ok(warnings)
}

/// Warn about a WAV whose format is not mono 16-bit 16 kHz PCM. `noun` names the
/// recording kind in the message (e.g. `" bulk recording"`, or `""` for a clip),
/// so the four validation loops share one wording. Pure: derives the warnings
/// from the header fields alone.
fn spec_warnings(
    id: &str,
    noun: &str,
    channels: u16,
    bits_per_sample: u16,
    sample_rate: u32,
) -> Vec<String> {
    let mut warnings = Vec::new();
    if channels != 1 {
        warnings.push(format!("{id}: expected mono{noun}, got {channels} channels"));
    }
    if bits_per_sample != 16 {
        warnings.push(format!(
            "{id}: expected 16-bit PCM{noun}, got {bits_per_sample} bits"
        ));
    }
    if sample_rate != 16_000 {
        warnings.push(format!("{id}: expected 16000 Hz{noun}, got {sample_rate}"));
    }
    warnings
}

/// Warn about a clip's amplitude: near-full-scale peaks (possible clipping) and a
/// very low RMS on non-empty audio (near silence). Pure: judges the two scalar
/// measurements the sample scan produced.
fn amplitude_warnings(id: &str, peak: i32, rms: f64, duration: f64) -> Vec<String> {
    let mut warnings = Vec::new();
    if peak >= 32_760 {
        warnings.push(format!("{id}: possible clipping, peak={peak}"));
    }
    if rms < 50.0 && duration > 0.0 {
        warnings.push(format!("{id}: very quiet audio, rms={rms:.1}"));
    }
    warnings
}

/// Import the pre-cut short clips into `data/real/<slug>/<category>`, upserting
/// the project and one DB row per newly copied clip. Returns the count imported.
pub(crate) fn import_bundle(
    bundle: &Path,
    manifest: &Manifest,
    data_root: &Path,
    db: &Mutex<rusqlite::Connection>,
) -> Result<usize, AppError> {
    let slug = manifest.wake_word.slug.clone();
    let phrase = manifest.wake_word.phrase();
    let external_id = manifest
        .wake_word
        .extra
        .get("id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty());
    let dest_root = data_root.join(&slug);
    fs::create_dir_all(&dest_root)?;

    let now = now_ms();
    {
        let conn = db.lock().expect("db lock poisoned");
        db::upsert_project(&conn, &slug, &phrase, external_id, now).map_err(crate::error::db_error)?;
    }

    let mut imported = 0;
    for clip in &manifest.clips {
        let category = category_for_label(&clip.label).expect("validated label");
        let src = safe_join(bundle, &clip.file)?;
        if !src.is_file() {
            return Err(AppError::bad_request(format!(
                "missing clip file: {}",
                clip.file
            )));
        }

        let dest_dir = dest_root.join(category);
        fs::create_dir_all(&dest_dir)?;
        let clip_phrase = if clip.spoken_phrase.is_empty() {
            clip.label.as_str()
        } else {
            clip.spoken_phrase.as_str()
        };
        let dest = dest_dir.join(format!("{}_{}.wav", clip.id, safe_filename(clip_phrase)));
        if dest.exists() {
            continue;
        }

        fs::copy(&src, &dest)?;
        let row = db::ClipRow {
            id: clip.id.clone(),
            project_slug: slug.clone(),
            label: clip.label.clone(),
            category: category.to_string(),
            spoken_phrase: clip_phrase.to_string(),
            wav_path: dest.to_string_lossy().to_string(),
            source_file: clip.file.clone(),
            bundle: Some("server_sync".to_string()),
        };
        let conn = db.lock().expect("db lock poisoned");
        if db::insert_clip(&conn, &row, now).map_err(crate::error::db_error)? {
            imported += 1;
        }
    }
    Ok(imported)
}

/// Pull per-take capture provenance out of a recording's flattened `extra`
/// JSON. The app nests it under a `capture` object; missing or empty values
/// become `None` so they never overwrite prior provenance on reprocess.
pub(crate) fn capture_from_extra(extra: &Value) -> db::CaptureMeta {
    let capture = extra.get("capture");
    let text = |key: &str| {
        capture
            .and_then(|value| value.get(key))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToString::to_string)
    };
    let number = |key: &str| {
        capture
            .and_then(|value| value.get(key))
            .and_then(Value::as_i64)
            .filter(|value| *value > 0)
    };
    db::CaptureMeta {
        device_manufacturer: text("device_manufacturer"),
        device_model: text("device_model"),
        os_version: text("os_version"),
        app_version: text("app_version"),
        input_route: text("input_route"),
        source_sample_rate_hz: number("source_sample_rate_hz"),
        source_channels: number("source_channels"),
        session_id: text("session_id"),
    }
}

/// Persist a recording's source WAV under `bulk_source/` and return both the
/// stored path and its full-file SHA-256. The checksum lets a device ask which
/// takes the server already holds and skip re-uploading unchanged ones.
pub(crate) fn store_bulk_source(
    dest_root: &Path,
    recording_id: &str,
    source: &Path,
) -> Result<(PathBuf, String), AppError> {
    let dir = dest_root.join("bulk_source");
    fs::create_dir_all(&dir)?;
    let dest = dir.join(format!("{}.wav", safe_filename(recording_id)));
    // Reprocess passes the already-stored source as the input; copying a file
    // onto itself would truncate it to zero bytes, so skip when they match.
    let same = match (source.canonicalize(), dest.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    };
    if !same {
        fs::copy(source, &dest)?;
    }
    let sha256 = file_sha256(&dest)?;
    Ok((dest, sha256))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_from_extra_reads_nested_capture_and_skips_blanks() {
        let extra = serde_json::json!({
            "session_id": "top-level-ignored",
            "capture": {
                "device_manufacturer": "Google",
                "device_model": "Pixel 7",
                "input_route": "builtin_mic: Pixel 7",
                "source_sample_rate_hz": 48000,
                "source_channels": 1,
                "session_id": "sess-abc",
                "os_version": "",
                "app_version": null,
            }
        });
        let capture = capture_from_extra(&extra);
        assert_eq!(capture.device_manufacturer.as_deref(), Some("Google"));
        assert_eq!(capture.device_model.as_deref(), Some("Pixel 7"));
        assert_eq!(capture.input_route.as_deref(), Some("builtin_mic: Pixel 7"));
        assert_eq!(capture.source_sample_rate_hz, Some(48000));
        assert_eq!(capture.source_channels, Some(1));
        assert_eq!(capture.session_id.as_deref(), Some("sess-abc"));
        // Empty string and null collapse to None so reprocess never clobbers.
        assert_eq!(capture.os_version, None);
        assert_eq!(capture.app_version, None);
    }

    #[test]
    fn capture_from_extra_absent_object_is_all_none() {
        let capture = capture_from_extra(&serde_json::json!({ "notes": "x" }));
        assert!(capture.device_model.is_none());
        assert!(capture.source_sample_rate_hz.is_none());
        assert!(capture.session_id.is_none());
    }

    #[test]
    fn spec_warnings_flags_each_wrong_field_with_noun() {
        // A well-formed clip header (mono, 16-bit, 16 kHz) yields no warnings.
        assert!(spec_warnings("c1", "", 1, 16, 16_000).is_empty());
        // Each wrong field produces its own message; the noun is spliced in.
        let w = spec_warnings("c2", "", 2, 8, 44_100);
        assert_eq!(
            w,
            vec![
                "c2: expected mono, got 2 channels".to_string(),
                "c2: expected 16-bit PCM, got 8 bits".to_string(),
                "c2: expected 16000 Hz, got 44100".to_string(),
            ]
        );
        // The noun distinguishes the recording kind in the wording.
        let w = spec_warnings("b1", " bulk recording", 2, 16, 16_000);
        assert_eq!(w, vec!["b1: expected mono bulk recording, got 2 channels"]);
    }

    #[test]
    fn amplitude_warnings_flags_clipping_and_quiet() {
        // Healthy audio: mid peak, ample RMS, positive duration -> nothing.
        assert!(amplitude_warnings("c1", 10_000, 500.0, 1.0).is_empty());
        // Near-full-scale peak is flagged as possible clipping.
        assert_eq!(
            amplitude_warnings("c2", 32_760, 500.0, 1.0),
            vec!["c2: possible clipping, peak=32760"]
        );
        // Low RMS on non-empty audio is flagged as very quiet.
        assert_eq!(
            amplitude_warnings("c3", 100, 49.9, 1.0),
            vec!["c3: very quiet audio, rms=49.9"]
        );
        // A quiet reading with zero duration is not flagged (the guard requires
        // duration > 0.0).
        assert!(amplitude_warnings("c4", 100, 0.0, 0.0).is_empty());
    }

    fn manifest_from(value: serde_json::Value) -> Manifest {
        serde_json::from_value(value).expect("valid manifest json")
    }

    #[test]
    fn validate_manifest_accepts_minimal_valid_manifest() {
        let manifest = manifest_from(serde_json::json!({
            "schema_version": 1,
            "wake_word": { "slug": "all_set", "phrase": "all set" },
            "clips": [],
        }));
        assert!(validate_manifest(&manifest).is_ok());
    }

    #[test]
    fn validate_manifest_rejects_bad_schema_and_slug() {
        let bad_version = manifest_from(serde_json::json!({
            "schema_version": 2,
            "wake_word": { "slug": "all_set" },
            "clips": [],
        }));
        assert!(validate_manifest(&bad_version).is_err());

        let bad_slug = manifest_from(serde_json::json!({
            "schema_version": 1,
            "wake_word": { "slug": "../escape" },
            "clips": [],
        }));
        assert!(validate_manifest(&bad_slug).is_err());
    }

    #[test]
    fn validate_manifest_rejects_bad_clip_and_unknown_label() {
        let missing_file = manifest_from(serde_json::json!({
            "schema_version": 1,
            "wake_word": { "slug": "all_set" },
            "clips": [{ "id": "c1", "file": "", "label": "positive" }],
        }));
        assert!(validate_manifest(&missing_file).is_err());

        let bad_label = manifest_from(serde_json::json!({
            "schema_version": 1,
            "wake_word": { "slug": "all_set" },
            "clips": [{ "id": "c1", "file": "a.wav", "label": "nonsense" }],
        }));
        assert!(validate_manifest(&bad_label).is_err());
    }

    #[test]
    fn validate_manifest_bulk_script_required_only_for_mixed_kinds() {
        // A mixed/legacy take with no script is rejected.
        let mixed_no_script = manifest_from(serde_json::json!({
            "schema_version": 1,
            "wake_word": { "slug": "all_set" },
            "clips": [],
            "bulk_recordings": [{ "id": "b1", "file": "b1.wav", "script": "  ", "kind": "" }],
        }));
        assert!(validate_manifest(&mixed_no_script).is_err());

        // A straight positive take needs no script.
        let positive_no_script = manifest_from(serde_json::json!({
            "schema_version": 1,
            "wake_word": { "slug": "all_set" },
            "clips": [],
            "bulk_recordings": [{ "id": "b1", "file": "b1.wav", "script": "", "kind": "positive" }],
        }));
        assert!(validate_manifest(&positive_no_script).is_ok());
    }

    #[test]
    fn validate_manifest_test_ids_must_be_prefixed() {
        let bad_prefix = manifest_from(serde_json::json!({
            "schema_version": 1,
            "wake_word": { "slug": "all_set" },
            "clips": [],
            "test_recordings": [{ "id": "rec1", "file": "t1.wav", "script": "" }],
        }));
        assert!(validate_manifest(&bad_prefix).is_err());

        let good = manifest_from(serde_json::json!({
            "schema_version": 1,
            "wake_word": { "slug": "all_set" },
            "clips": [],
            "test_recordings": [{ "id": "test_rec1", "file": "t1.wav", "script": "" }],
        }));
        assert!(validate_manifest(&good).is_ok());
    }
}
