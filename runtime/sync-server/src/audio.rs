//! WAV IO helpers: duration probing, slice extraction, file hashing, and the
//! slice-row builder.
//!
//! The slicing and alignment code needs to read a take's duration, carve a
//! `[start, end)` span out into a new mono 16-bit WAV, checksum a file, and turn
//! a run of transcribed words into the database row that describes the resulting
//! clip. These are the low-level, audio-touching primitives behind those steps.

use crate::db;
use crate::error::AppError;
use crate::whisper::WhisperWord;
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use sha2::{Digest, Sha256};
use std::fs;
use std::io;
use std::path::Path;

/// Duration of a WAV in seconds, or 0.0 for a zero-rate header.
pub(crate) fn wav_duration_seconds(path: &Path) -> Result<f64, AppError> {
    let reader = WavReader::open(path)
        .map_err(|error| AppError::bad_request(format!("cannot read WAV duration: {error}")))?;
    let spec = reader.spec();
    if spec.sample_rate == 0 {
        return Ok(0.0);
    }
    Ok(reader.duration() as f64 / spec.sample_rate as f64)
}

/// Streaming SHA-256 of a file's raw bytes, hex-encoded. Matches the digest the
/// Android client computes over the same WAV so the two can be compared.
pub(crate) fn file_sha256(path: &Path) -> Result<String, AppError> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

/// Extract `[start_sec, end_sec)` of `source` into a fresh mono 16-bit WAV at
/// `dest`. Returns `false` (writing nothing) when the span is empty. Requires
/// the source be mono 16-bit PCM, the format every take is stored in.
pub(crate) fn write_wav_slice(
    source: &Path,
    dest: &Path,
    start_sec: f64,
    end_sec: f64,
) -> Result<bool, AppError> {
    let mut reader = WavReader::open(source)
        .map_err(|error| AppError::bad_request(format!("cannot read WAV for slicing: {error}")))?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_format != SampleFormat::Int || spec.bits_per_sample != 16 {
        return Err(AppError::bad_request(
            "bulk slicing requires mono 16-bit PCM WAV",
        ));
    }
    let sample_rate = spec.sample_rate as f64;
    let start_sample = (start_sec * sample_rate).floor().max(0.0) as usize;
    let end_sample = (end_sec * sample_rate).ceil().max(start_sample as f64) as usize;
    if end_sample <= start_sample {
        return Ok(false);
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    let samples: Vec<i16> = reader
        .samples::<i16>()
        .skip(start_sample)
        .take(end_sample - start_sample)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| AppError::bad_request(format!("bad WAV sample while slicing: {error}")))?;
    if samples.is_empty() {
        return Ok(false);
    }
    let mut writer = WavWriter::create(
        dest,
        WavSpec {
            channels: 1,
            sample_rate: spec.sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        },
    )
    .map_err(|error| AppError::internal(format!("create WAV slice: {error}")))?;
    for sample in samples {
        writer
            .write_sample(sample)
            .map_err(|error| AppError::internal(format!("write WAV slice: {error}")))?;
    }
    writer
        .finalize()
        .map_err(|error| AppError::internal(format!("finish WAV slice: {error}")))?;
    Ok(true)
}

/// Build the database row for one written slice from its cut bounds and the
/// words it contains: joins the words into a spoken phrase and averages their
/// Whisper confidence.
pub(crate) fn build_slice_row(
    clip_id: &str,
    label: &str,
    category: &str,
    dest: &Path,
    file_name: &str,
    start_sec: f64,
    end_sec: f64,
    words: &[WhisperWord],
) -> db::SliceRow {
    let spoken_phrase = words
        .iter()
        .map(|word| word.word.trim())
        .collect::<Vec<_>>()
        .join(" ");
    let avg_probability = if words.is_empty() {
        0.0
    } else {
        words.iter().map(|word| word.probability).sum::<f64>() / words.len() as f64
    };
    db::SliceRow {
        id: clip_id.to_string(),
        label: label.to_string(),
        category: category.to_string(),
        spoken_phrase,
        source_start_ms: (start_sec * 1000.0).round() as i64,
        source_end_ms: (end_sec * 1000.0).round() as i64,
        avg_probability,
        word_count: words.len() as i64,
        wav_path: dest.to_string_lossy().to_string(),
        file_name: file_name.to_string(),
    }
}
