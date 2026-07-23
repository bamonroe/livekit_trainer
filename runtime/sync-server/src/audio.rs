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

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    /// Write a mono 16-bit PCM WAV of `n` samples (a simple ramp) at `rate` Hz.
    fn write_test_wav(path: &Path, rate: u32, n: usize) {
        let mut writer = WavWriter::create(
            path,
            WavSpec {
                channels: 1,
                sample_rate: rate,
                bits_per_sample: 16,
                sample_format: SampleFormat::Int,
            },
        )
        .expect("create wav");
        for i in 0..n {
            writer.write_sample((i % 100) as i16).expect("write sample");
        }
        writer.finalize().expect("finalize");
    }

    fn temp_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("lkww_audio_{tag}_{}.wav", Utc::now().timestamp_nanos_opt().unwrap_or(0)))
    }

    #[test]
    fn wav_duration_seconds_reports_length() {
        let path = temp_path("dur");
        // 16000 samples at 16 kHz = exactly 1.0s.
        write_test_wav(&path, 16_000, 16_000);
        let dur = wav_duration_seconds(&path).expect("duration");
        assert!((dur - 1.0).abs() < 1e-9);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn write_wav_slice_extracts_span_and_rejects_empty() {
        let src = temp_path("src");
        write_test_wav(&src, 16_000, 16_000); // 1.0s
        let dest = temp_path("slice");
        // Cut [0.25, 0.75) = 0.5s -> 8000 samples.
        let wrote = write_wav_slice(&src, &dest, 0.25, 0.75).expect("slice");
        assert!(wrote);
        let dur = wav_duration_seconds(&dest).expect("slice duration");
        assert!((dur - 0.5).abs() < 1e-3, "got {dur}");
        // An empty span writes nothing and reports false.
        let empty = temp_path("empty");
        assert!(!write_wav_slice(&src, &empty, 0.5, 0.5).expect("empty slice"));
        assert!(!empty.exists());
        let _ = fs::remove_file(&src);
        let _ = fs::remove_file(&dest);
    }

    #[test]
    fn file_sha256_is_stable_and_content_dependent() {
        let a = temp_path("sha_a");
        let b = temp_path("sha_b");
        write_test_wav(&a, 16_000, 1000);
        write_test_wav(&b, 16_000, 1000);
        // Identical content -> identical digest; hex-encoded 32-byte SHA-256.
        let da = file_sha256(&a).expect("sha a");
        let db_ = file_sha256(&b).expect("sha b");
        assert_eq!(da, db_);
        assert_eq!(da.len(), 64);
        // Different content -> different digest.
        let c = temp_path("sha_c");
        write_test_wav(&c, 16_000, 2000);
        assert_ne!(da, file_sha256(&c).expect("sha c"));
        let _ = fs::remove_file(&a);
        let _ = fs::remove_file(&b);
        let _ = fs::remove_file(&c);
    }

    #[test]
    fn build_slice_row_joins_words_and_averages_probability() {
        let words = [
            WhisperWord { word: " all".to_string(), start: 0.1, end: 0.4, probability: 0.8 },
            WhisperWord { word: "set ".to_string(), start: 0.5, end: 0.9, probability: 0.6 },
        ];
        let row = build_slice_row(
            "clip1",
            "positive",
            "positive",
            Path::new("/tmp/clip1.wav"),
            "clip1.wav",
            0.1,
            0.9,
            &words,
        );
        assert_eq!(row.spoken_phrase, "all set");
        assert_eq!(row.word_count, 2);
        assert!((row.avg_probability - 0.7).abs() < 1e-9);
        assert_eq!(row.source_start_ms, 100);
        assert_eq!(row.source_end_ms, 900);
        // No words -> empty phrase and zero average.
        let empty = build_slice_row(
            "c",
            "background",
            "background",
            Path::new("/tmp/c.wav"),
            "c.wav",
            0.0,
            2.0,
            &[],
        );
        assert_eq!(empty.spoken_phrase, "");
        assert_eq!(empty.avg_probability, 0.0);
    }
}
