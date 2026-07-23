//! Whisper transcription client and word-normalization helpers.
//!
//! Speech takes are aligned from Whisper word timestamps, so this module owns
//! both the HTTP call to the Whisper server (`transcribe_with_words`) and the
//! small token-normalization helpers that let the slicing code match a spoken
//! transcript against a wake phrase despite casing, punctuation, and Whisper's
//! per-segment timestamp quirks.

use crate::error::AppError;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// The Whisper server's `verbose_json` reply: the full transcript plus the
/// per-segment word timings we actually align against.
#[derive(Debug, Deserialize)]
pub(crate) struct WhisperResponse {
    #[serde(default)]
    pub(crate) text: String,
    #[serde(default)]
    pub(crate) segments: Vec<WhisperSegment>,
}

/// One Whisper segment. `start`/`end` bound the segment; `words` carries the
/// word-level timings, which occasionally need the segment offset re-applied.
#[derive(Debug, Deserialize)]
pub(crate) struct WhisperSegment {
    #[serde(default)]
    pub(crate) start: f64,
    #[serde(default)]
    pub(crate) end: f64,
    #[serde(default)]
    pub(crate) words: Vec<WhisperWord>,
}

/// A single transcribed word with its start/end (seconds) and Whisper's
/// confidence. Shared well beyond transcription: the slicing math cuts on these
/// times and the alignment API echoes them back to the app.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct WhisperWord {
    pub(crate) word: String,
    pub(crate) start: f64,
    pub(crate) end: f64,
    #[serde(default)]
    pub(crate) probability: f64,
}

/// POST a WAV to the Whisper server and parse its verbose-JSON transcript.
pub(crate) async fn transcribe_with_words(
    whisper_url: &str,
    wav_path: &Path,
    prompt: Option<&str>,
) -> Result<WhisperResponse, AppError> {
    let bytes = fs::read(wav_path)?;
    let file_name = wav_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("audio.wav")
        .to_string();
    let part = reqwest::multipart::Part::bytes(bytes)
        .file_name(file_name)
        .mime_str("audio/wav")
        .map_err(|error| AppError::internal(format!("prepare Whisper upload: {error}")))?;
    // The prompt is deliberately omitted for alignment and verification: on this
    // clean, scripted audio it does not improve accuracy and it biases Whisper
    // toward reporting the scripted words even where the audio differs.
    let mut form = reqwest::multipart::Form::new()
        .part("file", part)
        .text("response_format", "verbose_json")
        .text("temperature", "0.0")
        .text("no_context", "true")
        .text("word_timestamps", "true");
    if let Some(prompt) = prompt.filter(|value| !value.trim().is_empty()) {
        form = form.text("prompt", prompt.to_string());
    }
    let endpoint = format!("{}/inference", whisper_url.trim_end_matches('/'));
    let response = reqwest::Client::new()
        .post(endpoint)
        .multipart(form)
        .send()
        .await
        .map_err(|error| AppError::internal(format!("Whisper request failed: {error}")))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| AppError::internal(format!("Whisper response read failed: {error}")))?;
    if !status.is_success() {
        return Err(AppError::internal(format!(
            "Whisper returned {status}: {}",
            body.trim()
        )));
    }
    serde_json::from_str(&body)
        .map_err(|error| AppError::internal(format!("Whisper response JSON failed: {error}")))
}

/// Flatten a response's segments into a single word stream, re-applying the
/// per-segment offset where Whisper reported word times relative to the segment.
pub(crate) fn whisper_words(response: &WhisperResponse) -> Vec<WhisperWord> {
    response
        .segments
        .iter()
        .flat_map(|segment| {
            let offset = word_timestamp_offset(segment);
            segment.words.iter().cloned().map(move |mut word| {
                if offset > 0.0 {
                    word.start += offset;
                    word.end += offset;
                }
                word
            })
        })
        .filter(|word| word.end >= word.start)
        .collect()
}

/// Detect the case where a segment's word timestamps are segment-relative (both
/// the first word's start and last word's end lag the segment by >0.5s) and
/// return the offset to add back; otherwise 0.
pub(crate) fn word_timestamp_offset(segment: &WhisperSegment) -> f64 {
    let Some(first_word) = segment.words.first() else {
        return 0.0;
    };
    let Some(last_word) = segment.words.last() else {
        return 0.0;
    };
    let start_delta = segment.start - first_word.start;
    let end_delta = segment.end - last_word.end;
    if start_delta > 0.5 && end_delta > 0.5 {
        start_delta
    } else {
        0.0
    }
}

/// Normalize a spoken token for phrase matching: lowercase, keep only letters
/// and digits. Whisper emits leading spaces and stray punctuation on words.
pub(crate) fn normalize_token(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Trim surrounding non-alphanumerics and lowercase a word for sequence matching.
pub(crate) fn normalize_word(value: &str) -> String {
    value
        .trim()
        .trim_matches(|ch: char| !ch.is_ascii_alphanumeric())
        .to_ascii_lowercase()
}

/// Split and normalize an arbitrary phrase into its non-empty words.
pub(crate) fn normalized_words(value: &str) -> Vec<String> {
    value
        .split_whitespace()
        .map(normalize_word)
        .filter(|word| !word.is_empty())
        .collect()
}

/// True when the wake phrase appears at (or very near) the END of the normalized
/// transcript. Positives are tail-aligned, so the wake phrase must be the last
/// thing spoken; `TAIL_SLACK` words of trailing audio are tolerated for the tail
/// padding. Requiring the phrase at the tail — not just anywhere — rejects
/// slices that were cut too early and only contain the lead-in (e.g. "the next
/// words are ...") even when a flaky Whisper pass imagines the phrase mid-slice.
pub(crate) fn transcript_tail_has_phrase(text: &str, phrase_words: &[String]) -> bool {
    const TAIL_SLACK: usize = 2;
    if phrase_words.is_empty() {
        return false;
    }
    let heard = normalized_words(text);
    if heard.len() < phrase_words.len() {
        return false;
    }
    let last_start = heard.len() - phrase_words.len();
    for start in 0..=last_start {
        if &heard[start..start + phrase_words.len()] == phrase_words {
            let words_after = heard.len() - (start + phrase_words.len());
            if words_after <= TAIL_SLACK {
                return true;
            }
        }
    }
    false
}

/// True if `words` contains `phrase` as a contiguous run. Used to spot the
/// near-miss cue words that flag a wake phrase as a hard negative.
pub(crate) fn contains_word_sequence(words: &[String], phrase: &[&str]) -> bool {
    words
        .windows(phrase.len())
        .any(|window| window.iter().map(String::as_str).eq(phrase.iter().copied()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcript_tail_has_phrase_requires_phrase_at_end() {
        let phrase = normalized_words("all set");
        // Phrase at the tail (with a little trailing slack) is accepted.
        assert!(transcript_tail_has_phrase("the next words are all set.", &phrase));
        assert!(transcript_tail_has_phrase("All Set!", &phrase));
        assert!(transcript_tail_has_phrase("say all set now", &phrase));
        // Phrase absent, or buried mid-slice by a flaky pass, is rejected.
        assert!(!transcript_tail_has_phrase("the next words are over", &phrase));
        assert!(!transcript_tail_has_phrase(
            "the next words are all in the same sentence",
            &phrase
        ));
        assert!(!transcript_tail_has_phrase("all is not set", &phrase));
    }

    fn word(text: &str, start: f64, end: f64) -> WhisperWord {
        WhisperWord {
            word: text.to_string(),
            start,
            end,
            probability: 1.0,
        }
    }

    #[test]
    fn normalize_token_keeps_alnum_lowercased() {
        assert_eq!(normalize_token(" All!"), "all");
        assert_eq!(normalize_token("don't"), "dont");
        assert_eq!(normalize_token("Set."), "set");
        assert_eq!(normalize_token("42%"), "42");
        assert_eq!(normalize_token("--"), "");
    }

    #[test]
    fn normalize_word_trims_edges_only() {
        // Only surrounding non-alphanumerics are trimmed; interior stays.
        assert_eq!(normalize_word("  'All-Set!' "), "all-set");
        assert_eq!(normalize_word("Set."), "set");
        assert_eq!(normalize_word("...!"), "");
    }

    #[test]
    fn normalized_words_splits_and_drops_empties() {
        assert_eq!(
            normalized_words("  All,  Set!  "),
            vec!["all".to_string(), "set".to_string()]
        );
        assert!(normalized_words("   ").is_empty());
    }

    #[test]
    fn word_timestamp_offset_detects_segment_relative_times() {
        // Both edges lag the segment by > 0.5s -> the words are segment-relative,
        // so the offset (segment.start - first_word.start) is returned.
        let seg = WhisperSegment {
            start: 10.0,
            end: 12.0,
            words: vec![word("all", 0.1, 0.4), word("set", 0.9, 1.4)],
        };
        assert!((word_timestamp_offset(&seg) - 9.9).abs() < 1e-9);

        // Already-absolute times (edges within 0.5s of the segment) -> no offset.
        let seg = WhisperSegment {
            start: 10.0,
            end: 12.0,
            words: vec![word("all", 10.1, 10.4), word("set", 11.6, 11.9)],
        };
        assert_eq!(word_timestamp_offset(&seg), 0.0);

        // No words -> no offset.
        let seg = WhisperSegment { start: 1.0, end: 2.0, words: vec![] };
        assert_eq!(word_timestamp_offset(&seg), 0.0);
    }

    #[test]
    fn whisper_words_reapplies_offset_and_drops_inverted() {
        let response = WhisperResponse {
            text: String::new(),
            segments: vec![WhisperSegment {
                start: 10.0,
                end: 12.0,
                words: vec![word("all", 0.1, 0.4), word("set", 0.9, 1.4)],
            }],
        };
        let words = whisper_words(&response);
        assert_eq!(words.len(), 2);
        // The 9.9s offset is added back to both edges.
        assert!((words[0].start - 10.0).abs() < 1e-9);
        assert!((words[1].end - 11.3).abs() < 1e-9);
    }

    #[test]
    fn contains_word_sequence_finds_contiguous_run() {
        let words: Vec<String> = ["not", "the", "wake", "phrase", "all", "set"]
            .iter()
            .map(|w| w.to_string())
            .collect();
        assert!(contains_word_sequence(&words, &["wake", "phrase"]));
        assert!(contains_word_sequence(&words, &["not", "the", "wake"]));
        // Non-contiguous words are not a match.
        assert!(!contains_word_sequence(&words, &["not", "wake"]));
        assert!(!contains_word_sequence(&words, &["missing"]));
    }
}
