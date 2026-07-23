//! Pure slicing math: turning Whisper word timings into cut boundaries.
//!
//! Everything here is side-effect free — it operates on word lists and numeric
//! spans, never on audio files or the database — so the cut geometry (phrase
//! location, context lead-in, padding, hard-max clamping, negative chunking,
//! background windows, and the energy/VAD fallback) is unit-testable in
//! isolation. The alignment code composes these with the audio and db modules.

use crate::constants::{
    BACKGROUND_CHUNK_SECONDS, BACKGROUND_MIN_CHUNK_SECONDS, ENERGY_CLOSE_FRACTION,
    ENERGY_FRAME_SECONDS, ENERGY_LEAD_PADDING_SECONDS, ENERGY_MERGE_GAP_SECONDS,
    ENERGY_MIN_BURST_SECONDS, ENERGY_OPEN_FRACTION, ENERGY_TAIL_PADDING_SECONDS, MAX_SLICE_SECONDS,
    NEGATIVE_TARGET_SECONDS, POSITIVE_MAX_SECONDS,
};
use crate::whisper::{contains_word_sequence, normalize_word, WhisperWord};
use serde_json::json;
use sha2::{Digest, Sha256};

/// Every `(first, last)` word-index range whose normalized words equal the wake
/// phrase, in order.
pub(crate) fn phrase_ranges(words: &[WhisperWord], phrase_words: &[String]) -> Vec<(usize, usize)> {
    let normalized: Vec<String> = words
        .iter()
        .map(|word| normalize_word(&word.word))
        .collect();
    let mut ranges = Vec::new();
    for start in 0..normalized.len() {
        let end = start + phrase_words.len();
        if end <= normalized.len() && normalized[start..end] == *phrase_words {
            ranges.push((start, end - 1));
        }
    }
    ranges
}

/// Pick the earliest word to include as lead-in context for a tail-aligned
/// positive, filling up to POSITIVE_MAX_SECONDS of speech before the phrase end.
pub(crate) fn positive_context_first(words: &[WhisperWord], first: usize, last: usize) -> usize {
    let anchor_end = words[last].end;
    let earliest_start = (anchor_end - POSITIVE_MAX_SECONDS).max(0.0);
    let mut context_first = first;
    while context_first > 0 && words[context_first - 1].start >= earliest_start {
        context_first -= 1;
    }
    context_first
}

/// Expand a word-index range into padded second bounds. The start is nudged
/// earlier and the end later so onsets and (crucially) the wake-phrase tail are
/// not clipped.
///
/// `clamp_tail_to_neighbor` controls how far the trailing edge may grow. For
/// negatives it is `true`: the end must not run into the next word, so a
/// negative can never accidentally swallow an adjacent wake phrase. For
/// positives it is `false`: the wake phrase sits at the very end and Whisper
/// routinely places its end (and the following word's start) too early, so the
/// tail is allowed to grow past the neighbor up to the recording end — capturing
/// a positive that would otherwise lose its wake phrase matters more than a
/// little trailing audio.
pub(crate) fn padded_bounds(
    words: &[WhisperWord],
    first: usize,
    last: usize,
    source_end: f64,
    lead: f64,
    tail: f64,
    clamp_tail_to_neighbor: bool,
) -> (f64, f64) {
    let raw_start = words[first].start.max(0.0);
    let prev_end = if first > 0 {
        words[first - 1].end.max(0.0)
    } else {
        0.0
    };
    // Move the start earlier, but not into the previous word.
    let start_floor = prev_end.min(raw_start);
    let start = (raw_start - lead).clamp(start_floor, raw_start);

    let raw_end = words[last].end.max(raw_start);
    let neighbor_cap = if clamp_tail_to_neighbor && last + 1 < words.len() {
        words[last + 1].start
    } else {
        source_end
    };
    let end_cap = neighbor_cap.max(raw_end).min(source_end.max(raw_end));
    let end = (raw_end + tail).clamp(raw_end, end_cap);
    (start, end)
}

/// Enforce the hard maximum slice length on already-padded bounds. Positives
/// pass `keep_tail = true` so the wake phrase (which ends the clip) is never
/// trimmed — the start moves in instead; negatives keep their start and move
/// the end in.
pub(crate) fn clamp_slice_span(start: f64, end: f64, keep_tail: bool) -> (f64, f64) {
    if end - start <= MAX_SLICE_SECONDS {
        return (start, end);
    }
    if keep_tail {
        (end - MAX_SLICE_SECONDS, end)
    } else {
        (start, start + MAX_SLICE_SECONDS)
    }
}

/// The still-audible sub-range of `first..=last` after the bounds were clamped,
/// so the stored transcript/word timings match what the slice actually contains
/// instead of over-claiming words that got trimmed off.
pub(crate) fn visible_range(
    words: &[WhisperWord],
    first: usize,
    last: usize,
    start: f64,
    end: f64,
) -> (usize, usize) {
    let mut visible_first = first;
    while visible_first < last && words[visible_first].end <= start {
        visible_first += 1;
    }
    let mut visible_last = last;
    while visible_last > visible_first && words[visible_last].start >= end {
        visible_last -= 1;
    }
    (visible_first, visible_last)
}

/// True when the wake phrase at `first..=last` sits inside an explicit near-miss
/// cue frame (e.g. "not the wake phrase ..."), so it should be filed as a hard
/// negative rather than a positive.
pub(crate) fn is_hard_negative_context(words: &[WhisperWord], first: usize, last: usize) -> bool {
    let context_start = first.saturating_sub(8);
    let context_end = (last + 5).min(words.len().saturating_sub(1));
    let context = words[context_start..=context_end]
        .iter()
        .map(|word| normalize_word(&word.word))
        .collect::<Vec<_>>();
    contains_word_sequence(&context, &["near", "match"])
        || contains_word_sequence(&context, &["hard", "negative"])
        || contains_word_sequence(&context, &["not", "the", "wake", "phrase"])
        || contains_word_sequence(&context, &["similar", "phrase"])
}

/// Chop the words not already claimed by a positive/hard-negative cut into
/// negative word-chunks bounded by NEGATIVE_TARGET_SECONDS. Returns each chunk's
/// `(start_sec, end_sec, first_word, last_word)`.
pub(crate) fn negative_ranges(
    words: &[WhisperWord],
    occupied: &[(f64, f64)],
) -> Vec<(f64, f64, usize, usize)> {
    let mut ranges = Vec::new();
    let mut index = 0;
    while index < words.len() {
        if word_overlaps_ranges(&words[index], occupied) {
            index += 1;
            continue;
        }

        let start_word = index;
        if words[start_word].end - words[start_word].start > NEGATIVE_TARGET_SECONDS {
            index += 1;
            continue;
        }

        let mut end_word = start_word;
        while end_word + 1 < words.len()
            && !word_overlaps_ranges(&words[end_word + 1], occupied)
            && words[end_word + 1].end - words[start_word].start <= NEGATIVE_TARGET_SECONDS
        {
            end_word += 1;
        }

        ranges.push((
            words[start_word].start.max(0.0),
            words[end_word].end,
            start_word,
            end_word,
        ));
        index = end_word + 1;
    }
    ranges
}

/// True if a word's time span overlaps any of the occupied `(start, end)` spans.
pub(crate) fn word_overlaps_ranges(word: &WhisperWord, ranges: &[(f64, f64)]) -> bool {
    ranges
        .iter()
        .any(|(start, end)| word.start < *end && word.end > *start)
}

/// Deterministic clip id for one slice, hashed over the recording identity, the
/// cut span, and the exact words. Stable across re-sync/reprocess so the same
/// cut yields the same id (and DB row) every time.
pub(crate) fn bulk_clip_hash_id(
    recording_id: &str,
    recorded_at: &str,
    recording_duration_ms: u64,
    category: &str,
    start_sec: f64,
    end_sec: f64,
    words: &[WhisperWord],
) -> String {
    let spoken_phrase = words
        .iter()
        .map(|word| word.word.trim())
        .collect::<Vec<_>>()
        .join(" ");
    let duration_ms = ((end_sec - start_sec).max(0.0) * 1000.0).round() as u64;
    let input = json!({
        "recording_id": recording_id,
        "recorded_at": recorded_at,
        "recording_duration_ms": recording_duration_ms,
        "category": category,
        "source_start_ms": (start_sec * 1000.0).round() as i64,
        "source_end_ms": (end_sec * 1000.0).round() as i64,
        "duration_ms": duration_ms,
        "spoken_phrase": spoken_phrase,
        "words": words.iter().map(|word| {
            json!({
                "word": word.word.trim(),
                "start_ms": (word.start * 1000.0).round() as i64,
                "end_ms": (word.end * 1000.0).round() as i64,
            })
        }).collect::<Vec<_>>(),
    });
    let mut hasher = Sha256::new();
    hasher.update(input.to_string().as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Fixed-length background windows covering `[0, total)` seconds. A trailing
/// remnant shorter than the minimum is dropped rather than kept as a stub clip
/// the trainer would pad. Pure so the chunking is unit-testable without audio IO.
pub(crate) fn background_chunk_bounds(total: f64) -> Vec<(f64, f64)> {
    let mut bounds = Vec::new();
    let mut index = 0usize;
    loop {
        let start = index as f64 * BACKGROUND_CHUNK_SECONDS;
        if start >= total {
            break;
        }
        let end = (start + BACKGROUND_CHUNK_SECONDS).min(total);
        if end - start < BACKGROUND_MIN_CHUNK_SECONDS {
            break;
        }
        bounds.push((start, end));
        index += 1;
    }
    bounds
}

/// Short-frame RMS energy envelope of a mono PCM take. Frame `i` covers
/// `[i*frame_len, (i+1)*frame_len)` samples (the final frame is short); each
/// value is that frame's root-mean-square amplitude.
fn rms_envelope(samples: &[i16], frame_len: usize) -> Vec<f64> {
    let mut rms: Vec<f64> = Vec::new();
    let mut i = 0;
    while i < samples.len() {
        let end = (i + frame_len).min(samples.len());
        let mut sum = 0.0f64;
        for &s in &samples[i..end] {
            let v = s as f64;
            sum += v * v;
        }
        rms.push((sum / (end - i) as f64).sqrt());
        i += frame_len;
    }
    rms
}

/// The hysteresis open/close thresholds for burst detection, derived from the
/// envelope's noise floor (a low percentile) and peak (loudest frame). Returns
/// `None` for a flat take whose peak does not rise meaningfully above its floor,
/// so no bursts should be cut. Caller guarantees `rms` is non-empty.
fn energy_open_close(rms: &[f64]) -> Option<(f64, f64)> {
    let mut sorted = rms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let floor = sorted[(sorted.len() as f64 * 0.10) as usize];
    let peak = *sorted.last().unwrap();
    // A flat take (no burst rises meaningfully above its own floor) yields nothing.
    if peak - floor < 1.0 {
        return None;
    }
    let open = floor + ENERGY_OPEN_FRACTION * (peak - floor);
    let close = floor + ENERGY_CLOSE_FRACTION * (peak - floor);
    Some((open, close))
}

/// Hysteresis walk over the RMS envelope: open a voiced run when energy rises to
/// `open`, close it when energy falls below `close`. Returns each run as a
/// `[start_frame, end_frame)` half-open frame-index span.
fn hysteresis_runs(rms: &[f64], open: f64, close: f64) -> Vec<(usize, usize)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut voiced = false;
    let mut run_start = 0usize;
    for (idx, &energy) in rms.iter().enumerate() {
        if voiced {
            if energy < close {
                runs.push((run_start, idx));
                voiced = false;
            }
        } else if energy >= open {
            voiced = true;
            run_start = idx;
        }
    }
    if voiced {
        runs.push((run_start, rms.len()));
    }
    runs
}

/// Turn frame-index runs into final second spans: convert to seconds, merge runs
/// whose silent gap is short enough to be inside one utterance, drop anything too
/// short to be a real burst, and pad the survivors' edges (clamped to `total`).
fn merge_and_pad_bursts(
    runs: &[(usize, usize)],
    frame_secs: f64,
    total: f64,
) -> Vec<(f64, f64)> {
    let mut bursts: Vec<(f64, f64)> = Vec::new();
    for (start_frame, end_frame) in runs {
        let start = *start_frame as f64 * frame_secs;
        let end = (*end_frame as f64 * frame_secs).min(total);
        match bursts.last_mut() {
            Some(last) if start - last.1 < ENERGY_MERGE_GAP_SECONDS => last.1 = end,
            _ => bursts.push((start, end)),
        }
    }
    bursts
        .into_iter()
        .filter(|(start, end)| end - start >= ENERGY_MIN_BURST_SECONDS)
        .map(|(start, end)| {
            (
                (start - ENERGY_LEAD_PADDING_SECONDS).max(0.0),
                (end + ENERGY_TAIL_PADDING_SECONDS).min(total),
            )
        })
        .collect()
}

/// Segment a mono PCM take into sound-burst windows by short-frame RMS energy.
/// Pure (no audio IO) so the burst detection is unit-testable. Returns padded,
/// clamped `(start, end)` spans in seconds, in order. Used as the positive-take
/// fallback when Whisper finds no words: each repeated sound burst (e.g. one
/// fast "beep beep") becomes one positive clip.
pub(crate) fn energy_burst_bounds(samples: &[i16], sample_rate: f64) -> Vec<(f64, f64)> {
    if sample_rate <= 0.0 || samples.is_empty() {
        return Vec::new();
    }
    let frame_len = ((ENERGY_FRAME_SECONDS * sample_rate).round() as usize).max(1);
    let rms = rms_envelope(samples, frame_len);
    if rms.is_empty() {
        return Vec::new();
    }
    let Some((open, close)) = energy_open_close(&rms) else {
        return Vec::new();
    };
    let frame_secs = frame_len as f64 / sample_rate;
    let runs = hysteresis_runs(&rms, open, close);
    let total = samples.len() as f64 / sample_rate;
    merge_and_pad_bursts(&runs, frame_secs, total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_word(word: &str, start: f64, end: f64) -> WhisperWord {
        WhisperWord {
            word: word.to_string(),
            start,
            end,
            probability: 1.0,
        }
    }

    #[test]
    fn positive_context_first_fills_tail_aligned_window() {
        let words = vec![
            test_word("alpha", 0.00, 0.20),
            test_word("bravo", 0.35, 0.55),
            test_word("charlie", 0.90, 1.10),
            test_word("wake", 1.30, 1.55),
            test_word("word", 1.65, 1.90),
        ];

        assert_eq!(positive_context_first(&words, 3, 4), 2);
    }

    #[test]
    fn padded_bounds_expand_without_crossing_neighbors() {
        let words = vec![
            test_word("alpha", 0.00, 0.20),
            test_word("bravo", 0.35, 0.55),
            test_word("charlie", 0.90, 1.10),
            test_word("wake", 1.30, 1.55),
            test_word("word", 1.65, 1.90),
        ];

        // Final word: start padded but not into the previous word; tail padded.
        let (start, end) = padded_bounds(&words, 2, 4, 3.0, 0.08, 0.18, false);
        assert!((start - 0.82).abs() < 1e-9);
        assert!((end - 2.08).abs() < 1e-9);

        // With neighbor clamping on (negatives), an interior end clamps to the
        // next word's start instead of overrunning it.
        let (_, clamped_end) = padded_bounds(&words, 2, 3, 3.0, 0.08, 0.18, true);
        assert!((clamped_end - 1.65).abs() < 1e-9);

        // With neighbor clamping off (positives), the tail grows past the next
        // word toward the recording end so the wake phrase is not clipped.
        let (_, open_end) = padded_bounds(&words, 2, 3, 3.0, 0.08, 0.18, false);
        assert!((open_end - 1.73).abs() < 1e-9);

        // A large lead never pulls the start before the previous word's end.
        let (floor_start, _) = padded_bounds(&words, 1, 1, 3.0, 0.50, 0.10, true);
        assert!((floor_start - 0.20).abs() < 1e-9);
    }

    #[test]
    fn negative_ranges_use_word_chunks_with_hard_max_duration() {
        let words = vec![
            test_word("one", 0.0, 0.2),
            test_word("two", 0.5, 0.7),
            test_word("wake", 1.7, 1.9),
            test_word("word", 2.0, 2.2),
            test_word("three", 2.7, 2.9),
            test_word("four", 3.4, 3.6),
            test_word("five", 4.1, 4.3),
            test_word("overlong", 5.0, 6.7),
        ];

        let ranges = negative_ranges(&words, &[(1.6, 2.3)]);

        assert_eq!(
            ranges,
            vec![(0.0, 0.7, 0, 1), (2.7, 3.6, 4, 5), (4.1, 4.3, 6, 6)]
        );
    }

    #[test]
    fn clamp_slice_span_enforces_hard_max() {
        // Within the cap: bounds are untouched.
        let (start, end) = clamp_slice_span(0.5, 1.5, true);
        assert!((start - 0.5).abs() < 1e-9 && (end - 1.5).abs() < 1e-9);

        // Positive over the cap: keep the tail, trim the start in.
        let (start, end) = clamp_slice_span(0.0, 1.8, true);
        assert!((start - 0.3).abs() < 1e-9);
        assert!((end - 1.8).abs() < 1e-9);
        assert!(end - start <= MAX_SLICE_SECONDS + 1e-9);

        // Negative over the cap: keep the start, trim the end in.
        let (start, end) = clamp_slice_span(0.0, 1.8, false);
        assert!((start - 0.0).abs() < 1e-9);
        assert!((end - 1.5).abs() < 1e-9);
    }

    #[test]
    fn energy_burst_bounds_splits_repeated_bursts() {
        let sr = 16000.0;
        let mut samples: Vec<i16> = Vec::new();
        let silence = |secs: f64, v: &mut Vec<i16>| {
            for _ in 0..(secs * sr) as usize {
                v.push(0);
            }
        };
        let tone = |secs: f64, v: &mut Vec<i16>| {
            for i in 0..(secs * sr) as usize {
                v.push(if i % 2 == 0 { 8000 } else { -8000 });
            }
        };
        // Lead silence, then a "beep beep" (two 0.10s tones with a 0.10s internal
        // gap — shorter than the merge gap, so they fuse into one clip), a ~1s
        // gap between repetitions, then a second "beep beep".
        silence(0.30, &mut samples);
        tone(0.10, &mut samples);
        silence(0.10, &mut samples);
        tone(0.10, &mut samples);
        silence(1.00, &mut samples);
        tone(0.10, &mut samples);
        silence(0.10, &mut samples);
        tone(0.10, &mut samples);
        silence(0.30, &mut samples);

        let bursts = energy_burst_bounds(&samples, sr);
        assert_eq!(bursts.len(), 2, "expected two merged beep-beep bursts, got {bursts:?}");
        // First burst: the two beeps are merged into a single span near 0.30s.
        assert!(bursts[0].0 > 0.15 && bursts[0].0 < 0.31);
        assert!(bursts[0].1 - bursts[0].0 > 0.25);
        // Second burst lands well after the ~1s inter-repetition gap.
        assert!(bursts[1].0 > 1.4);
    }

    #[test]
    fn rms_envelope_frames_and_averages() {
        // frame_len 2 over 5 samples => 3 frames, last one short (a single sample).
        let env = rms_envelope(&[3, 4, 0, 0, 6], 2);
        assert_eq!(env.len(), 3);
        // sqrt((9+16)/2) = 3.5355..
        assert!((env[0] - (25.0f64 / 2.0).sqrt()).abs() < 1e-9);
        assert!((env[1] - 0.0).abs() < 1e-9);
        assert!((env[2] - 6.0).abs() < 1e-9); // single-sample tail frame
    }

    #[test]
    fn energy_open_close_flat_take_is_none() {
        // Peak barely above floor -> no thresholds.
        assert!(energy_open_close(&[10.0, 10.0, 10.5]).is_none());
        // A clear peak yields open>close, both between floor and peak.
        let (open, close) = energy_open_close(&[0.0, 0.0, 0.0, 100.0]).unwrap();
        assert!(open > close);
        assert!(open > 0.0 && open < 100.0);
        assert!(close > 0.0 && close < 100.0);
    }

    #[test]
    fn hysteresis_runs_open_high_close_low() {
        // Rise above open (10), stay voiced through the dip that is still >= close
        // (5), then drop below close to end the run; a final voiced tail closes at
        // the envelope end.
        let rms = [0.0, 12.0, 6.0, 0.0, 11.0];
        let runs = hysteresis_runs(&rms, 10.0, 5.0);
        assert_eq!(runs, vec![(1, 3), (4, 5)]);
    }

    #[test]
    fn merge_and_pad_bursts_merges_short_gaps_and_drops_short() {
        // Two runs 0.10s apart (< merge gap) fuse; a lone tiny run is dropped for
        // being below the minimum burst length.
        let frame_secs = 0.05;
        // run A: frames 6..8 => 0.30-0.40s; run B: 10..14 => 0.50-0.70s (gap 0.10s).
        let bursts = merge_and_pad_bursts(&[(6, 8), (10, 14)], frame_secs, 2.0);
        assert_eq!(bursts.len(), 1, "expected the two close runs to merge: {bursts:?}");
        // A single 1-frame run (0.05s) is below ENERGY_MIN_BURST_SECONDS -> dropped.
        assert!(merge_and_pad_bursts(&[(0, 1)], frame_secs, 2.0).is_empty());
    }

    #[test]
    fn energy_burst_bounds_ignores_flat_take() {
        // Pure silence and a low steady hiss both lack a burst rising above the
        // floor, so nothing is sliced.
        assert!(energy_burst_bounds(&vec![0i16; 32000], 16000.0).is_empty());
        assert!(energy_burst_bounds(&vec![25i16; 32000], 16000.0).is_empty());
    }

    #[test]
    fn visible_range_drops_trimmed_words() {
        let words = vec![
            test_word("alpha", 0.00, 0.20),
            test_word("bravo", 0.35, 0.55),
            test_word("wake", 1.30, 1.55),
            test_word("word", 1.65, 1.90),
        ];

        // A clamped start past the first words drops them from the transcript.
        assert_eq!(visible_range(&words, 0, 3, 0.60, 2.00), (2, 3));
        // A clamped end drops trailing words instead.
        assert_eq!(visible_range(&words, 0, 3, 0.00, 1.20), (0, 1));
        // No trimming needed: the full range stays.
        assert_eq!(visible_range(&words, 0, 3, 0.00, 2.00), (0, 3));
    }

    #[test]
    fn bulk_clip_hash_id_is_stable_across_reprocess() {
        let words = vec![test_word("all", 1.30, 1.55), test_word("set", 1.65, 1.90)];
        // Same recording identity + cut + words => same id, so re-syncing or
        // reprocessing the stored source produces identical slice ids.
        let a = bulk_clip_hash_id("rec-1", "2026-01-01", 5000, "positive", 1.2, 1.9, &words);
        let b = bulk_clip_hash_id("rec-1", "2026-01-01", 5000, "positive", 1.2, 1.9, &words);
        assert_eq!(a, b);
        // A different recording or cut yields a different id.
        let c = bulk_clip_hash_id("rec-2", "2026-01-01", 5000, "positive", 1.2, 1.9, &words);
        assert_ne!(a, c);
        let d = bulk_clip_hash_id("rec-1", "2026-01-01", 5000, "negative", 1.2, 1.9, &words);
        assert_ne!(a, d);
    }

    #[test]
    fn hard_negative_context_flags_near_miss_frames() {
        // A wake phrase spoken after an explicit near-miss cue must be treated as
        // a hard negative, not a positive.
        let words = vec![
            test_word("this", 0.0, 0.2),
            test_word("is", 0.25, 0.4),
            test_word("not", 0.45, 0.6),
            test_word("the", 0.65, 0.75),
            test_word("wake", 0.8, 1.0),
            test_word("phrase", 1.05, 1.3),
            test_word("all", 1.5, 1.7),
            test_word("set", 1.75, 2.0),
        ];
        // "all set" at indices 6..=7 sits inside a "not the wake phrase" frame.
        assert!(is_hard_negative_context(&words, 6, 7));

        // The same phrase with an ordinary lead-in is a clean positive.
        let clean = vec![
            test_word("please", 0.0, 0.3),
            test_word("say", 0.35, 0.6),
            test_word("all", 0.7, 0.9),
            test_word("set", 0.95, 1.2),
        ];
        assert!(!is_hard_negative_context(&clean, 2, 3));
    }

    #[test]
    fn background_chunks_cover_source_and_drop_short_tail() {
        // Too short to yield any usable window.
        assert!(background_chunk_bounds(0.5).is_empty());
        // Exact single window.
        assert_eq!(background_chunk_bounds(2.0), vec![(0.0, 2.0)]);
        // A 1.0s tail meets the minimum and is kept.
        assert_eq!(
            background_chunk_bounds(5.0),
            vec![(0.0, 2.0), (2.0, 4.0), (4.0, 5.0)]
        );
        // A 0.5s remnant is below the minimum and dropped.
        assert_eq!(background_chunk_bounds(4.5), vec![(0.0, 2.0), (2.0, 4.0)]);
    }
}
