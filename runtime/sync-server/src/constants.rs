//! Tuning constants for slicing, padding, background chunking, and the
//! energy/VAD fallback.
//!
//! These are the knobs that shape how a long take is cut into training clips.
//! They live together so the trade-offs (why a positive keeps its tail, why a
//! background remnant is dropped) are documented in one place and shared by the
//! slicing, alignment, and scoring modules without duplication.

/// Longest span of speech (in seconds) pulled in as lead-in context before a
/// tail-aligned positive's wake phrase.
pub(crate) const POSITIVE_MAX_SECONDS: f64 = 1.5;
// Hard ceiling on the final padded slice length. The context/target budgets
// above bound only the word span; lead/tail padding is added on top, so without
// this every positive ran ~0.3s over. Positives keep their tail (the wake
// phrase ends the clip) so the start is trimmed in; negatives keep their start.
/// Hard ceiling on any final padded slice length, in seconds.
pub(crate) const MAX_SLICE_SECONDS: f64 = 1.5;
// Whisper word timestamps drift from the true audio, so cutting exactly at
// word.start/word.end clips onsets and (worst of all) chops the tail-aligned
// wake phrase. Nudge each cut outward, bounded by the neighboring words, to keep
// slices honest to their transcript. The positive tail is padded hardest because
// a positive that lost its wake phrase is the most damaging error.
/// Seconds of lead padding added before a slice's first word.
pub(crate) const CUT_LEAD_PADDING_SECONDS: f64 = 0.08;
/// Seconds of tail padding after a positive's last word (padded hardest so the
/// wake phrase is never clipped).
pub(crate) const POSITIVE_TAIL_PADDING_SECONDS: f64 = 0.28;
/// Seconds of tail padding after a negative's last word.
pub(crate) const NEGATIVE_TAIL_PADDING_SECONDS: f64 = 0.10;
/// Target length (seconds) a negative word-chunk grows toward.
pub(crate) const NEGATIVE_TARGET_SECONDS: f64 = 1.5;
// Ambient background recordings carry no speech to align, so they are chopped
// into fixed windows sized to the trainer's clip_duration (2.0s). Each window
// becomes an independent background training example; a trailing remnant shorter
// than the minimum is dropped rather than padded into a misleadingly short clip.
/// Fixed background chunk length, matching the trainer's clip_duration.
pub(crate) const BACKGROUND_CHUNK_SECONDS: f64 = 2.0;
/// Shortest background remnant kept; anything shorter is dropped.
pub(crate) const BACKGROUND_MIN_CHUNK_SECONDS: f64 = 1.0;
// Sentinel stored in a recording's `script` column to mark it as a background
// noise take rather than a scripted bulk read. Reprocess branches on this so
// background sources are re-chunked deterministically instead of Whisper-aligned.
/// Sentinel `script` value marking a take as background noise.
pub(crate) const BACKGROUND_SCRIPT_MARKER: &str = "__background_noise__";
// Sentinel stored in a positive take's `script` column when its wake word is a
// non-lexical sound. The app stamps it (from the project's energy-positives
// toggle) so this take is *always* energy-sliced, regardless of what Whisper
// transcribes — this is stronger than the empty-transcript auto-fallback, which
// misses a take where Whisper happens to catch one of many bursts. Must match
// `BulkRecording.ENERGY_POSITIVE_MARKER` in the Android app.
/// Sentinel `script` value forcing energy slicing of a positive take.
pub(crate) const ENERGY_POSITIVE_SCRIPT_MARKER: &str = "__energy_positive__";

// Energy/VAD fallback for non-lexical positive takes (sounds, not words — e.g. a
// fast "beep beep") where Whisper returns no words, so word-timestamp slicing
// finds nothing. Positives are recorded as repeated sound bursts with ~1s gaps,
// so we segment the take by sound-burst-vs-silence energy and cut each burst.
// Frames are short RMS windows; a burst opens above `OPEN` and closes below
// `CLOSE` (hysteresis) of the way from the noise floor to the loudest frame.
/// RMS frame length (seconds) for the energy envelope.
pub(crate) const ENERGY_FRAME_SECONDS: f64 = 0.02;
/// Hysteresis open threshold, as a fraction from floor to peak.
pub(crate) const ENERGY_OPEN_FRACTION: f64 = 0.22;
/// Hysteresis close threshold, as a fraction from floor to peak.
pub(crate) const ENERGY_CLOSE_FRACTION: f64 = 0.12;
// Bursts separated by a gap this short are merged into one clip, so the two
// quick sounds inside one "beep beep" stay together while the ~1s gap between
// repetitions still splits them into separate positives.
/// Silent gap (seconds) below which two bursts merge into one clip.
pub(crate) const ENERGY_MERGE_GAP_SECONDS: f64 = 0.35;
/// A voiced run shorter than this is treated as noise, not a real sound burst.
pub(crate) const ENERGY_MIN_BURST_SECONDS: f64 = 0.08;
/// Lead padding (seconds) cut around each detected burst.
pub(crate) const ENERGY_LEAD_PADDING_SECONDS: f64 = 0.10;
/// Tail padding (seconds) cut around each detected burst.
pub(crate) const ENERGY_TAIL_PADDING_SECONDS: f64 = 0.18;

/// Model firing can land up to ~1s from Whisper's reported word time.
pub(crate) const MAX_DRIFT_MS: f64 = 1200.0;
