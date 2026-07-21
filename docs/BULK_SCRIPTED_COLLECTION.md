# Bulk Scripted Collection

> **Current state (superseded flow below).** The app no longer records one long
> *mixed* scripted reading that interleaves positives, negatives, and hard
> negatives in a single take. The Record page is now **four straight recorders,
> one per take kind**: positives (say the wake phrase repeatedly with gaps),
> negatives (ordinary sentences), hard negatives (prompted near-miss phrases),
> and background noise. The server still transcribes speech takes with Whisper
> and slices by kind; background is chopped by fixed length. The "Idea" and
> "First Implementation" sections below describe the original single-script
> design and are kept for history. The still-live parts are: Whisper word-time
> slicing, padding/validation, provenance, and slice review.

## Non-Lexical Wake Words (planned)

Some wake words are sounds, not words — e.g. a fast "beep beep" — and Whisper
returns no words for them, so word-timestamp slicing yields nothing.

Plan: add a **per-take energy/VAD fallback for positive takes only**. When
Whisper finds no usable words in a positive take, segment the take by
sound-burst-versus-silence energy and slice each burst into a positive clip.
Positives are already recorded as repeated bursts with short gaps, so the burst
structure is exactly what the energy detector keys on. Whisper remains the
default for every take where it works (real-word positives and all negatives);
the energy path is a fallback, never a replacement. A project should carry a
flag (or auto-detection on empty transcript) that selects the energy path.

## Idea

Generate several paragraphs of readable text with the wake word inserted many
times in varied short contexts. The user records the whole script once. The repo
then sends the audio, plus the known script text, to a Whisper-style
transcription or alignment service that returns word timestamps.

From those timestamps, tooling can create:

- Positive clips: short windows around each wake-word occurrence.
- Negative clips: nearby speech windows that do not contain the wake word.
- Background clips: optional pauses or room-tone sections.

This should complement, not replace, the existing short prompt workflow.

## Requirements

- Keep wake-word occurrences separated enough that each can be sliced cleanly.
- Keep generated text natural enough to read without stumbling.
- Store the original long recording, script, alignment output, and sliced clips.
- Include provenance in metadata so every sliced clip points back to the source
  recording and word timestamp range.
- Add padding around each slice so the phrase is not cut off.
- Validate each slice for WAV format, duration, clipping, and silence.
- Treat Whisper timestamps as proposed alignment, not perfect ground truth.

## First Implementation

Implemented first pass:

1. The Android app generates a long read-aloud bulk script.
2. Bulk mode records one long WAV and stores it separately from short clips.
3. Bundle export includes `bulk_recordings` with the exact script text.
4. The Rust sync server sends bulk WAVs to the configured Whisper server using
   `response_format=verbose_json` and `word_timestamps=true`.
5. The sync server slices positives around aligned wake-word occurrences.
6. The sync server slices a small set of nearby negative speech windows.
7. Slices are written into the normal `data/real/<wake_word_slug>/` layout with
   provenance metadata.
8. The Android app can load generated slices from the sync server, play each
   slice, inspect the Whisper transcript and timing, and delete bad slices.

Next tuning step:

1. Record real phone batches.
2. Inspect the generated slices.
3. Tune negative sampling and review reporting.

Current tuning:

- Bulk scripts use generated natural sentence templates filled with dictionary
  content words, instead of a small repeated fixed sentence bank.
- Bulk scripts include phonetic near-miss phrases as hard-negative speech inside
  the long reading.
- The Android script view highlights true wake phrases in bold green and
  near-miss hard negatives in bold red to reduce read-aloud mistakes.
- The number of wake-phrase occurrences per bulk script is configurable in
  Android settings. The default is 8 so a read can be completed in a shorter
  single take.
- Each script starts with neutral lead-in sentences before the first wake phrase.
- Positive slicing uses extra trailing context when the wake phrase occurs too
  early to provide full pre-roll padding.
- Negative slicing prefers sentence-ending punctuation and can use longer
  windows so ordinary negative examples are less likely to cut off the final
  word.
- Positive review slicing currently uses Whisper's matched wake-word timestamps
  with only tiny edge padding, skips obvious hard-negative contexts such as
  near-match or not-the-wake-phrase sentences, and clears older generated slices
  for a source recording before reprocessing it.
- The sync server keeps processing later bulk recordings if one recording fails
  alignment, and reports that failure as a warning.
- The sync server corrects Whisper word timestamps when a Whisper backend
  reports words relative to segment audio while segment timestamps include
  leading silence.
- Before upload, Android asks the sync server which bulk recording IDs already
  have generated slices and omits those long WAVs from the next sync bundle.
- Review is currently slice-level. Word timestamps are shown as transcript,
  timing, duration, and confidence summaries, but word-by-word editing is not
  implemented.
- Alignment replay stores the original bulk WAV and Whisper word timestamps so
  the Android app can play the source recording, highlight the current word,
  and show generated cut markers in the word stream.

## Open Questions

- How much padding should be used before and after a wake-word occurrence.
- Whether the Android app should slice locally later, or always export the long
  recording for repo-side slicing first.
