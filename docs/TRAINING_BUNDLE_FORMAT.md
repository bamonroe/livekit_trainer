# Training Bundle Format

Android exports should use this format so repo tools can import recordings into
the LiveKit training layout without guessing.

## Directory Shape

```text
<bundle>/
  manifest.json
  audio/
    <clip_id>.wav
  bulk_audio/
    <bulk_recording_id>.wav
```

Bundles may be zipped for transfer, but the unzipped shape should match this
layout.

## Audio Requirements

Preferred WAV format:

- Mono.
- 16 kHz.
- 16-bit PCM.
- Short clips.
- For `positive` and `false_negative` clips, put the target phrase near the end
  of the clip when possible.

If the app records a different source format, it must write the source metadata
in the manifest and either export normalized WAV or mark the clip as needing
conversion.

## Labels

Allowed labels:

- `positive`
- `negative`
- `hard_negative`
- `background`
- `false_positive`
- `false_negative`

Importer category mapping:

| Label | LiveKit category |
| --- | --- |
| `positive` | `positive` |
| `false_negative` | `positive` |
| `negative` | `negative` |
| `hard_negative` | `negative` |
| `false_positive` | `negative` |
| `background` | `background` |

The importer should preserve the original label in metadata even when mapping it
into a simpler LiveKit category.

## Manifest Schema

Initial schema version: `1`.

```json
{
  "schema_version": 1,
  "exported_at": "2026-07-15T21:45:00Z",
  "app": {
    "package": "com.bam.livekittrainer",
    "version_name": "0.1.0",
    "version_code": 1
  },
  "device": {
    "manufacturer": "Google",
    "model": "Pixel 8",
    "android_sdk": 35
  },
  "wake_word": {
    "id": "uuid-or-local-id",
    "slug": "hey_buddy",
    "phrase": "hey buddy",
    "target_phrases": ["hey buddy"],
    "negative_phrases": ["hey body", "hey google"]
  },
  "prompt_batch": {
    "id": "uuid-or-local-id",
    "created_at": "2026-07-15T21:30:00Z",
    "strategy": "mixed_v1"
  },
  "clips": [
    {
      "id": "clip_20260715_214501_001",
      "file": "audio/clip_20260715_214501_001.wav",
      "label": "positive",
      "prompt": "Say hey buddy",
      "spoken_phrase": "hey buddy",
      "recorded_at": "2026-07-15T21:45:01Z",
      "duration_ms": 1800,
      "sample_rate_hz": 16000,
      "channels": 1,
      "encoding": "pcm_s16le",
      "session_id": "session_20260715_2140",
      "notes": ""
    }
  ],
  "bulk_recordings": [
    {
      "id": "bulk_20260716_134501_001",
      "file": "bulk_audio/bulk_20260716_134501_001.wav",
      "script": "Exact text shown to the speaker, including wake phrases.",
      "recorded_at": "2026-07-16T13:45:01Z",
      "duration_ms": 45000,
      "sample_rate_hz": 16000,
      "channels": 1,
      "encoding": "pcm_s16le",
      "session_id": "bulk",
      "notes": ""
    }
  ]
}
```

## Validation Rules

Importer tooling should reject a bundle when:

- `manifest.json` is missing.
- `schema_version` is unsupported.
- `wake_word.slug` is missing or unsafe for paths.
- A clip label is unknown.
- A referenced audio file is missing.
- A referenced audio file escapes the bundle directory.

Importer tooling should warn, but not necessarily reject, when:

- Audio is not 16 kHz mono PCM WAV.
- A clip has zero duration metadata.
- Positive clips are unusually long.
- Required device metadata is missing.

## Import Target

The importer should copy or normalize clips into:

```text
data/real/<wake_word_slug>/
  positive/
  negative/
  background/
```

Future versions may also write a sidecar metadata index under:

```text
data/real/<wake_word_slug>/metadata.jsonl
```
