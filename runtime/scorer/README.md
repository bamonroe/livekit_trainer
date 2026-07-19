# Wake-Word Scoring Service

A small, stateless HTTP service that turns an uploaded 16 kHz mono WAV into a
rolling **detection-score curve** for a trained wake-word `.onnx`, at a chosen
time resolution. It mirrors the Whisper-service pattern the sync-server already
uses (an external ML service called over HTTP), and is the engine behind the
planned in-app model-test view.

## Why two scan modes

The model's decision window is **2.0 s**; internally it emits one speech
embedding every 8 mel frames (~80 ms) and the classifier reads the last 16.
Those are fixed by the trained model. How often we *ask* for a score is free.

- **`full`** — score exactly as the model runs on a live stream: a 2 s window
  slides over the audio. Mel + embeddings run **once** over the whole clip (a
  dense embedding bank); only the cheap classifier head repeats per step, so
  10 ms resolution costs a fraction of a second and reproduces the native 80 ms
  grid exactly at the shared points.
- **`reset`** — at each step keep only the last `keep_ms` of real audio and pad
  silence in front, mimicking the isolated, silence-padded clips the trainer
  builds. This recovers detection of a phrase spoken mid-utterance (the current
  `all_set` model barely fires in continuous speech but fires ~0.99 under
  reset). One model pass per step, batched.

Both modes were validated to match the package's own `WakeWordModel.predict()`
to within float noise (max diff ~3e-8).

## API

```
GET  /health   -> {"status","models_dir","loaded":[...],"available":[...]}
POST /score    multipart form:
    file     WAV, 16 kHz mono 16-bit PCM
    slug     model resolved at <MODELS_DIR>/<slug>/<slug>.onnx
    mode     "full" | "reset"      (default full)
    step_ms  scan resolution       (default 10)
    keep_ms  reset: real audio kept (default 700)
  -> {"model","mode","duration_ms","window_ms":2000,"step_ms","keep_ms",
      "times_ms":[...],"scores":[...]}
```

`times_ms[i]` is the detection time; overlay Whisper word spans to compute
true/false positives and false negatives as the threshold and padding sliders
move.

## Fronted by the sync-server

The Rust sync-server calls this service so callers never talk to it directly for
stored takes. `GET /score/:slug/:recording_id` on the sync-server replays a
stored bulk recording through here and overlays the recording's current Whisper
transcript:

- It resolves the scorer from `SCORER_SERVER_URL` (or the `x-scorer-server-url`
  request header), reads the take's `bulk_source` WAV, and POSTs it here.
- It locates every trigger-phrase utterance in the transcript and tags each with
  the model's peak score in the window aligned to the phrase tail.
- Response adds `targets[]` (per-utterance peak + `detected`) and
  `true_positives`/`false_negatives`/`false_positives` at `threshold`, alongside
  the full `times_ms`/`scores` curve so a client can re-threshold for free.

Query params mirror this service: `mode` (full|reset), `step_ms`, `keep_ms`,
`threshold`. This is the honest streaming diagnostic — `full` shows the real
mid-sentence recall, `reset` shows the training-time (silence-padded) view.

## Run

```bash
docker compose up -d --build scorer      # serves on :8770, reads ./output
curl -s localhost:8770/health
curl -s -F file=@take.wav -F slug=all_set -F mode=reset localhost:8770/score
```

Notes: `full` mode needs ≥2 s of audio to produce any points (it can't form a
continuous 2 s window from a shorter clip); feed whole takes, not single slices.
