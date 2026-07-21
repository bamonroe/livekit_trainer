# AGENTS.md

Guidance for agents working in this repository.

## Project Goal

Build an independent wake-word training and collection project, separate from
`/data/claude_spawner_app`.

The target product is an Android app for collecting voice samples from one user
and managing any number of arbitrary wake words. The app should help record,
label, review, export, and eventually train wake-word models from those samples.

This repo should become the home for:

- The Android sample-collection app.
- The LiveKit wake-word training container and configs.
- Training data import/export tooling.
- Model output and evaluation helpers.
- Optional runtime test services for trained `.onnx` classifiers.
- `TODO.md`, the living project backlog.

This repository is the source of truth for wake-word model artifacts. Other
projects may consume the trained `.onnx` files, but training inputs, configs,
evaluation outputs, and final model files should be organized here first.

The intended result is a complete loop:

1. Define a wake word or phrase in the Android app.
2. Generate a recording plan with prompts for positives, negatives, hard
   negatives, background noise, silence, and later false-positive /
   false-negative correction batches.
3. Rapidly collect real human voice samples on-device.
4. Export the recordings and metadata into this repo.
5. Convert the export into the LiveKit training layout.
6. Generate or update the LiveKit YAML config.
7. Train, evaluate, and export an `.onnx` wake-word model.
8. Use evaluation results and runtime mistakes to create the next collection
   batch.

## Source Context

Useful existing work lives in sibling repos:

- `/data/claude_spawner_app/wakeword`
  - `Dockerfile.trainer`
  - `configs/beep.yaml`
  - `configs/bump.yaml`
  - `configs/bump_cal.yaml`
  - `patches/0001-parallel-augmentation.patch`
  - `service/`, a Rust HTTP sidecar for scoring clips with trained classifiers.
- `/data/claude_spawner_app/docker-compose.yml`
  - Shows how training data and model output are bind-mounted.
  - Uses `/data/storage/livekit-wakeword/real` for real recorded clips.
  - Uses `/data/storage/livekit-wakeword/output` for trained model artifacts.
- `/data/livekit-wakeword`
  - Local clone of the upstream LiveKit wake-word project.
  - Contains broader docs, configs, Swift package, tests, and current source.
- `/data/android`
  - Shared containerized Android build and emulator environment.
  - Read `/data/android/README.md` and `/data/android/BUILD-ENV-PLAN.md` before
    building or testing the Android app.
  - Build app projects with `/data/android/build.sh <project-dir> [gradle-task]`.
  - When Android changes pass the build and any emulator checks requested for
    the change, install the resulting debug APK onto the connected phone and
    tablet when both are available.

Do not make this repo depend on the claude spawner application. Copy or adapt
only the generic wake-word pieces that belong here.

## Wake-Word Training Notes

The inherited trainer is based on `livekit-wakeword[train,eval,export]`.

The existing trainer container:

- Uses `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime`.
- Installs `espeak-ng`, `sox`, `ffmpeg`, `libsndfile1`, `portaudio19-dev`, and
  `git`.
- Applies `patches/0001-parallel-augmentation.patch` to parallelize CPU-bound
  augmentation.
- Runs from `/work`.

The known training command pattern is:

```bash
docker build -f Dockerfile.trainer -t livekit-wakeword-trainer:latest .

docker run --rm --gpus all \
  -v /data/livekit-wakeword:/work \
  -v /data/storage/livekit-wakeword/data:/work/data \
  -v /data/storage/livekit-wakeword/output:/work/output \
  livekit-wakeword-trainer:latest \
  bash -lc "livekit-wakeword setup -c configs/example.yaml && livekit-wakeword run -c configs/example.yaml"
```

For this repo, prefer replacing `/data/livekit-wakeword:/work` with this
workspace once the trainer files are copied here.

Important config fields:

- `model_name`: stable slug for the wake word.
- `target_phrases`: phrases that should trigger.
- `custom_negative_phrases`: near misses and phrases that must not trigger.
- `data_dir`: usually `./data`.
- `output_dir`: usually `./output`.
- `model.model_type`: prefer `conv_attention`.
- `model.model_size`: start with `medium`.
- `n_samples`, `n_samples_val`, `steps`, and `target_fp_per_hour` control
  training cost and quality.

## Data Layout

Use a layout that supports many wake words and repeated recording sessions.

Recommended local layout:

```text
android/                 Android app source
trainer/                 Dockerfile, patches, config templates, scripts
runtime/                 Optional scorer service or model test harness
data/
  real/
    <wake_word_slug>/
      positive/
      negative/
      background/
  generated/             Synthetic data and downloaded corpora if local
output/
  <wake_word_slug>/
    <wake_word_slug>.onnx
    <wake_word_slug>.pt
    <wake_word_slug>_metrics.json
    <wake_word_slug>_eval.json
```

The Android app should export audio with enough metadata to reconstruct this
layout without guesswork.

Preferred clip format:

- WAV.
- Mono.
- 16 kHz when possible.
- 16-bit PCM.
- Short clips, with the wake phrase near the end when training expects a
  tail-aligned wake word.

Metadata should include:

- Wake-word slug.
- Spoken phrase.
- Label: `positive`, `negative`, or `background`.
- Timestamp.
- Device model.
- Microphone or input route if available.
- Sample rate and channel count before conversion.
- Session id.

## Collection Labels

The app and import tools should distinguish these recording purposes:

- `positive`: the target phrase, spoken naturally.
- `negative`: ordinary speech that should not trigger the model.
- `hard_negative`: near-miss phrases that sound similar to the target.
- `background`: room noise, silence, keyboard noise, appliance noise, and other
  non-speech audio.
- `false_positive`: audio that incorrectly triggered a trained model and should
  be added as future negative training data.
- `false_negative`: audio where the user spoke the target phrase but the model
  missed it and should be added as future positive training data.

When exporting to LiveKit's current directory shape, map these into the training
categories the trainer expects. Preserve the richer label in metadata so later
tools can rebalance or inspect the dataset.

## Android App Direction

Build the Android app as a real collection tool, not a demo.

The app is **bulk-import only** for speech. Data is collected by recording long
takes and letting the sync server slice them into clips; there is no
one-at-a-time short-prompt recording flow, no per-clip label picker, and no
manual Export/Sync of individual clips. That orphaned path was removed. Do not
reintroduce it without an explicit decision to change direction.

The Record page is now **four straight recorders, one per take kind** — it is no
longer a single mixed randomized script read in one take. Each recorder captures
a plain record-and-stop take of exactly one kind:

- **Positives** — say the wake phrase over and over with a short gap between each
  repetition. The server slices every clean repetition into its own positive
  clip.
- **Negatives** — say ordinary, unrelated sentences. The server chops the take
  into negative clips.
- **Hard negatives** — the only prompted recorder: it lists near-miss phrases to
  read aloud a few times each, and files the whole take as hard negatives. The
  prompt list is guidance for the reader; the server slices by take kind, not by
  matching those words.
- **Background noise** — a long ambient/non-speech take (room tone, near-silence,
  appliances, typing). It is *not* transcribed; the server chops it into
  fixed-length background clips.

Speech takes (positive/negative/hard_negative) are transcribed and sliced with
Whisper word timestamps. Background takes are chopped by fixed length. Keep both
long-take modes; do not fold them back into a single scripted read.

**Non-lexical wake words (sounds, not words).** Some wake words are fast or
non-lexical — e.g. "beep beep" said quickly — and Whisper returns no words for
them, so word-timestamp slicing produces nothing. The plan for these is a
**per-take energy/VAD fallback for positive takes only**: when Whisper finds no
usable words in a positive take, fall back to segmenting the take by
sound-burst-versus-gap energy and slicing each burst into a positive clip. This
works because positives are already recorded as repeated bursts with gaps.
Whisper stays the default everywhere it works — real-word positives, and all
negatives — and the energy path is only a fallback, never a replacement.

Core workflows:

- Create and edit wake-word projects.
- Record a straight per-kind take (positive, negative, hard negative, or
  background) in one take.
- Sync recordings to the server, which slices each take by its kind.
- Review the generated slices per recording: see each slice's transcript,
  confidence, and source timing; replay it; and delete bad slices.
- Inspect the source alignment (word timings and cut boundaries) for a recording.
- Show per-wake-word recording and slice counts.
- Support correction batches created from model evaluation mistakes.

Design expectations:

- Keep the record control large and reliable.
- Keep bulk recording, slice review, and settings on clearly separated pages.
- Avoid hidden state in the recording flow.
- Store raw bulk recordings locally before sync.
- Prefer deterministic filenames with timestamp and sanitized phrase.

Technical expectations:

- Use Android-native audio APIs.
- Request microphone permission clearly.
- Keep sample conversion code tested.
- Treat user voice recordings as private data.
- Upload bulk recordings only to the user's own configured sync server; do not
  add any other upload destination without an explicit request.

## Containerized Development

All project development should be reproducible inside Docker.

Expect to add a development container that can be used for Android tooling,
Python training helpers, audio tools, and model pipeline scripts. Agents may
install software as root inside the container rather than mutating the host.

Preferred shape:

- `docker-compose.yml` for local development services.
- `Dockerfile.dev` for the main development environment.
- Bind-mount this repo into the container.
- Keep Gradle caches, Android SDK caches, Python caches, and training corpora in
  named volumes or ignored host directories.
- Keep GPU-enabled model training in a dedicated trainer image so the Android
  development container stays lighter.

Do not assume host packages are available. If a command needs system
dependencies, add them to the relevant Dockerfile.

Use the project dev container with:

```bash
docker compose run --rm dev
```

The dev container bind-mounts this repo at `/workspace`, mounts `/data`, and
mounts the host Docker socket so it can call shared build tooling such as
`/data/android/build.sh`.

## Runtime Notes

The inherited Rust sidecar accepts raw little-endian i16 mono PCM and returns a
score map.

Existing HTTP contract:

- `GET /health` returns status and loaded model names.
- `POST /detect` accepts raw PCM and returns scores.
- `GET /stream` supports streaming PCM over WebSocket in the newer service code.

This is useful for local validation of trained models, but it should remain
optional for the Android collection app.

## Development Rules

- Keep this project independent from claude spawner.
- Keep `TODO.md` current whenever adding, completing, deferring, or discovering
  work. The file is the shared backlog for both Codex and Claude.
- Keep `CLAUDE.md` symlinked to `AGENTS.md` so Codex and Claude read the same
  repository instructions.
- Keep Android bring-up notes in `docs/ANDROID_APP_MAP.md` current as the app
  architecture and build workflow evolve.
- Commit freely, often, and regularly. Do not wait to be asked. Make small,
  clear commits as you complete each meaningful piece of work, so progress is
  always captured and easy to review or revert.
- Push to configured remotes whenever you need to, and by default whenever you
  have committed work, so it is backed up and available to other agents.
- Use the local `gh` CLI when needed to create or inspect the GitHub repository
  and remote configuration.
- Prefer working inside the project Docker development container.
- Add or update Dockerfiles when new system dependencies are needed.
- Prefer copying generic trainer/runtime pieces into this repo before modifying
  them.
- After changing the sync-server (or any server) code, rebuild its Docker image
  so the built image matches the source (`docker compose build sync-server`).
  Rebuild the image, but do not restart or recreate the running container as
  part of this rule; leave the live container running until a restart is
  actually wanted. Otherwise the container keeps running stale code that matches
  the last built image even though the source has moved on.
- Do not edit sibling repos unless the user explicitly asks.
- Do not commit large generated corpora or model outputs unless the repo policy
  is changed.
- Keep real voice recordings out of git by default.
- Add `.gitignore` rules before creating generated data or model artifacts.
- Use scripts for repeatable training and export steps.
- When running long builds, training jobs, servers, or watches, start them with:

```bash
/home/bam/.spawner-jobs/spawner-job start '<cmd>'
```

Check those jobs with:

```bash
/home/bam/.spawner-jobs/spawner-job list
```

## First Milestones

1. Copy the generic trainer files from `/data/claude_spawner_app/wakeword` into
   this repo under `trainer/`.
2. Add `.gitignore` for Android build output, training data, generated corpora,
   model output, and temporary audio files.
3. Scaffold the Android app under `android/`.
4. Define the training bundle format exported by the Android app.
5. Add a script that converts exported Android sessions into
   `data/real/<wake_word_slug>/{positive,negative,background}`.
6. Add a script that generates a LiveKit config for a wake word from project
   metadata.
7. Add a smoke-test training path using a tiny calibration config before running
   full 50,000-step training.
