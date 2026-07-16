# TODO.md

Living backlog for `/data/livekit_trainer`.

Keep this file current whenever work is added, completed, deferred, or
discovered. Prefer small, actionable items with clear status.

## Active

No active items. Pick the next item from Later when starting new work.

## Later

- [ ] Add optional runtime scorer service or test harness under `runtime/`.
- [ ] Add a post-sync cleanup policy for app-private clips after server import
  is acknowledged.
- [ ] Add emulator or instrumentation coverage for SQLite metadata migration
  and app-private WAV file retention.
- [ ] Add evaluation tooling that creates follow-up false-positive and
  false-negative collection batches.
- [ ] Add dataset summary reports by wake word, label, phrase, session, and
  device.
- [ ] Add audio validation for sample rate, channel count, duration, clipping,
  and silence.

## Done

- [x] Define the initial Android training bundle export format in
  `docs/TRAINING_BUNDLE_FORMAT.md`.
- [x] Build the first app screen for creating wake-word projects.
- [x] Add prompt batch data model and prompt generation rules.
- [x] Add basic microphone recording flow for prompted clips.
- [x] Persist clip metadata for recorded prompt clips.
- [x] Add local clip listing, replay, and delete flow.
- [x] Add prompt-by-prompt advancement after each saved recording.
- [x] Add a cleaner Android collection UI with project sidebar navigation and
  direct prompt picking.
- [x] Move Android project navigation behind a hamburger drawer and add a
  settings gear for sync URL and theme controls.
- [x] Add Android bundle export from recorded clip metadata.
- [x] Add Dockerized Rust app-to-repo server sync for Android training bundles.
- [x] Investigate local Docker builder hangs and verify `docker compose build
  sync-server` completes with the smaller Debian runtime image.
- [x] Decide Android persistence and storage strategy: SQLite metadata with
  app-private WAV files written at capture time.
- [x] Add audio validation for duration, silence, clipping, and WAV format.
- [x] Add sample bundle tests for `scripts/import_bundle.py`.
- [x] Add import tooling to convert exported sessions into
  `data/real/<wake_word_slug>/{positive,negative,background}`.
- [x] Add config-generation tooling for LiveKit wake-word YAML files.
- [x] Define the model artifact source-of-truth policy and export conventions
  for downstream projects.
- [x] Add a smoke-test training path using a tiny calibration config.
- [x] Verify trainer smoke job `1784196277_3287142_4aad` completes the full
  generate, augment, feature, train, export, and eval pipeline.
- [x] Fix trainer Dockerfile package-path lookup for namespace-package
  `livekit`.
- [x] Add `Dockerfile.dev` for containerized Android and pipeline development.
- [x] Add `docker-compose.yml` for the development container and persistent
  caches.
- [x] Re-examine `/data/android` and update the Android map with the latest
  builder facts.
- [x] Scaffold the Android app under `android/` with a Gradle wrapper compatible
  with `/data/android/build.sh`.
- [x] Verify the Android scaffold builds with
  `/data/android/build.sh /data/livekit_trainer/android :app:assembleDebug`.
- [x] Add repository policy to commit often and push remotes frequently.
- [x] Initialize `/data/livekit_trainer` as a Git repository on branch `main`.
- [x] Add `docs/ANDROID_APP_MAP.md` referencing the `/data/android` build and
  emulator directions.
- [x] Copy training-only files from `/data/claude_spawner_app/wakeword` into
  `trainer/`.
- [x] Leave the old runtime sidecar out of the trainer copy.
- [x] Add `.gitignore` for Android build output, Gradle caches, training data,
  generated corpora, model output, and temporary audio files.
- [x] Add a trainer runner script for building the trainer image and launching
  `livekit-wakeword setup` plus `run`.
- [x] Create initial `AGENTS.md`.
- [x] Define project goal, data layout, Android direction, and Docker-first
  development expectations.
