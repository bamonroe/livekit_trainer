# TODO.md

Living backlog for `/data/livekit_trainer`.

Keep this file current whenever work is added, completed, deferred, or
discovered. Prefer small, actionable items with clear status.

## Active

- [ ] Add `Dockerfile.dev` for containerized Android and pipeline development.
- [ ] Add `docker-compose.yml` for the development container and persistent
  caches.
- [ ] Add microphone recording flow for prompted clips.
- [ ] Add local clip review, replay, and delete flow.
- [ ] Add import tooling to convert exported sessions into
  `data/real/<wake_word_slug>/{positive,negative,background}`.
- [ ] Add config-generation tooling for LiveKit wake-word YAML files.
- [ ] Add a smoke-test training path using a tiny calibration config.
- [ ] Define the model artifact source-of-truth policy and export conventions
  for downstream projects.
- [ ] Decide Android persistence layer and audio storage strategy.

## Later

- [ ] Add optional runtime scorer service or test harness under `runtime/`.
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
