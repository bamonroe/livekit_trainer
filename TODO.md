# TODO.md

Living backlog for `/data/livekit_trainer`.

Keep this file current whenever work is added, completed, deferred, or
discovered. Prefer small, actionable items with clear status.

## Active

- [ ] Scaffold the Android app under `android/` with a Gradle wrapper compatible
  with `/data/android/build.sh`.
- [ ] Add `Dockerfile.dev` for containerized Android and pipeline development.
- [ ] Add `docker-compose.yml` for the development container and persistent
  caches.
- [ ] Define the Android training bundle export format.
- [ ] Add import tooling to convert exported sessions into
  `data/real/<wake_word_slug>/{positive,negative,background}`.
- [ ] Add config-generation tooling for LiveKit wake-word YAML files.
- [ ] Add a smoke-test training path using a tiny calibration config.
- [ ] Define the model artifact source-of-truth policy and export conventions
  for downstream projects.
- [ ] Decide Android package name, minimum SDK, persistence layer, and audio
  storage strategy.

## Later

- [ ] Add optional runtime scorer service or test harness under `runtime/`.
- [ ] Add evaluation tooling that creates follow-up false-positive and
  false-negative collection batches.
- [ ] Add dataset summary reports by wake word, label, phrase, session, and
  device.
- [ ] Add audio validation for sample rate, channel count, duration, clipping,
  and silence.

## Done

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
