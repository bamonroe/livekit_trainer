# TODO.md

Living backlog for `/data/livekit_trainer`.

Keep this file current whenever work is added, completed, deferred, or
discovered. Prefer small, actionable items with clear status.

## Active

- [ ] Test bulk scripted collection with real phone recordings and tune slice
  padding, negative sampling, and review output. See
  `docs/BULK_SCRIPTED_COLLECTION.md`.

## Recently done

- [x] Capture richer per-recording metadata the server used to drop. Each bulk
  and background take now carries a `capture` object (device manufacturer/model,
  OS version, app version, resolved input route, the mic's native
  `source_sample_rate_hz`/`source_channels` before the 16 kHz mono conversion,
  and a per-sitting `session_id`). The app records the route/native format from
  `AudioRecord.routedDevice` (`WavRecorder.describeInput`), persists it in
  SQLite (DB v6, `capture_*` columns on both recording tables), and emits it in
  the manifest. The sync server parses it (`capture_from_extra`), stores it on
  the `bulk_recordings` row (schema v2, COALESCE-preserved across reprocess).
  Tests: `capture_from_extra_reads_nested_capture_and_skips_blanks`,
  `capture_from_extra_absent_object_is_all_none`.
- [x] Collect real background noise: new "Record background" control on the app's
  Record page records a long ambient take (no script, no transcription). The
  bundle carries a `background_recordings[]` array; the sync server chops each
  take into fixed-length (~2s) clips under `data/real/<slug>/background/`
  (`slice_background_recording` / `align_background_recordings`, marker script so
  reprocess re-chunks without Whisper). Pooled into every wake word by
  `assemble_training_data.py`. Tests: `background_chunks_cover_source_and_drop_short_tail`.
- [x] Preserve hard negatives: the bulk slicer no longer discards wake phrases
  spoken in a near-miss frame. They are filed under the `negative` category (so
  the trainer treats them as negatives) but tagged with the distinct
  `hard_negative` label in the DB. `build_slice_row` now takes label + category
  separately. Test: `hard_negative_context_flags_near_miss_frames`.

- [x] Surface pooled clip counts in the app: `/projects` now returns per-slug
  positive/negative/background counts plus `pooled_negative_count` (other words'
  negatives + their positives reused as negatives); the Review "Sync & process"
  card shows a Training pool block and warns when positives are heavily
  outnumbered. `generate_config.py --positive-per-batch` emits `batch_n_per_class`
  to overweight positives (the correct lever — disk duplication is a no-op given
  the trainer's modulo-wraparound batching).
- [x] Cross-wake-word negative reuse: `scripts/assemble_training_data.py` builds
  a pooled `data/train/<slug>` tree where negatives include every other wake
  word's negatives and (as hard negatives) their positives; background is pooled
  too, positives stay own-only. `generate_config.py` now emits `real_samples_dir`
  (defaults to `./data/train`). Tests in `tests/test_assemble_training_data.py`.
- [x] Hard-cap slices at 1.5s (word span + padding clamped; transcript trimmed
  to audible words). Verified on `all_set`: all clips <= 1.5s, none dropped.
- [x] Add server reprocess endpoints (`POST /reprocess/:slug`,
  `POST /reprocess/:slug/:recording_id`) that re-slice stored `bulk_source`
  audio with no re-upload, plus `DELETE /bulk/:slug/:recording_id` to remove a
  recording, its slices, and files (cleans orphaned server-side takes).
- [x] Redesign the Android app: bottom nav (Record / Review / Settings),
  dedicated pages, clear "Sync & process" (was "Split batch"), project-wide and
  per-recording Reprocess buttons, server-aware delete, project picker chip.
  Default the app's Whisper URL to the WhisperX port to match the server.

## Later

- [ ] Surface background takes in the app's Review page for per-take replay and
  delete (they already round-trip through the server slices/review machinery;
  the app just doesn't list them yet).
- [ ] Emit distinct `false_positive` / `false_negative` labels from evaluation
  mistakes into correction batches (labels are already accepted end to end).
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
- [ ] Add prompt coverage reporting so future batches deliberately target
  underrepresented labels, conditions, and phrase variants.

## Done

- [x] Define the initial Android training bundle export format in
  `docs/TRAINING_BUNDLE_FORMAT.md`.
- [x] Build the first app screen for creating wake-word projects.
- [x] Add prompt batch data model and prompt generation rules.
- [x] Improve hard-negative prompt generation with pronounceable sound-alike
  phrase variants.
- [x] Add basic microphone recording flow for prompted clips.
- [x] Persist clip metadata for recorded prompt clips.
- [x] Add local clip listing, replay, and delete flow.
- [x] Add prompt-by-prompt advancement after each saved recording.
- [x] Refresh prompt batches after completion and shorten generated prompt text
  for focused wake-word clips.
- [x] Split settings into a dedicated app page, add optional Whisper server URL
  settings, and add a bulk-script recording mode entry point.
- [x] Send saved Android sync and Whisper settings to the Rust sync server.
- [x] Add first-pass bulk script recording export, Whisper word alignment, and
  server-side positive and negative clip slicing.
- [x] Tune bulk scripts toward natural read-aloud sentences and pad very early
  wake-phrase slices.
- [x] Add delete controls for saved bulk recordings in the Android app.
- [x] Add app-side review for generated bulk slices with server streaming and
  delete controls.
- [x] Use dictionary words and phonetic hard negatives in bulk scripts, and keep
  server alignment running across later bulk recordings after per-recording
  failures.
- [x] Skip already-processed bulk recordings during Android sync and highlight
  true wake phrases versus near-miss negatives in bulk scripts.
- [x] Make bulk script wake-word placement count configurable in Android
  settings.
- [x] Correct Whisper segment-relative word timestamps before slicing bulk
  recordings.
- [x] Add server project metadata loading so another device can review synced
  bulk slices without downloading source recordings.
- [x] Show compact slice IDs in Android bulk review rows for reporting examples.
- [x] Tune generated negative slices to prefer sentence boundaries and avoid
  cutting off sentence endings.
- [x] Show full positive slice transcript context and highlight wake phrases in
  bulk review.
- [x] Skip hard-negative contexts when creating positive bulk slices and clear
  stale generated slices during reprocessing.
- [x] Temporarily make positive bulk review slices use tight Whisper timestamp
  windows for alignment debugging.
- [x] Add source-recording alignment replay with timed Whisper words and cut
  markers in Android bulk review.
- [x] Use content hash IDs for generated bulk review slices and prune stale
  slice metadata during reprocessing.
- [x] Add a dedicated Android `Split batch` button for running server-side bulk
  alignment and slice generation from saved bulk recordings.
- [x] Make the Android main workflow bulk-first and remove the short-prompt
  collector from the primary screen.
- [x] Add basic positive/negative/all filtering for generated bulk review
  slices.
- [x] Make review audio controls toggle between play and pause, and rename
  per-slice alignment controls to source timing.
- [x] Split Android bulk collection into overview, new-recording, and
  per-recording detail pages.
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
