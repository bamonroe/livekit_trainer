# TODO.md

Living backlog for `/data/livekit_trainer`.

Keep this file current whenever work is added, completed, deferred, or
discovered. Prefer small, actionable items with clear status.

## Active

- [ ] In-app model-test flow (server-scored, decided). A dedicated "record
  test" prompt flow in the app, kept **separate** from training data: gives
  words to say, records one take, Whisper-slices it, and the server scores each
  cut word so the app can visualize per-word scores. Sliders for detection
  threshold and silence-pad amount; live true-positive / false-negative /
  false-positive counts vs Whisper ground truth. Remaining:
  - [x] Server scoring engine: `runtime/scorer/` HTTP service (mode `full` =
    continuous rolling window via a dense embedding bank computed once; mode
    `reset` = silence-pad each step). Validated exact vs `predict()` (~3e-8);
    `full` 10 ms is ~sub-second, `reset` near real-time at ~40 ms. Compose
    service `scorer` on :8770, reads `./output`.
  - [ ] Wire the sync-server to call the scorer for test takes and overlay
    Whisper word timings into per-word TP/FP/FN.
  - [ ] Separate "test" recording category/storage so test takes never enter
    the training pool.
  - [ ] App: "record test" prompt flow + score-curve/threshold/pad UI.
  - [ ] Host-runnable unit test for the scorer (currently validated only via
    the container smoke test).
  First model under test: `output/all_set/all_set.onnx` — see
  [streaming-recall-gap]: 99.8% synthetic recall but ~1% in continuous real
  speech; fires ~0.99 only under `reset`.

- [ ] Test bulk scripted collection with real phone recordings and tune slice
  padding, negative sampling, and review output. See
  `docs/BULK_SCRIPTED_COLLECTION.md`.

## Recently done

- [x] Checksum-based incremental sync: a sync no longer re-uploads and
  re-transcribes every take. The server stores each recording's source-WAV
  SHA-256 (`bulk_recordings.source_sha256`, schema v3) and returns an id→sha map
  from `GET /bulk/:slug/recordings`. Before uploading, the app hashes each local
  take and skips any the server already holds with a matching checksum (legacy
  rows with no checksum fall back to id-matching; a changed take re-uploads).
  When nothing is new the app says "Already up to date" and does no upload.
  Existing server rows backfill their checksum on the next reprocess. Verified:
  server unit test for the round-trip + reprocess preservation, and a live
  reprocess whose stored SHA matched the file's `sha256sum`.
- [x] Make the server the master record for recordings, manageable from any
  device. New `GET /bulk/:slug/recordings/detail` returns every recording with
  active slice counts and capture device; the app's Review page lists server
  recordings (not just local takes), labels each by capturing device ("This
  device" / "From <model>"), and lets any device delete any recording (local
  copy removed too). Local takes not yet uploaded show as pending. Verified on
  the emulator: an `all set` project with zero local takes shows all server
  takes attributed to the Pixel 8a. `db::recording_details` + `ServerRecording`
  + `loadServerRecordings`, with a server unit test.
- [x] Fixed the pool-count vs visible-takes mismatch: old synced takes lingered
  on the server with no way to see or delete them from the phone. Purged the
  five orphaned `all_set` takes (server back to 34 positives across 3 takes).
- [x] Follow the system light/dark theme (System/Light/Dark) and fix the
  status-bar icons staying dark-on-dark in dark mode.
- [x] Fixed a startup crash from the version-6 capture-column migration
  re-adding columns the background table's CREATE already had (idempotent now).

- [x] Personalization tooling so one user's voice can actually shape the model
  against the trainer's large synthetic pool (see `docs/PERSONALIZATION.md`).
  Three levers: (1) app **Dense** bulk-script style (Settings → Bulk script
  style) generates short, prosodically varied wake-phrase repetitions with
  frequent near misses, packing many positives/hard negatives per minute
  (`PromptGenerator.denseBulkScript`); (2) `assemble_training_data.py
  --positive-boost N` replicates the target's own positives N times to raise
  their pool share; (3) `generate_config.py --personal` shrinks `n_samples` to
  3000 and bumps per-batch positives to 100. Tests: `test_generate_config.py`
  plus boost cases in `test_assemble_training_data.py`. Also removed the
  duplicate upstream trainer clone and 28 GB of legacy corpora under
  `/data/storage/livekit-wakeword` — our repo drives the pip-installed trainer,
  so the clone was only a reference.
- [x] Add a dataset summary report: `scripts/dataset_report.py` reads the sync
  server SQLite DB and prints, per wake word and pooled, active slice counts by
  category and by the richer label (hard negatives shown separately), source
  recording counts, and — when the capture provenance columns are present —
  breakdowns by device, input route, and session. Tolerates older DBs without
  the capture columns. Tests in `tests/test_dataset_report.py`.
- [x] Surface background takes in the app's Review page: a "Background takes"
  card lists each ambient take with local playback (Play/Pause) and a Delete
  that removes the local file, the SQLite row, and the server-side clips (reusing
  `DELETE /bulk/:slug/:recording_id`, which already accepts background ids).
- [x] Emit distinct `false_positive` / `false_negative` labels from evaluation
  mistakes: new `scripts/import_corrections.py` reads a correction batch
  (`corrections.json` + `audio/`), compares each clip's score to the detection
  threshold, and files only the mistakes into `data/real/<slug>/{positive,
  negative}/` with the mistake label preserved in `metadata.jsonl`. Misses
  (`false_negative`) train as positives, wrongful fires (`false_positive`) as
  negatives; correct clips are skipped. Format in
  `docs/CORRECTION_BATCH_FORMAT.md`; tests in `tests/test_import_corrections.py`.
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

- [ ] Add optional runtime scorer service or test harness under `runtime/`. When
  it lands it should emit `corrections.json` (see `docs/CORRECTION_BATCH_FORMAT.md`)
  so live detection mistakes flow straight back into training.
- [ ] Add a post-sync cleanup policy for app-private clips after server import
  is acknowledged.
- [ ] Add emulator or instrumentation coverage for SQLite metadata migration
  and app-private WAV file retention.
- [ ] Add evaluation tooling that creates follow-up false-positive and
  false-negative collection batches.
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
