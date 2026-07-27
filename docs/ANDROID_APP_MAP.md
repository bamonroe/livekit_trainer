# Android App Bring-Up Map

This map tracks the path from no Android app to a buildable, testable wake-word
sample collection app.

## Local Android Build Context

Use the shared Android tooling in `/data/android`.

Important files:

- `/data/android/README.md` explains the headless emulator workflow.
- `/data/android/BUILD-ENV-PLAN.md` explains the containerized build plan.
- `/data/android/build.sh` builds Android projects in a disposable container.
- `/data/android/docker-compose.yml` runs the headless emulator.

Current builder facts:

- Builder image: `android-builder:local`.
- JDK: 21.
- SDKs baked into the image: Android 34 and Android 35.
- Each app keeps its own Gradle cache at `<project>/.gradle-cache/`.
- The build script mounts the project parent at `/workspace` and works from
  `/workspace/<project-dir-name>`.

Build command shape:

```bash
/data/android/build.sh /data/livekit_trainer/android :app:assembleDebug
```

Emulator command shape:

```bash
cd /data/android
docker compose up -d --build
docker exec android-emulator adb wait-for-device
docker exec android-emulator adb shell getprop sys.boot_completed
```

The emulator requires `/dev/kvm`, which depends on VT-x being enabled in BIOS.
See `/data/android/README.md` before relying on emulator tests.

## App Scaffold

Create the Android project under:

```text
android/
```

The project should include its own Gradle wrapper so `/data/android/build.sh`
can build it without host Android tooling.

Initial stack preference:

- Kotlin.
- Native Android UI first, with room to move to Jetpack Compose later when the
  recorder workflow needs richer state handling.
- SQLite-backed local persistence for projects, prompt state, sessions, and
  clips. Room can be introduced later if query complexity starts to justify it.
- Android-native audio recording APIs.
- Local-only storage by default.

Current scaffold:

- Package: `com.bam.livekittrainer`.
- Minimum SDK: 26.
- Target SDK: 35.
- Compile SDK: 35.
- Local project and clip metadata uses an app-private SQLite database at
  `wake_word_collection.db`.
- The SQLite store includes a one-time migration from the original
  `SharedPreferences` metadata format.
- Prompt generation creates deterministic mixed batches per project and batch
  number. Completing a batch advances to a fresh lexicon-backed batch instead of
  looping the same prompts forever.
- The primary collection workflow is now bulk-script recording, server-side
  split generation, and review of generated slices. The old one-prompt-at-a-time
  collector is no longer part of the primary project screen.
- Navigation is a **persistent top app bar** plus a **collapsible left
  navigation rail** (hand-rolled from framework views — no DrawerLayout, no
  AndroidX). The top bar carries, left to right: a hamburger (☰) that toggles
  the rail, the project dropdown chip (opens the project picker), a flexible
  spacer, and a settings gear (⚙) that navigates to the Settings page. The left
  rail lists the primary destinations Record / Review / Test / Train (Settings
  now lives on the gear, not the rail); the active destination is highlighted.
  While Review (or its Detail child) is the active destination, the rail nests
  the Review sub-sections as indented sub-rows beneath it — Wake word /
  Negatives / Hard negatives / Background / Synthetic — with the active
  sub-section highlighted (`reviewSubRow`). The Review page shows only one of
  these sections at a time (see below); the chosen section is held in the
  `reviewSection` activity field so an in-place `render()` keeps it.
  The rail overlays the workspace from the left with a scrim behind it; tapping
  the scrim or a destination, or tapping the hamburger again, closes it. It also
  opens on a left-edge left-to-right swipe and closes on a right-to-left swipe
  (handled by a custom `FrameLayout.onInterceptTouchEvent`). `railOpen` is an
  activity field so an in-place `render()` never disturbs the rail. The old
  five-tab bottom navigation bar was replaced by this top bar + rail.
  Note: a right-to-left close swipe that runs all the way into the left
  system-gesture edge zone can also trigger Android's own back gesture; ending
  the swipe short of the edge closes only the rail.
- Settings include the sync server URL, an optional Whisper server URL,
  appearance controls, and reset controls.
- Prompt recording supports direct prompt picking, previous, and skip controls
  so the user is not forced to restart or follow the generated order exactly.
- The project page has a recording-mode selector for short prompted collection
  and bulk scripted collection. Bulk mode records one long script WAV and exports
  the exact script for server-side Whisper alignment and slicing. Bulk scripts
  use generated natural read-aloud sentences with dictionary content words,
  phonetic near-miss phrases, and neutral lead-in before the first wake phrase.
  The displayed script highlights true wake phrases in bold green and near-miss
  hard negatives in bold red. Settings include a configurable wake-placement
  count for controlling bulk script length.
- Basic recording currently writes 16 kHz mono PCM WAV files into app-private
  storage under `filesDir/clips/<wake_word_slug>/`.
- Bundle export currently writes unzipped training bundles into app-private
  storage under `filesDir/exports/<wake_word_slug>_<timestamp>/`.
- Build command verified:

```bash
/data/android/build.sh /data/livekit_trainer/android :app:assembleDebug
```

Verified output:

```text
android/app/build/outputs/apk/debug/app-debug.apk
```

## First App Features

1. Create a wake-word project with a phrase and slug.
2. Generate a randomized short prompt batch.
3. Pick any prompt from the batch and record clips in the order that makes
   sense during collection.
4. Switch to bulk script mode to record a generated long read-aloud script for
   server-side Whisper alignment and slicing.
5. Label clips as `positive`, `negative`, `hard_negative`, `background`,
   `false_positive`, or `false_negative`.
6. Review, replay, and delete clips.
7. Show collection counts by label and phrase.
8. Export a training bundle.

## Export Contract

The export format is defined in `docs/TRAINING_BUNDLE_FORMAT.md`.

The first export format is simple and explicit:

```text
bundle/
  manifest.json
  audio/
    <clip_id>.wav
```

`manifest.json` should contain:

- Bundle schema version.
- Wake-word project id, slug, and phrase.
- Prompt batch id.
- Clip records with filename, label, spoken phrase, timestamp, device metadata,
  sample rate, channel count, duration, and session id.

The importer in this repo will convert the bundle into:

```text
data/real/<wake_word_slug>/
  positive/
  negative/
  background/
```

Richer labels should remain in metadata even when mapped to the current trainer
categories.

## Server Sync

The app can upload a zipped bundle directly to the repo-side Rust sync server.
Start it from the repo root with Docker Compose:

```bash
docker compose up -d --build sync-server
```

On this machine the default app URL is:

```text
http://100.64.0.2:8765
```

The container exposes `POST /sync` on port `8765`, stores the raw upload under
`incoming/bundles/`, validates the bundle, and imports clips into
`data/real/<wake_word_slug>/`. Repeated syncs are idempotent for already-imported
clip files.

When settings are saved, the Android app sends the sync server URL and optional
Whisper server URL to `POST /settings`; the sync server persists them in
`data/server_settings.json`. When configured, the app also sends the optional
Whisper server URL in the `X-Whisper-Server-Url` request header during
`POST /sync`, and the sync server echoes that value in the sync response so
alignment tooling can be connected next.

Settings can also load server projects with `GET /projects`. This imports only
wake-word project metadata into the local SQLite database, so a tablet can
select a project created on a phone and load server-side bulk review slices
without downloading the original phone recordings.

If a bundle includes `bulk_recordings`, the sync server posts the long WAV to
the configured Whisper server with `response_format=verbose_json` and
`word_timestamps=true`. It slices positive clips around aligned wake-phrase
occurrences and negative clips from nearby non-wake speech, then writes them
into the normal `data/real/<wake_word_slug>/` layout with provenance metadata.
When a wake phrase is too close to the start of the recording for full pre-roll
padding, the server adds extra trailing context so the positive slice is still
long enough to train on.
The server treats each bulk recording independently, so a Whisper or slicing
failure on one long recording is reported as a warning and does not stop later
bulk recordings from being processed.

Before uploading, the Android app calls `GET /bulk/<wake_word_slug>/recordings`
and leaves out long bulk WAVs whose IDs already have generated server slices.

Bulk mode can also load generated slice review data back from the sync server.
The project overview has a `Split batch` button that uploads all saved bulk
recordings for the selected project to `POST /sync`, runs server-side Whisper
alignment and slicing, then reloads generated review clips. Recording a new bulk
script is a separate page, and each saved bulk recording opens a detail page
with the original script, source timing, and generated slices from that
recording. The app also calls
`GET /review/<wake_word_slug>/bulk`, lists the generated positive and negative
slices with Whisper transcript text, timing, duration, and average word
confidence, shows the first six characters of each generated slice hash for
reporting bad examples, highlights the wake phrase green in positive slices and
red in negative slices, streams each slice from
`GET /review/<slug>/<category>/<file>`, and can reject a bad slice with
`DELETE /review/<slug>/<category>/<file>`. Review slices can be filtered between
all, positive, and negative clips.

Each review row can also open source timing for its source recording. The app
loads `GET /review/<slug>/bulk/<recording_id>/alignment`, streams the original
bulk WAV from `GET /review/<slug>/bulk/<recording_id>/audio`, highlights the
current Whisper word during playback, and shows generated cut markers inline.
Slice and source playback buttons toggle between play and pause while their
audio is active.

The Review page is split into **per-kind sub-sections chosen from the left
rail**, showing only one section's card(s) at a time (`renderReviewPage`
switches on `reviewSection`). The `Sync & process` card is global to Review and
always stays on top, above whichever section is selected. The sections are:
Wake word (the POSITIVE `recordingsCard`), Negatives, Hard negatives (their
respective `recordingsCard`s, always shown even when empty so the section stays
visible), Background (`backgroundRecordingsCard`), and Synthetic. The Synthetic
section stacks **two independent `syntheticSamplesCard(project, source, …)`
cards**, one per synthetic-positive source the server exposes behind a
`?source=f5|zonos` query param: "F5 voice clones" and "Zonos voice clones
(prosody levers)". Each card loads/plays its source's sample spread, runs its own
"Generate N now" and "Delete all", and polls its own generation status. All the
per-source UI state is keyed by source string in `HashMap`s (`syntheticSlug`,
`syntheticSamples`, `loadingSynthetic`, `synthGenSlug`, `synthGenerating`,
`synthGenMessage`) so the two cards never clobber each other; the playback key is
`synthetic:<source>:<id>`. The client methods
(`loadSyntheticSamples` / `startSyntheticGeneration` / `deleteSyntheticSamples` /
`syntheticGenerationStatus` and the audio URL) all take a `source` param that
defaults to `"f5"`. Legacy ENROLLMENT and "other"/mixed takes have no
recorder of their own; they are folded under the Wake word section as extra
cards ("Enrollment takes (legacy)" / "Other takes (legacy)") that surface only
when such takes actually exist, so legacy takes stay reviewable and deletable
without a permanent near-empty section.

The Review page also lists saved **background takes** (the ambient/non-speech
recordings from the Record page's "Record background" control) in the Background
section. Each row plays the local take back with a Play/Pause button and
can delete it; delete removes the local WAV, its SQLite row, and the server-side
background clips via `DELETE /bulk/<slug>/<recording_id>`.

The **Train page** (`renderTrainPage`) exposes the model's positive-input knobs
plus one negative knob. A model's total *positive* input is four streams the
live calculator sums:
`total = n_samples (Kokoro built-in TTS) + f5_count + zonos_count + realPositives × positive_boost`.
A fifth field, **Impostor negatives (female voices)** (`impostor_neg_count`),
sits under a "Negatives" header; it is the wake phrase spoken by other people's
voices and is shown on its own breakdown line (`+ N impostor negatives`),
**never summed into the positive total**. Each stream has its own numeric field
(Kokoro / F5 / Zonos clips, positive boost, impostor negatives); the calculator
(`recomputeTotal`) re-renders the breakdown as any field changes. The counts
persist via `saveTrainNumbers` (`KEY_TRAIN_KOKORO_COUNT` / `KEY_TRAIN_F5_COUNT` /
`KEY_TRAIN_ZONOS_COUNT` / `KEY_TRAIN_IMPOSTOR_NEG_COUNT`) and are sent in the
train-request JSON as `n_samples` / `f5_count` / `zonos_count` /
`impostor_neg_count` (`buildTrainRequestBody`). At train time the sync-server
pools up to `f5_count` F5 clips and `zonos_count` Zonos clips as two of the three
positive sources, and up to `impostor_neg_count` Kokoro impostor clips into the
negative set. During the pre-train generation phase the status renderer maps
`f5gen` / `zonosgen` / `impostorgen` to their live `<tag>_wrote` /
`<tag>_requested` counts so each source's file count ticks up on-screen.

## Build And Test Loop

1. Build with `/data/android/build.sh /data/livekit_trainer/android`.
2. If the emulator is available, install the debug APK with `adb install -r`.
3. Launch with `adb shell monkey -p <package> -c android.intent.category.LAUNCHER 1`.
4. Capture screenshots with `adb exec-out screencap -p`.
5. For audio paths, add unit tests around WAV writing, sample conversion, and
   export manifest creation before relying on device testing.

## Open Decisions

- Whether to keep native Android views or move to Jetpack Compose.
- Whether to add Room once prompt sessions, correction batches, and review
  filters need richer queries.
- Whether to keep writing WAV at capture time long term or store raw PCM first
  and encode WAV on export.
- Whether synced clips should be pruned from app-private storage after the
  server acknowledges import.
