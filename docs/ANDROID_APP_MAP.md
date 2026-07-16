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
- Prompt generation currently creates a deterministic mixed preview per project.
- Prompt recording advances through the generated batch one prompt at a time.
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
2. Generate a randomized prompt batch.
3. Record prompted clips.
4. Label clips as `positive`, `negative`, `hard_negative`, `background`,
   `false_positive`, or `false_negative`.
5. Review, replay, and delete clips.
6. Show collection counts by label and phrase.
7. Export a training bundle.

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

The app can upload a zipped bundle directly to the repo-side sync receiver.
Start it from the repo root:

```bash
scripts/sync_server.py --host 0.0.0.0 --port 8765
```

On this machine the default app URL is:

```text
http://100.64.0.2:8765
```

`POST /sync` stores the raw upload under `incoming/bundles/`, validates the
bundle, and imports clips into `data/real/<wake_word_slug>/`. Repeated syncs are
idempotent for already-imported clip files.

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
