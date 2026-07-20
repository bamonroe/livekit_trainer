//! SQLite accounting layer for the sync server.
//!
//! The database is the single source of truth for how raw bulk recordings,
//! their transcripts, the prompts that produced them, and the derived slices
//! relate to each other. WAV audio still lives on disk, but every file on disk
//! is tracked by exactly one row here, so reprocessing a recording is a clean
//! atomic swap instead of log rewriting and filename-prefix guessing.

use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;

/// Bumped whenever the schema changes so `migrate` can evolve an existing file.
const SCHEMA_VERSION: i64 = 7;

/// Per-take capture provenance forwarded by the app: which device and input
/// route recorded it, the mic's native format before the app's 16 kHz mono
/// conversion, and the session that groups takes recorded together. Every field
/// is optional so a reprocess (which has no fresh manifest) can leave prior
/// values untouched. Stored on the `bulk_recordings` row, which also backs
/// background takes.
#[derive(Debug, Clone, Default)]
pub struct CaptureMeta {
    pub device_manufacturer: Option<String>,
    pub device_model: Option<String>,
    pub os_version: Option<String>,
    pub app_version: Option<String>,
    pub input_route: Option<String>,
    pub source_sample_rate_hz: Option<i64>,
    pub source_channels: Option<i64>,
    pub session_id: Option<String>,
}

/// A word with millisecond timing, as stored under a transcript.
#[derive(Debug, Clone)]
pub struct WordRow {
    pub word: String,
    pub start_ms: i64,
    pub end_ms: i64,
    pub probability: f64,
}

/// A derived slice ready to be recorded in the database. The WAV file it points
/// at is written to disk by the caller before this is committed.
#[derive(Debug, Clone)]
pub struct SliceRow {
    pub id: String,
    pub label: String,
    pub category: String,
    pub spoken_phrase: String,
    pub source_start_ms: i64,
    pub source_end_ms: i64,
    pub avg_probability: f64,
    pub word_count: i64,
    pub wav_path: String,
    pub file_name: String,
}

/// Everything needed to replace one recording's alignment in a single
/// transaction: the recording, the fresh transcript, its words and prompts,
/// and the slices cut from it.
#[derive(Debug, Clone)]
pub struct RecordingAlignment {
    pub recording_id: String,
    pub project_slug: String,
    pub phrase: String,
    pub external_id: Option<String>,
    pub script: String,
    /// How the take should be sliced: `positive` (wake phrase repeated, cut into
    /// positives), `negative` (ordinary speech cut into negatives),
    /// `hard_negative` (near-miss phrases cut into hard negatives), or `mixed`
    /// (legacy single-script take with all three inferred from context).
    pub kind: String,
    pub recorded_at: String,
    pub duration_ms: i64,
    pub source_wav: String,
    pub source_sha256: Option<String>,
    pub bundle: Option<String>,
    pub transcript_text: String,
    pub whisper_url: Option<String>,
    pub words: Vec<WordRow>,
    pub prompts: Vec<String>,
    pub slices: Vec<SliceRow>,
    pub capture: CaptureMeta,
}

/// An imported single clip (non-bulk) tracked for training export.
#[derive(Debug, Clone)]
pub struct ClipRow {
    pub id: String,
    pub project_slug: String,
    pub label: String,
    pub category: String,
    pub spoken_phrase: String,
    pub wav_path: String,
    pub source_file: String,
    pub bundle: Option<String>,
}

/// Row shape returned to the bulk review screen.
#[derive(Debug, Clone)]
pub struct ReviewSlice {
    pub id: String,
    pub label: String,
    pub spoken_phrase: String,
    pub source_recording: String,
    pub source_start_ms: i64,
    pub source_end_ms: i64,
    pub duration_ms: i64,
    pub avg_probability: f64,
    pub word_count: i64,
    pub category: String,
    pub file_name: String,
}

/// A slice cut, as shown on the alignment timeline.
#[derive(Debug, Clone)]
pub struct CutRow {
    pub id: String,
    pub label: String,
    pub start_ms: i64,
    pub end_ms: i64,
}

/// Project summary for the overview screen.
#[derive(Debug, Clone)]
pub struct ProjectRow {
    pub slug: String,
    pub phrase: String,
    pub external_id: String,
    pub created_at_ms: i64,
    pub bulk_slice_count: i64,
    pub positive_count: i64,
    pub negative_count: i64,
    pub background_count: i64,
    /// Own negatives that are eligible to be pooled into other projects — plain
    /// negatives only, excluding project-scoped hard negatives.
    pub poolable_negative_count: i64,
}

/// Everything a reprocess pass needs about a stored recording to re-run
/// alignment from its already-saved source WAV, without a fresh upload.
#[derive(Debug, Clone)]
pub struct RecordingMeta {
    pub id: String,
    pub project_slug: String,
    pub script: String,
    pub kind: String,
    pub recorded_at: String,
    pub duration_ms: i64,
}

/// Open (creating if needed) the database and apply the schema.
pub fn open(path: &Path) -> Result<Connection, rusqlite::Error> {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let conn = Connection::open(path)?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    conn.pragma_update(None, "busy_timeout", 5000)?;
    migrate(&conn)?;
    Ok(conn)
}

fn migrate(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS projects (
            slug          TEXT PRIMARY KEY,
            phrase        TEXT NOT NULL,
            external_id   TEXT,
            created_at_ms INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS bulk_recordings (
            id            TEXT PRIMARY KEY,
            project_slug  TEXT NOT NULL REFERENCES projects(slug) ON DELETE CASCADE,
            script        TEXT NOT NULL,
            recorded_at   TEXT,
            duration_ms   INTEGER,
            source_wav    TEXT NOT NULL,
            bundle        TEXT,
            imported_at_ms INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS prompts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            recording_id  TEXT NOT NULL REFERENCES bulk_recordings(id) ON DELETE CASCADE,
            ordinal       INTEGER NOT NULL,
            text          TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS transcripts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            recording_id  TEXT NOT NULL REFERENCES bulk_recordings(id) ON DELETE CASCADE,
            version       INTEGER NOT NULL,
            text          TEXT,
            whisper_url   TEXT,
            is_current    INTEGER NOT NULL DEFAULT 1,
            created_at_ms INTEGER NOT NULL,
            UNIQUE(recording_id, version)
        );

        CREATE TABLE IF NOT EXISTS transcript_words (
            transcript_id INTEGER NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
            ordinal       INTEGER NOT NULL,
            word          TEXT NOT NULL,
            start_ms      INTEGER NOT NULL,
            end_ms        INTEGER NOT NULL,
            probability   REAL,
            PRIMARY KEY (transcript_id, ordinal)
        );

        CREATE TABLE IF NOT EXISTS slices (
            id             TEXT PRIMARY KEY,
            recording_id   TEXT NOT NULL REFERENCES bulk_recordings(id) ON DELETE CASCADE,
            transcript_id  INTEGER REFERENCES transcripts(id) ON DELETE SET NULL,
            project_slug   TEXT NOT NULL REFERENCES projects(slug) ON DELETE CASCADE,
            label          TEXT NOT NULL,
            category       TEXT NOT NULL,
            spoken_phrase  TEXT,
            source_start_ms INTEGER NOT NULL,
            source_end_ms  INTEGER NOT NULL,
            duration_ms    INTEGER NOT NULL,
            avg_probability REAL,
            word_count     INTEGER,
            wav_path       TEXT NOT NULL,
            file_name      TEXT NOT NULL,
            status         TEXT NOT NULL DEFAULT 'active',
            created_at_ms  INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS score_curves (
            recording_id  TEXT NOT NULL,
            mode          TEXT NOT NULL,
            step_ms       INTEGER NOT NULL,
            keep_ms       INTEGER NOT NULL,
            model_fp      TEXT NOT NULL,
            duration_ms   REAL NOT NULL,
            times_ms      TEXT NOT NULL,
            scores        TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL,
            PRIMARY KEY (recording_id, mode, step_ms, keep_ms)
        );

        CREATE TABLE IF NOT EXISTS clips (
            id            TEXT PRIMARY KEY,
            project_slug  TEXT NOT NULL REFERENCES projects(slug) ON DELETE CASCADE,
            label         TEXT NOT NULL,
            category      TEXT NOT NULL,
            spoken_phrase TEXT,
            wav_path      TEXT NOT NULL,
            source_file   TEXT,
            bundle        TEXT,
            imported_at_ms INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS training_runs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            slug          TEXT NOT NULL,
            steps         INTEGER NOT NULL,
            model_size    TEXT,
            personal      INTEGER NOT NULL DEFAULT 0,
            started_at    TEXT NOT NULL,
            finished_at   TEXT,
            duration_ms   INTEGER,
            state         TEXT NOT NULL,
            UNIQUE(slug, started_at)
        );

        -- One row per trained model artifact, versioned by run_id so retraining a
        -- wake word never overwrites prior models. Linked to the wake word by a
        -- real foreign key; the current live model (the one behind
        -- output/<slug>/<slug>.onnx) is flagged with is_current = 1.
        CREATE TABLE IF NOT EXISTS models (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            slug          TEXT NOT NULL REFERENCES projects(slug) ON DELETE CASCADE,
            run_id        TEXT NOT NULL,
            onnx_path     TEXT NOT NULL,
            onnx_sha256   TEXT,
            pt_sha256     TEXT,
            steps         INTEGER,
            model_size    TEXT,
            personal      INTEGER NOT NULL DEFAULT 0,
            git_commit    TEXT,
            metrics       TEXT,
            started_at    TEXT,
            finished_at   TEXT,
            created_at_ms INTEGER NOT NULL,
            is_current    INTEGER NOT NULL DEFAULT 0,
            -- Version 6: full training provenance. Every knob that shaped the
            -- model, the real vs synthetic data counts, and the complete manifest
            -- verbatim (params_json) so nothing is ever lost even as knobs evolve.
            model_type          TEXT,
            token_type          TEXT,
            positive_boost      INTEGER,
            n_samples           INTEGER,
            n_samples_val       INTEGER,
            positive_per_batch  INTEGER,
            target_fp_per_hour  REAL,
            context_fix         INTEGER,
            real_positive       INTEGER,
            real_negative       INTEGER,
            real_background     INTEGER,
            trainer_image       TEXT,
            onnx_bytes          INTEGER,
            eval                TEXT,
            params_json         TEXT,
            UNIQUE(slug, run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_models_slug ON models(slug, created_at_ms DESC);

        -- Pending/active training jobs. The training scheduler runs at most one
        -- trainer container at a time (single GPU) and dispatches these oldest
        -- first. `params_json` is a validated ValidatedTrain payload rebuilt into
        -- the docker argv at dispatch; `state` walks queued -> running ->
        -- succeeded/failed/canceled.
        CREATE TABLE IF NOT EXISTS train_queue (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            slug           TEXT NOT NULL,
            params_json    TEXT NOT NULL,
            state          TEXT NOT NULL DEFAULT 'queued',
            container_name TEXT,
            enqueued_at_ms INTEGER NOT NULL,
            started_at_ms  INTEGER,
            finished_at_ms INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_train_queue_state ON train_queue(state, id);

        CREATE INDEX IF NOT EXISTS idx_slices_project ON slices(project_slug, status);
        CREATE INDEX IF NOT EXISTS idx_slices_recording ON slices(recording_id, status);
        CREATE INDEX IF NOT EXISTS idx_slices_file ON slices(project_slug, category, file_name);
        CREATE INDEX IF NOT EXISTS idx_bulk_project ON bulk_recordings(project_slug);
        CREATE INDEX IF NOT EXISTS idx_words_transcript ON transcript_words(transcript_id, ordinal);
        "#,
    )?;

    // Version 2: per-take capture provenance on bulk_recordings. Added via
    // guarded ALTERs so both fresh databases (columns absent from the CREATE
    // above) and existing ones evolve to the same shape.
    for (name, decl) in [
        ("capture_device_manufacturer", "TEXT"),
        ("capture_device_model", "TEXT"),
        ("capture_os_version", "TEXT"),
        ("capture_app_version", "TEXT"),
        ("capture_input_route", "TEXT"),
        ("capture_source_sample_rate_hz", "INTEGER"),
        ("capture_source_channels", "INTEGER"),
        ("capture_session_id", "TEXT"),
        // Version 3: full-file SHA-256 of the source WAV, so a device can ask the
        // server which takes it already holds and skip re-uploading unchanged
        // recordings.
        ("source_sha256", "TEXT"),
        // Version 7: how this take should be sliced (positive/negative/
        // hard_negative), replacing the old single mixed-script inference. Legacy
        // rows read back as 'mixed' via COALESCE and keep their prior behavior.
        ("kind", "TEXT"),
    ] {
        if !column_exists(conn, "bulk_recordings", name)? {
            conn.execute(
                &format!("ALTER TABLE bulk_recordings ADD COLUMN {name} {decl}"),
                [],
            )?;
        }
    }

    // Version 6: full training-provenance columns on `models`. Guarded ALTERs so
    // databases created before v6 evolve to the same shape as a fresh CREATE.
    for (name, decl) in [
        ("model_type", "TEXT"),
        ("token_type", "TEXT"),
        ("positive_boost", "INTEGER"),
        ("n_samples", "INTEGER"),
        ("n_samples_val", "INTEGER"),
        ("positive_per_batch", "INTEGER"),
        ("target_fp_per_hour", "REAL"),
        ("context_fix", "INTEGER"),
        ("real_positive", "INTEGER"),
        ("real_negative", "INTEGER"),
        ("real_background", "INTEGER"),
        ("trainer_image", "TEXT"),
        ("onnx_bytes", "INTEGER"),
        ("eval", "TEXT"),
        ("params_json", "TEXT"),
    ] {
        if !column_exists(conn, "models", name)? {
            conn.execute(
                &format!("ALTER TABLE models ADD COLUMN {name} {decl}"),
                [],
            )?;
        }
    }

    conn.pragma_update(None, "user_version", SCHEMA_VERSION)?;
    Ok(())
}

/// Whether a column already exists on a table, so migrations can add it once.
fn column_exists(conn: &Connection, table: &str, column: &str) -> Result<bool, rusqlite::Error> {
    let count: i64 = conn.query_row(
        &format!("SELECT COUNT(*) FROM pragma_table_info('{table}') WHERE name = ?1"),
        params![column],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

/// Ensure a project row exists, filling phrase/external_id and stamping the
/// creation time the first time we see it.
pub fn upsert_project(
    conn: &Connection,
    slug: &str,
    phrase: &str,
    external_id: Option<&str>,
    now_ms: i64,
) -> Result<(), rusqlite::Error> {
    conn.execute(
        "INSERT INTO projects (slug, phrase, external_id, created_at_ms)
         VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(slug) DO UPDATE SET
            phrase = excluded.phrase,
            external_id = COALESCE(excluded.external_id, projects.external_id)",
        params![slug, phrase, external_id, now_ms],
    )?;
    Ok(())
}

/// Paths of the active WAV slices currently on disk for a recording. Used to
/// delete stale files before a reprocess writes the new set.
pub fn active_slice_paths(
    conn: &Connection,
    recording_id: &str,
) -> Result<Vec<String>, rusqlite::Error> {
    let mut stmt = conn.prepare("SELECT wav_path FROM slices WHERE recording_id = ?1")?;
    let rows = stmt.query_map(params![recording_id], |row| row.get::<_, String>(0))?;
    rows.collect()
}

/// Atomically replace a recording's alignment: upsert the project and
/// recording, retire prior transcripts, delete prior slices/prompts, and write
/// the fresh transcript, words, prompts and slices. Returns the new version.
pub fn store_recording_alignment(
    conn: &mut Connection,
    alignment: &RecordingAlignment,
    now_ms: i64,
) -> Result<i64, rusqlite::Error> {
    let tx = conn.transaction()?;

    upsert_project(
        &tx,
        &alignment.project_slug,
        &alignment.phrase,
        alignment.external_id.as_deref(),
        now_ms,
    )?;

    let capture = &alignment.capture;
    tx.execute(
        "INSERT INTO bulk_recordings
            (id, project_slug, script, recorded_at, duration_ms, source_wav, bundle, imported_at_ms,
             capture_device_manufacturer, capture_device_model, capture_os_version,
             capture_app_version, capture_input_route, capture_source_sample_rate_hz,
             capture_source_channels, capture_session_id, source_sha256, kind)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)
         ON CONFLICT(id) DO UPDATE SET
            project_slug = excluded.project_slug,
            script = excluded.script,
            -- Keep a prior kind if this pass carries none (defensive; reprocess
            -- passes the stored kind back in).
            kind = COALESCE(excluded.kind, bulk_recordings.kind),
            recorded_at = excluded.recorded_at,
            duration_ms = excluded.duration_ms,
            source_wav = excluded.source_wav,
            bundle = excluded.bundle,
            -- Keep a prior checksum if this pass carries none (e.g. reprocess).
            source_sha256 = COALESCE(excluded.source_sha256, bulk_recordings.source_sha256),
            -- Reprocess carries no manifest, so keep prior provenance when the
            -- incoming values are null instead of erasing it.
            capture_device_manufacturer = COALESCE(excluded.capture_device_manufacturer, bulk_recordings.capture_device_manufacturer),
            capture_device_model = COALESCE(excluded.capture_device_model, bulk_recordings.capture_device_model),
            capture_os_version = COALESCE(excluded.capture_os_version, bulk_recordings.capture_os_version),
            capture_app_version = COALESCE(excluded.capture_app_version, bulk_recordings.capture_app_version),
            capture_input_route = COALESCE(excluded.capture_input_route, bulk_recordings.capture_input_route),
            capture_source_sample_rate_hz = COALESCE(excluded.capture_source_sample_rate_hz, bulk_recordings.capture_source_sample_rate_hz),
            capture_source_channels = COALESCE(excluded.capture_source_channels, bulk_recordings.capture_source_channels),
            capture_session_id = COALESCE(excluded.capture_session_id, bulk_recordings.capture_session_id)",
        params![
            alignment.recording_id,
            alignment.project_slug,
            alignment.script,
            alignment.recorded_at,
            alignment.duration_ms,
            alignment.source_wav,
            alignment.bundle,
            now_ms,
            capture.device_manufacturer,
            capture.device_model,
            capture.os_version,
            capture.app_version,
            capture.input_route,
            capture.source_sample_rate_hz,
            capture.source_channels,
            capture.session_id,
            alignment.source_sha256,
            alignment.kind,
        ],
    )?;

    // Retire prior slices (rows) and prompts; keep old transcripts as history.
    tx.execute(
        "DELETE FROM slices WHERE recording_id = ?1",
        params![alignment.recording_id],
    )?;
    tx.execute(
        "DELETE FROM prompts WHERE recording_id = ?1",
        params![alignment.recording_id],
    )?;
    tx.execute(
        "UPDATE transcripts SET is_current = 0 WHERE recording_id = ?1",
        params![alignment.recording_id],
    )?;

    let next_version: i64 = tx
        .query_row(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM transcripts WHERE recording_id = ?1",
            params![alignment.recording_id],
            |row| row.get(0),
        )
        .optional()?
        .unwrap_or(1);

    tx.execute(
        "INSERT INTO transcripts (recording_id, version, text, whisper_url, is_current, created_at_ms)
         VALUES (?1, ?2, ?3, ?4, 1, ?5)",
        params![
            alignment.recording_id,
            next_version,
            alignment.transcript_text,
            alignment.whisper_url,
            now_ms,
        ],
    )?;
    let transcript_id = tx.last_insert_rowid();

    {
        let mut stmt = tx.prepare(
            "INSERT INTO transcript_words (transcript_id, ordinal, word, start_ms, end_ms, probability)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        )?;
        for (ordinal, word) in alignment.words.iter().enumerate() {
            stmt.execute(params![
                transcript_id,
                ordinal as i64,
                word.word,
                word.start_ms,
                word.end_ms,
                word.probability,
            ])?;
        }
    }

    {
        let mut stmt = tx.prepare(
            "INSERT INTO prompts (recording_id, ordinal, text) VALUES (?1, ?2, ?3)",
        )?;
        for (ordinal, text) in alignment.prompts.iter().enumerate() {
            stmt.execute(params![alignment.recording_id, ordinal as i64, text])?;
        }
    }

    {
        let mut stmt = tx.prepare(
            "INSERT INTO slices
                (id, recording_id, transcript_id, project_slug, label, category,
                 spoken_phrase, source_start_ms, source_end_ms, duration_ms,
                 avg_probability, word_count, wav_path, file_name, status, created_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, 'active', ?15)",
        )?;
        for slice in &alignment.slices {
            let duration_ms = (slice.source_end_ms - slice.source_start_ms).max(0);
            stmt.execute(params![
                slice.id,
                alignment.recording_id,
                transcript_id,
                alignment.project_slug,
                slice.label,
                slice.category,
                slice.spoken_phrase,
                slice.source_start_ms,
                slice.source_end_ms,
                duration_ms,
                slice.avg_probability,
                slice.word_count,
                slice.wav_path,
                slice.file_name,
                now_ms,
            ])?;
        }
    }

    tx.commit()?;
    Ok(next_version)
}

/// Record an imported single clip. Idempotent on clip id.
pub fn insert_clip(conn: &Connection, clip: &ClipRow, now_ms: i64) -> Result<bool, rusqlite::Error> {
    let changed = conn.execute(
        "INSERT OR IGNORE INTO clips
            (id, project_slug, label, category, spoken_phrase, wav_path, source_file, bundle, imported_at_ms)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            clip.id,
            clip.project_slug,
            clip.label,
            clip.category,
            clip.spoken_phrase,
            clip.wav_path,
            clip.source_file,
            clip.bundle,
            now_ms,
        ],
    )?;
    Ok(changed > 0)
}

/// All active slices for a project, ordered for the review screen.
pub fn review_slices(conn: &Connection, slug: &str) -> Result<Vec<ReviewSlice>, rusqlite::Error> {
    let mut stmt = conn.prepare(
        "SELECT id, label, spoken_phrase, recording_id, source_start_ms, source_end_ms,
                duration_ms, avg_probability, word_count, category, file_name
         FROM slices
         WHERE project_slug = ?1 AND status = 'active'
         ORDER BY recording_id DESC, source_start_ms ASC",
    )?;
    let rows = stmt.query_map(params![slug], |row| {
        Ok(ReviewSlice {
            id: row.get(0)?,
            label: row.get(1)?,
            spoken_phrase: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
            source_recording: row.get(3)?,
            source_start_ms: row.get(4)?,
            source_end_ms: row.get(5)?,
            duration_ms: row.get(6)?,
            avg_probability: row.get::<_, Option<f64>>(7)?.unwrap_or(0.0),
            word_count: row.get::<_, Option<i64>>(8)?.unwrap_or(0),
            category: row.get(9)?,
            file_name: row.get(10)?,
        })
    })?;
    rows.collect()
}

/// Every recording id in a project paired with its stored source-WAV SHA-256
/// (None for legacy rows recorded before checksums existed). Lets a device
/// decide which takes it can skip re-uploading.
pub fn recording_checksums(
    conn: &Connection,
    slug: &str,
) -> Result<Vec<(String, Option<String>)>, rusqlite::Error> {
    let mut stmt = conn.prepare(
        "SELECT id, source_sha256 FROM bulk_recordings WHERE project_slug = ?1 ORDER BY id",
    )?;
    let rows = stmt.query_map(params![slug], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
    })?;
    rows.collect()
}

/// Full reprocess metadata for every recording in a project, newest first.
pub fn recordings_for_reprocess(
    conn: &Connection,
    slug: &str,
) -> Result<Vec<RecordingMeta>, rusqlite::Error> {
    let mut stmt = conn.prepare(
        "SELECT id, project_slug, script, COALESCE(recorded_at, ''),
                COALESCE(duration_ms, 0), COALESCE(kind, 'mixed')
         FROM bulk_recordings WHERE project_slug = ?1 ORDER BY id DESC",
    )?;
    let rows = stmt.query_map(params![slug], |row| {
        Ok(RecordingMeta {
            id: row.get(0)?,
            project_slug: row.get(1)?,
            script: row.get(2)?,
            recorded_at: row.get(3)?,
            duration_ms: row.get(4)?,
            kind: row.get(5)?,
        })
    })?;
    rows.collect()
}

/// Reprocess metadata for a single recording, if it exists.
pub fn recording_meta(
    conn: &Connection,
    recording_id: &str,
) -> Result<Option<RecordingMeta>, rusqlite::Error> {
    conn.query_row(
        "SELECT id, project_slug, script, COALESCE(recorded_at, ''),
                COALESCE(duration_ms, 0), COALESCE(kind, 'mixed')
         FROM bulk_recordings WHERE id = ?1",
        params![recording_id],
        |row| {
            Ok(RecordingMeta {
                id: row.get(0)?,
                project_slug: row.get(1)?,
                script: row.get(2)?,
                recorded_at: row.get(3)?,
                duration_ms: row.get(4)?,
                kind: row.get(5)?,
            })
        },
    )
    .optional()
}

/// Per-recording metadata, active slice tallies, and capture provenance for
/// every recording the server holds for a wake word. This is what lets any
/// device list and manage recordings it never created itself (e.g. takes that
/// were synced from a different phone or tablet).
#[derive(Debug, Clone)]
pub struct RecordingDetail {
    pub id: String,
    pub kind: String,
    pub recorded_at: String,
    pub duration_ms: i64,
    pub positive_count: i64,
    pub negative_count: i64,
    pub background_count: i64,
    pub device_manufacturer: Option<String>,
    pub device_model: Option<String>,
    pub app_version: Option<String>,
    pub input_route: Option<String>,
    pub session_id: Option<String>,
}

/// Every recording for a wake word with its active slice counts and the device
/// that captured it, newest first. Background takes share this table and are
/// distinguished by the caller via the `background_` id prefix.
pub fn recording_details(
    conn: &Connection,
    slug: &str,
) -> Result<Vec<RecordingDetail>, rusqlite::Error> {
    let mut stmt = conn.prepare(
        "SELECT b.id, COALESCE(b.recorded_at, ''), COALESCE(b.duration_ms, 0),
                (SELECT COUNT(*) FROM slices s WHERE s.recording_id = b.id
                    AND s.status = 'active' AND s.category = 'positive'),
                (SELECT COUNT(*) FROM slices s WHERE s.recording_id = b.id
                    AND s.status = 'active' AND s.category IN ('negative', 'hard_negative')),
                (SELECT COUNT(*) FROM slices s WHERE s.recording_id = b.id
                    AND s.status = 'active' AND s.category = 'background'),
                b.capture_device_manufacturer, b.capture_device_model,
                b.capture_app_version, b.capture_input_route, b.capture_session_id,
                COALESCE(b.kind, 'mixed')
         FROM bulk_recordings b
         WHERE b.project_slug = ?1
         ORDER BY b.id DESC",
    )?;
    let rows = stmt.query_map(params![slug], |row| {
        Ok(RecordingDetail {
            id: row.get(0)?,
            recorded_at: row.get(1)?,
            duration_ms: row.get(2)?,
            positive_count: row.get(3)?,
            negative_count: row.get(4)?,
            background_count: row.get(5)?,
            device_manufacturer: row.get(6)?,
            device_model: row.get(7)?,
            app_version: row.get(8)?,
            input_route: row.get(9)?,
            session_id: row.get(10)?,
            kind: row.get(11)?,
        })
    })?;
    rows.collect()
}

/// A project's wake phrase and external id, if the project is known.
pub fn project_phrase(
    conn: &Connection,
    slug: &str,
) -> Result<Option<(String, Option<String>)>, rusqlite::Error> {
    conn.query_row(
        "SELECT phrase, external_id FROM projects WHERE slug = ?1",
        params![slug],
        |row| Ok((row.get(0)?, row.get::<_, Option<String>>(1)?)),
    )
    .optional()
}

/// One row of the training queue.
#[derive(Debug, Clone)]
pub struct QueueEntry {
    pub id: i64,
    pub slug: String,
    pub params_json: String,
    pub state: String,
    pub container_name: Option<String>,
    pub enqueued_at_ms: i64,
    pub started_at_ms: Option<i64>,
    pub finished_at_ms: Option<i64>,
}

fn queue_entry_from_row(row: &rusqlite::Row) -> Result<QueueEntry, rusqlite::Error> {
    Ok(QueueEntry {
        id: row.get(0)?,
        slug: row.get(1)?,
        params_json: row.get(2)?,
        state: row.get(3)?,
        container_name: row.get(4)?,
        enqueued_at_ms: row.get(5)?,
        started_at_ms: row.get(6)?,
        finished_at_ms: row.get(7)?,
    })
}

const QUEUE_COLS: &str =
    "id, slug, params_json, state, container_name, enqueued_at_ms, started_at_ms, finished_at_ms";

/// Append a training job to the queue and return its new id.
pub fn enqueue_training(
    conn: &Connection,
    slug: &str,
    params_json: &str,
    now_ms: i64,
) -> Result<i64, rusqlite::Error> {
    conn.execute(
        "INSERT INTO train_queue (slug, params_json, state, enqueued_at_ms)
         VALUES (?1, ?2, 'queued', ?3)",
        params![slug, params_json, now_ms],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Every queued or running job, oldest first — the live view of the pipeline.
pub fn active_queue(conn: &Connection) -> Result<Vec<QueueEntry>, rusqlite::Error> {
    let mut stmt = conn.prepare(&format!(
        "SELECT {QUEUE_COLS} FROM train_queue
         WHERE state IN ('queued', 'running') ORDER BY id ASC"
    ))?;
    let rows = stmt.query_map([], queue_entry_from_row)?;
    rows.collect()
}

/// Jobs currently marked running (the scheduler reconciles these against live
/// containers). Normally at most one.
pub fn running_queue(conn: &Connection) -> Result<Vec<QueueEntry>, rusqlite::Error> {
    let mut stmt = conn.prepare(&format!(
        "SELECT {QUEUE_COLS} FROM train_queue WHERE state = 'running' ORDER BY id ASC"
    ))?;
    let rows = stmt.query_map([], queue_entry_from_row)?;
    rows.collect()
}

/// The oldest queued job, if any.
pub fn next_queued(conn: &Connection) -> Result<Option<QueueEntry>, rusqlite::Error> {
    conn.query_row(
        &format!(
            "SELECT {QUEUE_COLS} FROM train_queue WHERE state = 'queued' ORDER BY id ASC LIMIT 1"
        ),
        [],
        queue_entry_from_row,
    )
    .optional()
}

/// Fetch a single queue entry by id.
pub fn queue_entry(conn: &Connection, id: i64) -> Result<Option<QueueEntry>, rusqlite::Error> {
    conn.query_row(
        &format!("SELECT {QUEUE_COLS} FROM train_queue WHERE id = ?1"),
        params![id],
        queue_entry_from_row,
    )
    .optional()
}

/// Mark a queued job as running and record the container launched for it.
pub fn mark_queue_running(
    conn: &Connection,
    id: i64,
    container_name: &str,
    now_ms: i64,
) -> Result<(), rusqlite::Error> {
    conn.execute(
        "UPDATE train_queue SET state = 'running', container_name = ?2, started_at_ms = ?3
         WHERE id = ?1",
        params![id, container_name, now_ms],
    )?;
    Ok(())
}

/// Move a job to a terminal state (succeeded/failed/canceled), stamping the
/// finish time.
pub fn finish_queue_entry(
    conn: &Connection,
    id: i64,
    state: &str,
    now_ms: i64,
) -> Result<(), rusqlite::Error> {
    conn.execute(
        "UPDATE train_queue SET state = ?2, finished_at_ms = ?3 WHERE id = ?1",
        params![id, state, now_ms],
    )?;
    Ok(())
}

/// The 1-based position of a job among the currently queued jobs (running counts
/// as 0). None when the job isn't queued/running.
pub fn queue_position(conn: &Connection, id: i64) -> Result<Option<i64>, rusqlite::Error> {
    conn.query_row(
        "SELECT
           CASE WHEN state = 'running' THEN 0
                WHEN state = 'queued' THEN
                  (SELECT COUNT(*) FROM train_queue q2
                     WHERE q2.state = 'queued' AND q2.id <= q1.id)
                ELSE NULL END
         FROM train_queue q1 WHERE id = ?1",
        params![id],
        |row| row.get::<_, Option<i64>>(0),
    )
    .optional()
    .map(|opt| opt.flatten())
}

/// The active (queued or running) entry for a slug, if any — used so the status
/// endpoint can report a pending run before its container exists.
pub fn active_entry_for_slug(
    conn: &Connection,
    slug: &str,
) -> Result<Option<QueueEntry>, rusqlite::Error> {
    conn.query_row(
        &format!(
            "SELECT {QUEUE_COLS} FROM train_queue
             WHERE slug = ?1 AND state IN ('queued', 'running') ORDER BY id ASC LIMIT 1"
        ),
        params![slug],
        queue_entry_from_row,
    )
    .optional()
}

/// Cancel every queued/running job for a slug, returning the affected entries so
/// the caller can kill any live containers. Rows are marked canceled here.
pub fn cancel_slug_jobs(
    conn: &Connection,
    slug: &str,
    now_ms: i64,
) -> Result<Vec<QueueEntry>, rusqlite::Error> {
    let entries = {
        let mut stmt = conn.prepare(&format!(
            "SELECT {QUEUE_COLS} FROM train_queue
             WHERE slug = ?1 AND state IN ('queued', 'running')"
        ))?;
        let rows = stmt.query_map(params![slug], queue_entry_from_row)?;
        rows.collect::<Result<Vec<_>, _>>()?
    };
    conn.execute(
        "UPDATE train_queue SET state = 'canceled', finished_at_ms = ?2
         WHERE slug = ?1 AND state IN ('queued', 'running')",
        params![slug, now_ms],
    )?;
    Ok(entries)
}

/// Cancel a single queue entry by id if it is still queued/running, returning it.
pub fn cancel_queue_entry(
    conn: &Connection,
    id: i64,
    now_ms: i64,
) -> Result<Option<QueueEntry>, rusqlite::Error> {
    let entry = queue_entry(conn, id)?;
    let Some(entry) = entry else { return Ok(None) };
    if !matches!(entry.state.as_str(), "queued" | "running") {
        return Ok(None);
    }
    conn.execute(
        "UPDATE train_queue SET state = 'canceled', finished_at_ms = ?2 WHERE id = ?1",
        params![id, now_ms],
    )?;
    Ok(Some(entry))
}

/// Record a completed (terminal) training run. Idempotent on (slug, started_at),
/// so it is safe to call on every status poll — repeats are ignored.
pub fn record_training_run(
    conn: &Connection,
    slug: &str,
    steps: i64,
    model_size: Option<&str>,
    personal: bool,
    started_at: &str,
    finished_at: Option<&str>,
    duration_ms: Option<i64>,
    state: &str,
) -> Result<bool, rusqlite::Error> {
    let changed = conn.execute(
        "INSERT OR IGNORE INTO training_runs
           (slug, steps, model_size, personal, started_at, finished_at, duration_ms, state)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            slug,
            steps,
            model_size,
            personal as i64,
            started_at,
            finished_at,
            duration_ms,
            state
        ],
    )?;
    Ok(changed > 0)
}

/// A trained model artifact to record, gathered from a run's manifest and the
/// terminal training status. Every optional field tolerates an older run whose
/// manifest predates that field.
#[derive(Debug, Clone, Default)]
pub struct ModelRecord {
    pub slug: String,
    pub phrase: String,
    pub run_id: String,
    pub onnx_path: String,
    pub onnx_sha256: Option<String>,
    pub pt_sha256: Option<String>,
    pub steps: Option<i64>,
    pub model_size: Option<String>,
    pub personal: bool,
    pub git_commit: Option<String>,
    pub metrics: Option<String>,
    pub started_at: Option<String>,
    pub finished_at: Option<String>,
    // Version 6 provenance: every knob and the real vs synthetic data counts,
    // plus the full manifest verbatim in `params_json`.
    pub model_type: Option<String>,
    pub token_type: Option<String>,
    pub positive_boost: Option<i64>,
    pub n_samples: Option<i64>,
    pub n_samples_val: Option<i64>,
    pub positive_per_batch: Option<i64>,
    pub target_fp_per_hour: Option<f64>,
    pub context_fix: Option<bool>,
    pub real_positive: Option<i64>,
    pub real_negative: Option<i64>,
    pub real_background: Option<i64>,
    pub trainer_image: Option<String>,
    pub onnx_bytes: Option<i64>,
    pub eval: Option<String>,
    pub params_json: Option<String>,
}

/// Record a versioned trained model. Ensures the wake word exists, inserts the
/// model row (idempotent on `(slug, run_id)` so it is safe to call on every
/// status poll), and when `make_current` is set flips this run to the sole
/// current model for the wake word. Returns true if a new model row was added.
pub fn record_model(
    conn: &Connection,
    model: &ModelRecord,
    now_ms: i64,
    make_current: bool,
) -> Result<bool, rusqlite::Error> {
    upsert_project(conn, &model.slug, &model.phrase, None, now_ms)?;
    let changed = conn.execute(
        "INSERT OR IGNORE INTO models
            (slug, run_id, onnx_path, onnx_sha256, pt_sha256, steps, model_size,
             personal, git_commit, metrics, started_at, finished_at, created_at_ms,
             is_current, model_type, token_type, positive_boost, n_samples,
             n_samples_val, positive_per_batch, target_fp_per_hour, context_fix,
             real_positive, real_negative, real_background, trainer_image,
             onnx_bytes, eval, params_json)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, 0,
             ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26,
             ?27, ?28)",
        params![
            model.slug,
            model.run_id,
            model.onnx_path,
            model.onnx_sha256,
            model.pt_sha256,
            model.steps,
            model.model_size,
            model.personal as i64,
            model.git_commit,
            model.metrics,
            model.started_at,
            model.finished_at,
            now_ms,
            model.model_type,
            model.token_type,
            model.positive_boost,
            model.n_samples,
            model.n_samples_val,
            model.positive_per_batch,
            model.target_fp_per_hour,
            model.context_fix.map(|b| b as i64),
            model.real_positive,
            model.real_negative,
            model.real_background,
            model.trainer_image,
            model.onnx_bytes,
            model.eval,
            model.params_json,
        ],
    )?;
    if make_current {
        conn.execute(
            "UPDATE models SET is_current = CASE WHEN run_id = ?2 THEN 1 ELSE 0 END
             WHERE slug = ?1",
            params![model.slug, model.run_id],
        )?;
    }
    Ok(changed > 0)
}

/// Every recorded model with its full training provenance, newest first, each
/// row rendered as a JSON object string. `metrics`, `eval`, and `params_json`
/// are re-parsed via `json(...)` so they nest as objects rather than escaped
/// strings; a malformed blob falls back to SQL NULL. The caller parses each
/// string into a value and assembles the array, so this stays serde-free.
pub fn list_model_records(conn: &Connection) -> Result<Vec<String>, rusqlite::Error> {
    let mut stmt = conn.prepare(
        "SELECT json_object(
            'slug', slug,
            'run_id', run_id,
            'is_current', is_current,
            'onnx_path', onnx_path,
            'onnx_sha256', onnx_sha256,
            'pt_sha256', pt_sha256,
            'onnx_bytes', onnx_bytes,
            'git_commit', git_commit,
            'trainer_image', trainer_image,
            'steps', steps,
            'model_type', model_type,
            'model_size', model_size,
            'personal', personal,
            'positive_boost', positive_boost,
            'n_samples', n_samples,
            'n_samples_val', n_samples_val,
            'positive_per_batch', positive_per_batch,
            'target_fp_per_hour', target_fp_per_hour,
            'token_type', token_type,
            'context_fix', context_fix,
            'real_positive', real_positive,
            'real_negative', real_negative,
            'real_background', real_background,
            'started_at', started_at,
            'finished_at', finished_at,
            'created_at_ms', created_at_ms,
            'metrics', json(metrics),
            'eval', json(eval),
            'params', json(params_json)
         )
         FROM models
         ORDER BY created_at_ms DESC, id DESC",
    )?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    rows.collect()
}

/// Average milliseconds per training step from past *successful* runs, preferring
/// the same model size and falling back to all sizes when that size has no
/// history. Returns (avg_ms_per_step, run_count), or None when there is no data.
pub fn avg_ms_per_step(
    conn: &Connection,
    model_size: Option<&str>,
) -> Result<Option<(f64, i64)>, rusqlite::Error> {
    if let Some(size) = model_size {
        if let Some(result) = avg_ms_per_step_where(conn, Some(size))? {
            return Ok(Some(result));
        }
    }
    avg_ms_per_step_where(conn, None)
}

fn avg_ms_per_step_where(
    conn: &Connection,
    model_size: Option<&str>,
) -> Result<Option<(f64, i64)>, rusqlite::Error> {
    let row: (Option<f64>, i64) = match model_size {
        Some(size) => conn.query_row(
            "SELECT AVG(CAST(duration_ms AS REAL) / steps), COUNT(*)
               FROM training_runs
              WHERE state = 'succeeded' AND duration_ms IS NOT NULL AND steps > 0
                AND model_size = ?1",
            params![size],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?,
        None => conn.query_row(
            "SELECT AVG(CAST(duration_ms AS REAL) / steps), COUNT(*)
               FROM training_runs
              WHERE state = 'succeeded' AND duration_ms IS NOT NULL AND steps > 0",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?,
    };
    match row {
        (Some(avg), count) if count > 0 => Ok(Some((avg, count))),
        _ => Ok(None),
    }
}

/// Delete a recording and everything derived from it (transcripts, words,
/// prompts, slices cascade). WAV files on disk are removed by the caller.
pub fn delete_recording(conn: &Connection, recording_id: &str) -> Result<bool, rusqlite::Error> {
    // score_curves has no FK to bulk_recordings (test takes score too), so clear
    // any cached curves for this recording explicitly when it goes away.
    conn.execute(
        "DELETE FROM score_curves WHERE recording_id = ?1",
        params![recording_id],
    )?;
    let changed = conn.execute(
        "DELETE FROM bulk_recordings WHERE id = ?1",
        params![recording_id],
    )?;
    Ok(changed > 0)
}

/// A cached rolling-score curve for a recording, valid only if the stored model
/// fingerprint still matches the current model. Returns `(duration_ms, times_ms,
/// scores)`. A fingerprint mismatch is treated as a miss (stale model).
pub fn cached_score_curve(
    conn: &Connection,
    recording_id: &str,
    mode: &str,
    step_ms: u64,
    keep_ms: u64,
    model_fp: &str,
) -> Result<Option<(f64, Vec<f64>, Vec<f64>)>, rusqlite::Error> {
    let row = conn
        .query_row(
            "SELECT duration_ms, times_ms, scores FROM score_curves
             WHERE recording_id = ?1 AND mode = ?2 AND step_ms = ?3
               AND keep_ms = ?4 AND model_fp = ?5",
            params![recording_id, mode, step_ms as i64, keep_ms as i64, model_fp],
            |row| {
                let duration_ms: f64 = row.get(0)?;
                let times_json: String = row.get(1)?;
                let scores_json: String = row.get(2)?;
                Ok((duration_ms, times_json, scores_json))
            },
        )
        .optional()?;
    let Some((duration_ms, times_json, scores_json)) = row else {
        return Ok(None);
    };
    // Corrupt cached JSON should behave as a miss, not blow up the request.
    let (Ok(times_ms), Ok(scores)) = (
        serde_json::from_str::<Vec<f64>>(&times_json),
        serde_json::from_str::<Vec<f64>>(&scores_json),
    ) else {
        return Ok(None);
    };
    Ok(Some((duration_ms, times_ms, scores)))
}

/// Cache a freshly computed score curve, replacing any prior entry for the same
/// recording/mode/params (e.g. after a retrain changed the fingerprint).
#[allow(clippy::too_many_arguments)]
pub fn store_score_curve(
    conn: &Connection,
    recording_id: &str,
    mode: &str,
    step_ms: u64,
    keep_ms: u64,
    model_fp: &str,
    duration_ms: f64,
    times_ms: &[f64],
    scores: &[f64],
    now_ms: i64,
) -> Result<(), rusqlite::Error> {
    let times_json = serde_json::to_string(times_ms).unwrap_or_else(|_| "[]".to_string());
    let scores_json = serde_json::to_string(scores).unwrap_or_else(|_| "[]".to_string());
    conn.execute(
        "INSERT OR REPLACE INTO score_curves
         (recording_id, mode, step_ms, keep_ms, model_fp, duration_ms, times_ms, scores, created_at_ms)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            recording_id,
            mode,
            step_ms as i64,
            keep_ms as i64,
            model_fp,
            duration_ms,
            times_json,
            scores_json,
            now_ms
        ],
    )?;
    Ok(())
}

/// The script for a recording, if known.
pub fn recording_script(
    conn: &Connection,
    recording_id: &str,
) -> Result<Option<String>, rusqlite::Error> {
    conn.query_row(
        "SELECT script FROM bulk_recordings WHERE id = ?1",
        params![recording_id],
        |row| row.get(0),
    )
    .optional()
}

/// Words of a recording's current transcript, in order.
pub fn current_transcript_words(
    conn: &Connection,
    recording_id: &str,
) -> Result<Vec<WordRow>, rusqlite::Error> {
    let mut stmt = conn.prepare(
        "SELECT w.word, w.start_ms, w.end_ms, w.probability
         FROM transcript_words w
         JOIN transcripts t ON t.id = w.transcript_id
         WHERE t.recording_id = ?1 AND t.is_current = 1
         ORDER BY w.ordinal ASC",
    )?;
    let rows = stmt.query_map(params![recording_id], |row| {
        Ok(WordRow {
            word: row.get(0)?,
            start_ms: row.get(1)?,
            end_ms: row.get(2)?,
            probability: row.get::<_, Option<f64>>(3)?.unwrap_or(0.0),
        })
    })?;
    rows.collect()
}

/// Active slice cuts for a recording's alignment timeline.
pub fn recording_cuts(
    conn: &Connection,
    recording_id: &str,
) -> Result<Vec<CutRow>, rusqlite::Error> {
    let mut stmt = conn.prepare(
        "SELECT id, label, source_start_ms, source_end_ms
         FROM slices
         WHERE recording_id = ?1 AND status = 'active'
         ORDER BY source_start_ms ASC",
    )?;
    let rows = stmt.query_map(params![recording_id], |row| {
        Ok(CutRow {
            id: row.get(0)?,
            label: row.get(1)?,
            start_ms: row.get(2)?,
            end_ms: row.get(3)?,
        })
    })?;
    rows.collect()
}

/// Project summaries with active bulk-slice counts.
pub fn project_summaries(conn: &Connection) -> Result<Vec<ProjectRow>, rusqlite::Error> {
    let mut stmt = conn.prepare(
        "SELECT p.slug, p.phrase, COALESCE(p.external_id, p.slug), p.created_at_ms,
                (SELECT COUNT(*) FROM slices s WHERE s.project_slug = p.slug AND s.status = 'active'),
                (SELECT COUNT(*) FROM slices s WHERE s.project_slug = p.slug AND s.status = 'active' AND s.category = 'positive'),
                (SELECT COUNT(*) FROM slices s WHERE s.project_slug = p.slug AND s.status = 'active' AND s.category IN ('negative', 'hard_negative')),
                (SELECT COUNT(*) FROM slices s WHERE s.project_slug = p.slug AND s.status = 'active' AND s.category = 'background'),
                (SELECT COUNT(*) FROM slices s WHERE s.project_slug = p.slug AND s.status = 'active' AND s.category = 'negative')
         FROM projects p
         ORDER BY p.slug ASC",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(ProjectRow {
            slug: row.get(0)?,
            phrase: row.get(1)?,
            external_id: row.get(2)?,
            created_at_ms: row.get(3)?,
            bulk_slice_count: row.get(4)?,
            positive_count: row.get(5)?,
            negative_count: row.get(6)?,
            background_count: row.get(7)?,
            poolable_negative_count: row.get(8)?,
        })
    })?;
    rows.collect()
}

/// Mark a slice deleted by its file name. Returns true if a row was updated.
pub fn delete_slice_by_file(
    conn: &Connection,
    slug: &str,
    category: &str,
    file_name: &str,
) -> Result<bool, rusqlite::Error> {
    let changed = conn.execute(
        "UPDATE slices SET status = 'deleted'
         WHERE project_slug = ?1 AND category = ?2 AND file_name = ?3 AND status = 'active'",
        params![slug, category, file_name],
    )?;
    Ok(changed > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_conn() -> Connection {
        let conn = Connection::open_in_memory().expect("open in-memory db");
        migrate(&conn).expect("migrate");
        conn
    }

    fn insert_recording(conn: &Connection, id: &str, slug: &str, duration_ms: i64, device: &str) {
        conn.execute(
            "INSERT INTO bulk_recordings
                (id, project_slug, script, recorded_at, duration_ms, source_wav,
                 imported_at_ms, capture_device_model)
             VALUES (?1, ?2, '', '2026-07-18T00:00:00Z', ?3, '/x.wav', 0, ?4)",
            params![id, slug, duration_ms, device],
        )
        .expect("insert recording");
    }

    fn insert_slice(conn: &Connection, id: &str, rec: &str, slug: &str, category: &str, status: &str) {
        conn.execute(
            "INSERT INTO slices
                (id, recording_id, project_slug, label, category, source_start_ms,
                 source_end_ms, duration_ms, wav_path, file_name, status, created_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?4, 0, 100, 100, '/s.wav', ?1, ?5, 0)",
            params![id, rec, slug, category, status],
        )
        .expect("insert slice");
    }

    #[test]
    fn recording_details_reports_counts_device_and_background_flag() {
        let conn = test_conn();
        conn.execute(
            "INSERT INTO projects (slug, phrase, created_at_ms) VALUES ('all_set', 'all set', 0)",
            [],
        )
        .unwrap();
        insert_recording(&conn, "bulk_100_a", "all_set", 43000, "Pixel 8a");
        insert_recording(&conn, "background_200_b", "all_set", 60000, "SM-P620");
        // Two active positives, one active negative, one deleted positive (ignored).
        insert_slice(&conn, "s1", "bulk_100_a", "all_set", "positive", "active");
        insert_slice(&conn, "s2", "bulk_100_a", "all_set", "positive", "active");
        insert_slice(&conn, "s3", "bulk_100_a", "all_set", "negative", "active");
        insert_slice(&conn, "s4", "bulk_100_a", "all_set", "positive", "deleted");
        insert_slice(&conn, "s5", "background_200_b", "all_set", "background", "active");

        let details = recording_details(&conn, "all_set").expect("details");
        assert_eq!(details.len(), 2);
        // Ordered by id DESC: "bulk_..." sorts after "background_...".
        let bulk = &details[0];
        assert_eq!(bulk.id, "bulk_100_a");
        assert_eq!(bulk.positive_count, 2);
        assert_eq!(bulk.negative_count, 1);
        assert_eq!(bulk.background_count, 0);
        assert_eq!(bulk.duration_ms, 43000);
        assert_eq!(bulk.device_model.as_deref(), Some("Pixel 8a"));

        let background = &details[1];
        assert_eq!(background.id, "background_200_b");
        assert_eq!(background.background_count, 1);
        assert_eq!(background.device_model.as_deref(), Some("SM-P620"));
    }

    #[test]
    fn record_model_versions_and_tracks_current() {
        let conn = test_conn();
        let mut m = ModelRecord {
            slug: "all_set".to_string(),
            phrase: "all set".to_string(),
            run_id: "20260719T120000Z".to_string(),
            onnx_path: "runs/20260719T120000Z/all_set.onnx".to_string(),
            onnx_sha256: Some("aaa".to_string()),
            steps: Some(50_000),
            model_size: Some("medium".to_string()),
            ..Default::default()
        };
        assert!(record_model(&conn, &m, 0, true).expect("first model"));
        // Idempotent on (slug, run_id): a repeat poll adds nothing.
        assert!(!record_model(&conn, &m, 1, true).expect("repeat"));

        // A second run becomes current and demotes the first.
        m.run_id = "20260719T130000Z".to_string();
        m.onnx_sha256 = Some("bbb".to_string());
        assert!(record_model(&conn, &m, 2, true).expect("second model"));

        let current: String = conn
            .query_row(
                "SELECT run_id FROM models WHERE slug = 'all_set' AND is_current = 1",
                [],
                |r| r.get(0),
            )
            .expect("one current");
        assert_eq!(current, "20260719T130000Z");
        let total: i64 = conn
            .query_row("SELECT COUNT(*) FROM models WHERE slug = 'all_set'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(total, 2);
        // The project row was created by the model insert's foreign-key upsert.
        let phrase = project_phrase(&conn, "all_set").expect("q").expect("row");
        assert_eq!(phrase.0, "all set");
    }

    #[test]
    fn recording_checksums_round_trips_and_preserves_on_reprocess() {
        let conn = Connection::open_in_memory().expect("open in-memory db");
        migrate(&conn).expect("migrate");
        let mut conn = conn;

        let mut alignment = RecordingAlignment {
            recording_id: "bulk_1".to_string(),
            project_slug: "all_set".to_string(),
            phrase: "all set".to_string(),
            external_id: None,
            script: "all set".to_string(),
            kind: "positive".to_string(),
            recorded_at: "2026-07-18T00:00:00Z".to_string(),
            duration_ms: 5000,
            source_wav: "/x.wav".to_string(),
            source_sha256: Some("abc123".to_string()),
            bundle: None,
            transcript_text: String::new(),
            whisper_url: None,
            words: Vec::new(),
            prompts: Vec::new(),
            slices: Vec::new(),
            capture: CaptureMeta::default(),
        };
        store_recording_alignment(&mut conn, &alignment, 0).expect("store");

        let sums = recording_checksums(&conn, "all_set").expect("checksums");
        assert_eq!(sums, vec![("bulk_1".to_string(), Some("abc123".to_string()))]);

        // Reprocess carries no checksum; the prior one must survive.
        alignment.source_sha256 = None;
        store_recording_alignment(&mut conn, &alignment, 1).expect("reprocess store");
        let sums = recording_checksums(&conn, "all_set").expect("checksums");
        assert_eq!(sums, vec![("bulk_1".to_string(), Some("abc123".to_string()))]);
    }
}
