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
const SCHEMA_VERSION: i64 = 2;

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
    pub recorded_at: String,
    pub duration_ms: i64,
    pub source_wav: String,
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
}

/// Everything a reprocess pass needs about a stored recording to re-run
/// alignment from its already-saved source WAV, without a fresh upload.
#[derive(Debug, Clone)]
pub struct RecordingMeta {
    pub id: String,
    pub project_slug: String,
    pub script: String,
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
    ] {
        if !column_exists(conn, "bulk_recordings", name)? {
            conn.execute(
                &format!("ALTER TABLE bulk_recordings ADD COLUMN {name} {decl}"),
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
             capture_source_channels, capture_session_id)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)
         ON CONFLICT(id) DO UPDATE SET
            project_slug = excluded.project_slug,
            script = excluded.script,
            recorded_at = excluded.recorded_at,
            duration_ms = excluded.duration_ms,
            source_wav = excluded.source_wav,
            bundle = excluded.bundle,
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

/// Distinct recording ids that currently have any slices or a recording row.
pub fn recording_ids(conn: &Connection, slug: &str) -> Result<Vec<String>, rusqlite::Error> {
    let mut stmt = conn.prepare(
        "SELECT id FROM bulk_recordings WHERE project_slug = ?1 ORDER BY id",
    )?;
    let rows = stmt.query_map(params![slug], |row| row.get::<_, String>(0))?;
    rows.collect()
}

/// Full reprocess metadata for every recording in a project, newest first.
pub fn recordings_for_reprocess(
    conn: &Connection,
    slug: &str,
) -> Result<Vec<RecordingMeta>, rusqlite::Error> {
    let mut stmt = conn.prepare(
        "SELECT id, project_slug, script, COALESCE(recorded_at, ''),
                COALESCE(duration_ms, 0)
         FROM bulk_recordings WHERE project_slug = ?1 ORDER BY id DESC",
    )?;
    let rows = stmt.query_map(params![slug], |row| {
        Ok(RecordingMeta {
            id: row.get(0)?,
            project_slug: row.get(1)?,
            script: row.get(2)?,
            recorded_at: row.get(3)?,
            duration_ms: row.get(4)?,
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
                COALESCE(duration_ms, 0)
         FROM bulk_recordings WHERE id = ?1",
        params![recording_id],
        |row| {
            Ok(RecordingMeta {
                id: row.get(0)?,
                project_slug: row.get(1)?,
                script: row.get(2)?,
                recorded_at: row.get(3)?,
                duration_ms: row.get(4)?,
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
                    AND s.status = 'active' AND s.category = 'negative'),
                (SELECT COUNT(*) FROM slices s WHERE s.recording_id = b.id
                    AND s.status = 'active' AND s.category = 'background'),
                b.capture_device_manufacturer, b.capture_device_model,
                b.capture_app_version, b.capture_input_route, b.capture_session_id
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

/// Delete a recording and everything derived from it (transcripts, words,
/// prompts, slices cascade). WAV files on disk are removed by the caller.
pub fn delete_recording(conn: &Connection, recording_id: &str) -> Result<bool, rusqlite::Error> {
    let changed = conn.execute(
        "DELETE FROM bulk_recordings WHERE id = ?1",
        params![recording_id],
    )?;
    Ok(changed > 0)
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
                (SELECT COUNT(*) FROM slices s WHERE s.project_slug = p.slug AND s.status = 'active' AND s.category = 'negative'),
                (SELECT COUNT(*) FROM slices s WHERE s.project_slug = p.slug AND s.status = 'active' AND s.category = 'background')
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
}
