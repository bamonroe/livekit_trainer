use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use chrono::Utc;
use hound::WavReader;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    env,
    fs::{self, File, OpenOptions},
    io::{self, Cursor, Write},
    net::SocketAddr,
    path::{Component, Path, PathBuf},
    sync::Arc,
};
use zip::ZipArchive;

#[derive(Clone)]
struct AppState {
    data_root: Arc<PathBuf>,
    incoming_root: Arc<PathBuf>,
}

#[derive(Debug, Serialize)]
struct SyncResponse {
    status: &'static str,
    archive: String,
    wake_word_slug: String,
    clip_count: usize,
    imported_count: usize,
    validate_output: String,
    import_output: String,
    warnings: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    schema_version: u64,
    wake_word: WakeWord,
    clips: Vec<Clip>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct WakeWord {
    slug: String,
    #[serde(flatten)]
    extra: Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Clip {
    id: String,
    file: String,
    label: String,
    #[serde(default)]
    spoken_phrase: String,
    #[serde(flatten)]
    extra: Value,
}

#[tokio::main]
async fn main() {
    let bind_addr = env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:8765".to_string());
    let data_root = env::var("DATA_ROOT").unwrap_or_else(|_| "/data/real".to_string());
    let incoming_root =
        env::var("INCOMING_ROOT").unwrap_or_else(|_| "/incoming/bundles".to_string());

    let state = AppState {
        data_root: Arc::new(PathBuf::from(data_root)),
        incoming_root: Arc::new(PathBuf::from(incoming_root)),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/sync", post(sync))
        .layer(DefaultBodyLimit::max(512 * 1024 * 1024))
        .with_state(state);

    let addr: SocketAddr = bind_addr.parse().expect("invalid BIND_ADDR");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("bind failed");
    println!("listening on http://{addr}");
    axum::serve(listener, app).await.expect("server failed");
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn sync(State(state): State<AppState>, body: Bytes) -> Result<Json<SyncResponse>, AppError> {
    if body.is_empty() {
        return Err(AppError::bad_request("empty upload"));
    }

    fs::create_dir_all(&*state.incoming_root)?;
    let archive = state
        .incoming_root
        .join(format!("bundle_{}.zip", Utc::now().timestamp_millis()));
    fs::write(&archive, &body)?;

    let extract_root = env::temp_dir().join(format!(
        "livekit_trainer_bundle_{}",
        Utc::now().timestamp_millis()
    ));
    fs::create_dir_all(&extract_root)?;

    let result = (|| {
        extract_zip(&body, &extract_root)?;
        let manifest = read_manifest(&extract_root)?;
        validate_manifest(&manifest)?;
        let warnings = validate_wavs(&extract_root, &manifest)?;
        let imported_count = import_bundle(&extract_root, &manifest, &state.data_root)?;
        let archive_name = archive
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("bundle.zip");
        let import_output = format!(
            "Imported {imported_count} clips into data/real/{}",
            manifest.wake_word.slug
        );
        let validate_output = validation_summary(&warnings, manifest.clips.len());
        Ok(SyncResponse {
            status: "imported",
            archive: format!("incoming/bundles/{archive_name}"),
            wake_word_slug: manifest.wake_word.slug,
            clip_count: manifest.clips.len(),
            imported_count,
            validate_output,
            import_output,
            warnings,
        })
    })();

    let _ = fs::remove_dir_all(&extract_root);
    result.map(Json)
}

fn extract_zip(bytes: &[u8], dest: &Path) -> Result<(), AppError> {
    let reader = Cursor::new(bytes);
    let mut archive = ZipArchive::new(reader)
        .map_err(|error| AppError::bad_request(format!("invalid zip: {error}")))?;

    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .map_err(|error| AppError::bad_request(format!("invalid zip entry: {error}")))?;
        let name = entry.name();
        let output = safe_join(dest, name)?;
        if entry.is_dir() {
            fs::create_dir_all(&output)?;
            continue;
        }
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = File::create(output)?;
        io::copy(&mut entry, &mut file)?;
    }
    Ok(())
}

fn read_manifest(bundle: &Path) -> Result<Manifest, AppError> {
    let manifest_path = bundle.join("manifest.json");
    let contents = fs::read_to_string(&manifest_path)
        .map_err(|error| AppError::bad_request(format!("missing manifest: {error}")))?;
    serde_json::from_str(&contents)
        .map_err(|error| AppError::bad_request(format!("invalid manifest JSON: {error}")))
}

fn validate_manifest(manifest: &Manifest) -> Result<(), AppError> {
    if manifest.schema_version != 1 {
        return Err(AppError::bad_request(format!(
            "unsupported schema_version: {}",
            manifest.schema_version
        )));
    }
    if !is_safe_slug(&manifest.wake_word.slug) {
        return Err(AppError::bad_request(format!(
            "unsafe wake_word.slug: {}",
            manifest.wake_word.slug
        )));
    }
    for (index, clip) in manifest.clips.iter().enumerate() {
        if clip.id.is_empty() {
            return Err(AppError::bad_request(format!("clip {index} missing id")));
        }
        if clip.file.is_empty() {
            return Err(AppError::bad_request(format!("clip {index} missing file")));
        }
        if category_for_label(&clip.label).is_none() {
            return Err(AppError::bad_request(format!(
                "clip {index} has unknown label {}",
                clip.label
            )));
        }
    }
    Ok(())
}

fn validate_wavs(bundle: &Path, manifest: &Manifest) -> Result<Vec<String>, AppError> {
    let mut warnings = Vec::new();
    for clip in &manifest.clips {
        let path = safe_join(bundle, &clip.file)?;
        let mut reader = WavReader::open(&path).map_err(|error| {
            AppError::bad_request(format!("{}: cannot read WAV: {error}", clip.id))
        })?;
        let spec = reader.spec();
        let duration = reader.duration() as f64 / spec.sample_rate as f64;
        if spec.channels != 1 {
            warnings.push(format!("{}: expected mono, got {} channels", clip.id, spec.channels));
        }
        if spec.bits_per_sample != 16 {
            warnings.push(format!(
                "{}: expected 16-bit PCM, got {} bits",
                clip.id, spec.bits_per_sample
            ));
        }
        if spec.sample_rate != 16_000 {
            warnings.push(format!(
                "{}: expected 16000 Hz, got {}",
                clip.id, spec.sample_rate
            ));
        }
        if duration <= 0.0 {
            warnings.push(format!("{}: zero duration", clip.id));
        }
        if duration > 5.0 {
            warnings.push(format!("{}: long clip {:.2}s", clip.id, duration));
        }

        let mut peak = 0i32;
        let mut total = 0f64;
        let mut count = 0f64;
        for sample in reader.samples::<i16>() {
            let sample =
                sample.map_err(|error| AppError::bad_request(format!("bad WAV sample: {error}")))?;
            let sample_i32 = i32::from(sample);
            peak = peak.max(sample_i32.abs());
            total += f64::from(sample) * f64::from(sample);
            count += 1.0;
        }
        let rms = if count > 0.0 { (total / count).sqrt() } else { 0.0 };
        if peak >= 32_760 {
            warnings.push(format!("{}: possible clipping, peak={peak}", clip.id));
        }
        if rms < 50.0 && duration > 0.0 {
            warnings.push(format!("{}: very quiet audio, rms={rms:.1}", clip.id));
        }
    }
    Ok(warnings)
}

fn import_bundle(bundle: &Path, manifest: &Manifest, data_root: &Path) -> Result<usize, AppError> {
    let dest_root = data_root.join(&manifest.wake_word.slug);
    fs::create_dir_all(&dest_root)?;
    let metadata_path = dest_root.join("metadata.jsonl");
    let mut metadata = OpenOptions::new()
        .create(true)
        .append(true)
        .open(metadata_path)?;

    let mut imported = 0;
    for clip in &manifest.clips {
        let category = category_for_label(&clip.label).expect("validated label");
        let src = safe_join(bundle, &clip.file)?;
        if !src.is_file() {
            return Err(AppError::bad_request(format!(
                "missing clip file: {}",
                clip.file
            )));
        }

        let dest_dir = dest_root.join(category);
        fs::create_dir_all(&dest_dir)?;
        let phrase = if clip.spoken_phrase.is_empty() {
            clip.label.as_str()
        } else {
            clip.spoken_phrase.as_str()
        };
        let dest = dest_dir.join(format!("{}_{}.wav", clip.id, safe_filename(phrase)));
        if dest.exists() {
            continue;
        }

        fs::copy(&src, &dest)?;
        let record = json!({
            "source_bundle": "server_sync",
            "source_file": clip.file,
            "imported_file": dest.to_string_lossy(),
            "livekit_category": category,
            "original_label": clip.label,
            "wake_word": manifest.wake_word,
            "clip": clip,
        });
        writeln!(metadata, "{}", serde_json::to_string(&record).expect("metadata JSON"))?;
        imported += 1;
    }
    Ok(imported)
}

fn safe_join(root: &Path, relative: &str) -> Result<PathBuf, AppError> {
    let path = Path::new(relative);
    if path.is_absolute() {
        return Err(AppError::bad_request(format!(
            "path escapes bundle: {relative}"
        )));
    }
    for component in path.components() {
        if matches!(component, Component::ParentDir | Component::Prefix(_)) {
            return Err(AppError::bad_request(format!(
                "path escapes bundle: {relative}"
            )));
        }
    }
    Ok(root.join(path))
}

fn category_for_label(label: &str) -> Option<&'static str> {
    match label {
        "positive" | "false_negative" => Some("positive"),
        "negative" | "hard_negative" | "false_positive" => Some("negative"),
        "background" => Some("background"),
        _ => None,
    }
}

fn is_safe_slug(slug: &str) -> bool {
    let mut chars = slug.chars();
    match chars.next() {
        Some(ch) if ch.is_ascii_lowercase() || ch.is_ascii_digit() => {}
        _ => return false,
    }
    chars.all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
}

fn safe_filename(value: &str) -> String {
    let mut output = String::new();
    let mut last_dash = false;
    for ch in value.trim().to_ascii_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            output.push(ch);
            last_dash = false;
        } else if !last_dash {
            output.push('-');
            last_dash = true;
        }
    }
    let cleaned = output.trim_matches('-').to_string();
    if cleaned.is_empty() {
        "clip".to_string()
    } else {
        cleaned
    }
}

fn validation_summary(warnings: &[String], clip_count: usize) -> String {
    if warnings.is_empty() {
        format!("Validated {clip_count} clips")
    } else {
        let mut output = String::from("Warnings:");
        for warning in warnings {
            output.push_str("\n- ");
            output.push_str(warning);
        }
        output.push_str(&format!("\nValidated {clip_count} clips"));
        output
    }
}

#[derive(Debug)]
struct AppError {
    status: StatusCode,
    message: String,
}

impl AppError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }
}

impl From<io::Error> for AppError {
    fn from(error: io::Error) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: error.to_string(),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = Json(json!({"error": self.message}));
        (self.status, body).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_labels_to_categories() {
        assert_eq!(category_for_label("positive"), Some("positive"));
        assert_eq!(category_for_label("false_negative"), Some("positive"));
        assert_eq!(category_for_label("hard_negative"), Some("negative"));
        assert_eq!(category_for_label("false_positive"), Some("negative"));
        assert_eq!(category_for_label("background"), Some("background"));
        assert_eq!(category_for_label("other"), None);
    }

    #[test]
    fn sanitizes_filenames() {
        assert_eq!(safe_filename("Beep Beep!"), "beep-beep");
        assert_eq!(safe_filename(""), "clip");
    }

    #[test]
    fn validates_safe_slugs() {
        assert!(is_safe_slug("beep_beep"));
        assert!(is_safe_slug("a1"));
        assert!(!is_safe_slug("_bad"));
        assert!(!is_safe_slug("bad-name"));
        assert!(!is_safe_slug("../bad"));
    }
}
