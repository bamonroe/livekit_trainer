//! Small, pure helpers shared across handlers: path-safety guards, identifier
//! validation, filename sanitizing, and a minimal query-string parser.
//!
//! Everything here is side-effect free and cheap to unit test, which is why it
//! lives apart from the request handlers that call it.

use crate::error::AppError;
use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};

/// Join `relative` onto `root`, rejecting any path that could escape `root`.
///
/// Absolute paths and any `..`/prefix component are refused with a 400 so an
/// attacker-controlled manifest entry can never reference a file outside the
/// extracted bundle.
pub(crate) fn safe_join(root: &Path, relative: &str) -> Result<PathBuf, AppError> {
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

/// Map a rich collection label to the coarse training category it feeds.
///
/// Correction labels fold into their training role: `false_negative` audio was
/// a missed positive, `false_positive` audio was an errant trigger, so they
/// train as positive and negative respectively. Unknown labels return `None`.
pub(crate) fn category_for_label(label: &str) -> Option<&'static str> {
    match label {
        "positive" | "false_negative" => Some("positive"),
        "negative" | "false_positive" => Some("negative"),
        "hard_negative" => Some("hard_negative"),
        "background" => Some("background"),
        _ => None,
    }
}

/// True for a wake-word slug safe to use as a path segment: it must start with a
/// lowercase letter or digit and contain only `[a-z0-9_]`.
pub(crate) fn is_safe_slug(slug: &str) -> bool {
    let mut chars = slug.chars();
    match chars.next() {
        Some(ch) if ch.is_ascii_lowercase() || ch.is_ascii_digit() => {}
        _ => return false,
    }
    chars.all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
}

/// True for a recording id safe to use as a path segment: non-empty and made of
/// ASCII alphanumerics, `_`, or `-`.
pub(crate) fn is_safe_recording_id(recording_id: &str) -> bool {
    !recording_id.is_empty()
        && recording_id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-'))
}

/// A run id is a compact UTC timestamp like `20260719T153000Z`; restrict it to
/// ASCII alphanumerics so it can never escape the run directory.
pub(crate) fn is_safe_run_id(run_id: &str) -> bool {
    !run_id.is_empty() && run_id.chars().all(|ch| ch.is_ascii_alphanumeric())
}

/// Collapse arbitrary text into a lowercase, dash-separated filename stem,
/// falling back to `"clip"` when nothing usable remains.
pub(crate) fn safe_filename(value: &str) -> String {
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

/// Render a human-readable import summary: either a plain validated-count line
/// or a bulleted list of warnings followed by that count.
pub(crate) fn validation_summary(warnings: &[String], clip_count: usize) -> String {
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

/// Minimal `a=b&c=d` query parser; avoids pulling in axum's `query` feature.
pub(crate) fn parse_query(raw: Option<&str>) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let Some(raw) = raw else { return map };
    for pair in raw.split('&').filter(|p| !p.is_empty()) {
        let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
        map.insert(key.to_string(), value.to_string());
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_labels_to_categories() {
        assert_eq!(category_for_label("positive"), Some("positive"));
        assert_eq!(category_for_label("false_negative"), Some("positive"));
        assert_eq!(category_for_label("hard_negative"), Some("hard_negative"));
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

    #[test]
    fn run_ids_reject_path_separators() {
        assert!(is_safe_run_id("20260719T153000Z"));
        assert!(!is_safe_run_id(""));
        assert!(!is_safe_run_id("../run"));
        assert!(!is_safe_run_id("run-1"));
    }

    #[test]
    fn safe_join_rejects_escapes() {
        let root = Path::new("/data/bundle");
        assert!(safe_join(root, "clips/a.wav").is_ok());
        assert!(safe_join(root, "../escape").is_err());
        assert!(safe_join(root, "/etc/passwd").is_err());
    }

    #[test]
    fn recording_ids_allow_dash_and_underscore() {
        assert!(is_safe_recording_id("rec_1-2"));
        assert!(!is_safe_recording_id(""));
        assert!(!is_safe_recording_id("rec/1"));
    }
}
