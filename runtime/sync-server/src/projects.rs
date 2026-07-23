//! Project (wake-word) registry endpoints.
//!
//! A project is one wake word the user is collecting for. These handlers list
//! every project with its per-category clip counts (including the negatives it
//! can borrow from other projects) and register/refresh a project by slug so it
//! propagates to the user's other devices before any recording exists.

use crate::error::{db_error, AppError};
use crate::state::{now_ms, AppState};
use crate::util::is_safe_slug;
use crate::db;
use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// The `GET /projects` reply.
#[derive(Debug, Serialize)]
pub(crate) struct ProjectsResponse {
    pub(crate) status: &'static str,
    pub(crate) projects: Vec<ProjectSummary>,
}

/// A project a device is registering with the server so it propagates to the
/// user's other devices even before any recording exists for it. The device's
/// own project id travels as `external_id` so the round-tripped project keeps a
/// stable identity across devices.
#[derive(Debug, Deserialize)]
pub(crate) struct CreateProjectRequest {
    #[serde(alias = "external_id")]
    pub(crate) id: Option<String>,
    pub(crate) slug: String,
    pub(crate) phrase: String,
}

/// One project's summary line for the app's project list.
#[derive(Debug, Serialize)]
pub(crate) struct ProjectSummary {
    pub(crate) id: String,
    pub(crate) slug: String,
    pub(crate) phrase: String,
    pub(crate) created_at_millis: i64,
    pub(crate) bulk_slice_count: usize,
    pub(crate) positive_count: usize,
    pub(crate) negative_count: usize,
    pub(crate) background_count: usize,
    /// Negatives available from every *other* project, reusable for this one.
    pub(crate) pooled_negative_count: usize,
}

/// `GET /projects`: every project with its counts and cross-project negative pool.
pub(crate) async fn projects(
    State(state): State<AppState>,
) -> Result<Json<ProjectsResponse>, AppError> {
    let rows = {
        let conn = state.db.lock().expect("db lock poisoned");
        db::project_summaries(&conn).map_err(db_error)?
    };
    // Cross-wake-word reuse: every other project's plain negatives, plus their
    // positives (reused as hard negatives), are available to this project. Hard
    // negatives are project-scoped, so they never enter the shared pool.
    let total_poolable_negatives: i64 = rows.iter().map(|r| r.poolable_negative_count).sum();
    let total_positives: i64 = rows.iter().map(|r| r.positive_count).sum();
    let projects = rows
        .into_iter()
        .map(|row| {
            let pooled = (total_poolable_negatives - row.poolable_negative_count)
                + (total_positives - row.positive_count);
            ProjectSummary {
                id: row.external_id,
                slug: row.slug,
                phrase: row.phrase,
                created_at_millis: row.created_at_ms,
                bulk_slice_count: row.bulk_slice_count as usize,
                positive_count: row.positive_count as usize,
                negative_count: row.negative_count as usize,
                background_count: row.background_count as usize,
                pooled_negative_count: pooled.max(0) as usize,
            }
        })
        .collect();
    Ok(Json(ProjectsResponse {
        status: "ok",
        projects,
    }))
}

/// Register a project on the server so it shows up on the user's other devices
/// without waiting for a recording to be synced. Idempotent on slug: creating
/// the same wake word again just refreshes its phrase and id.
pub(crate) async fn create_project(
    State(state): State<AppState>,
    Json(request): Json<CreateProjectRequest>,
) -> Result<Json<Value>, AppError> {
    let slug = request.slug.trim().to_string();
    let phrase = request.phrase.trim().to_string();
    if !is_safe_slug(&slug) {
        return Err(AppError::bad_request(format!("unsafe slug: {slug:?}")));
    }
    if phrase.is_empty() {
        return Err(AppError::bad_request("phrase is required"));
    }
    let external_id = request
        .id
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    {
        let conn = state.db.lock().expect("db lock poisoned");
        db::upsert_project(&conn, &slug, &phrase, external_id, now_ms()).map_err(db_error)?;
    }
    Ok(Json(json!({ "status": "ok", "slug": slug })))
}
