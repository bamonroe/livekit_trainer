//! HTTP error type shared by every request handler.
//!
//! All fallible handlers return `Result<_, AppError>`. `AppError` carries an
//! HTTP status and a human-readable message and knows how to render itself into
//! a JSON error body, so handlers never build error responses by hand.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use std::io;

/// An error that can be turned directly into an HTTP response.
///
/// Construct one with [`AppError::bad_request`] for client errors (400) or
/// [`AppError::internal`] for server-side failures (500). Conversions from
/// common lower-level error types ([`io::Error`], [`rusqlite::Error`] via
/// [`db_error`]) keep the `?` operator ergonomic inside handlers.
#[derive(Debug)]
pub(crate) struct AppError {
    pub(crate) status: StatusCode,
    pub(crate) message: String,
}

impl AppError {
    /// A 400 Bad Request carrying a caller-facing explanation.
    pub(crate) fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    /// A 500 Internal Server Error carrying a caller-facing explanation.
    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
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

/// Wrap a `rusqlite` failure as a 500 error. Used with `.map_err(db_error)` at
/// database call sites so SQLite errors surface as clean JSON.
pub(crate) fn db_error(error: rusqlite::Error) -> AppError {
    AppError::internal(format!("database error: {error}"))
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
    use axum::response::IntoResponse;

    #[test]
    fn bad_request_carries_400() {
        let response = AppError::bad_request("nope").into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn internal_carries_500() {
        let response = AppError::internal("boom").into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn io_error_maps_to_500() {
        let err: AppError = io::Error::new(io::ErrorKind::Other, "disk").into();
        assert_eq!(err.into_response().status(), StatusCode::INTERNAL_SERVER_ERROR);
    }
}
