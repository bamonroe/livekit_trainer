use axum::{
    extract::DefaultBodyLimit,
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use std::{
    env,
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex},
};

mod align;
mod audio;
mod bundle;
mod constants;
mod db;
mod docker;
mod error;
mod projects;
mod reprocess;
mod review;
mod score;
mod settings;
mod slicing;
mod state;
mod sync;
mod synth;
mod train;
mod util;
mod whisper;

use projects::{create_project, projects};
use reprocess::{reprocess_project, reprocess_recording};
use review::{
    bulk_alignment, bulk_recording_details, bulk_recording_ids, bulk_review, bulk_source_audio,
    delete_recording, delete_review_clip, review_audio,
};
use score::{model_test_grades, score_all_test_takes, score_recording};
use settings::{get_settings, load_settings, update_settings};
use state::AppState;
use sync::sync;
use synth::{
    delete_synth, generate_synth, generate_synth_status, synthetic_audio, synthetic_samples,
};
use train::{
    cancel_training, delete_queue_entry, list_model_runs, list_models, start_training, training_log,
    training_queue, training_scheduler, training_status,
};

#[tokio::main]
async fn main() {
    let bind_addr = env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:8765".to_string());
    let data_root = env::var("DATA_ROOT").unwrap_or_else(|_| "/data/real".to_string());
    let incoming_root =
        env::var("INCOMING_ROOT").unwrap_or_else(|_| "/incoming/bundles".to_string());
    let settings_path =
        env::var("SETTINGS_PATH").unwrap_or_else(|_| "/data/server_settings.json".to_string());
    let settings_path = PathBuf::from(settings_path);
    let db_path = env::var("DB_PATH").unwrap_or_else(|_| "/data/trainer.db".to_string());
    let db = db::open(&PathBuf::from(&db_path)).expect("open database");
    println!("database at {db_path}");

    let whisper_url = env::var("WHISPER_SERVER_URL")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    match &whisper_url {
        Some(url) => println!("whisper server at {url}"),
        None => println!("WHISPER_SERVER_URL not set; transcription will fail until configured"),
    }

    let scorer_url = env::var("SCORER_SERVER_URL")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    match &scorer_url {
        Some(url) => println!("scorer server at {url}"),
        None => println!("SCORER_SERVER_URL not set; model-test scoring disabled"),
    }

    let models_root = env::var("MODELS_DIR")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "/output".to_string());

    let host_repo_root = env::var("HOST_REPO_ROOT")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    match &host_repo_root {
        Some(root) => println!("host repo root {root}; training enabled"),
        None => println!("HOST_REPO_ROOT not set; training endpoints disabled"),
    }
    let trainer_image = env::var("TRAINER_IMAGE")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "livekit-wakeword-trainer:latest".to_string());
    let trainer_gpu = env::var("TRAINER_USE_GPU")
        .map(|value| value.trim() != "0")
        .unwrap_or(true);

    let state = AppState {
        data_root: Arc::new(PathBuf::from(data_root)),
        incoming_root: Arc::new(PathBuf::from(incoming_root)),
        settings: Arc::new(Mutex::new(load_settings(&settings_path))),
        settings_path: Arc::new(settings_path),
        whisper_url: Arc::new(whisper_url),
        scorer_url: Arc::new(scorer_url),
        models_root: Arc::new(PathBuf::from(models_root)),
        host_repo_root: Arc::new(host_repo_root),
        trainer_image: Arc::new(trainer_image),
        trainer_gpu: Arc::new(trainer_gpu),
        db: Arc::new(Mutex::new(db)),
        dispatch_lock: Arc::new(tokio::sync::Mutex::new(())),
        synth_jobs: Arc::new(Mutex::new(std::collections::HashMap::new())),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/settings", get(get_settings).post(update_settings))
        .route("/projects", get(projects).post(create_project))
        .route("/sync", post(sync))
        .route("/reprocess/:slug", post(reprocess_project))
        .route("/reprocess/:slug/:recording_id", post(reprocess_recording))
        .route("/bulk/:slug/recordings", get(bulk_recording_ids))
        .route("/bulk/:slug/recordings/detail", get(bulk_recording_details))
        .route(
            "/bulk/:slug/:recording_id",
            axum::routing::delete(delete_recording),
        )
        .route("/review/:slug/bulk", get(bulk_review))
        .route(
            "/review/:slug/bulk/:recording_id/alignment",
            get(bulk_alignment),
        )
        .route(
            "/review/:slug/bulk/:recording_id/audio",
            get(bulk_source_audio),
        )
        .route("/score/:slug/:recording_id", get(score_recording))
        .route("/score-all/:slug", post(score_all_test_takes))
        .route("/score-grades/:slug", get(model_test_grades))
        .route("/models", get(list_models))
        .route("/models/runs", get(list_model_runs))
        .route("/train/:slug", post(start_training))
        .route("/train/:slug/status", get(training_status))
        .route("/train/:slug/log", get(training_log))
        .route("/train/:slug/cancel", post(cancel_training))
        .route("/queue", get(training_queue))
        .route("/queue/:id", axum::routing::delete(delete_queue_entry))
        .route(
            "/review/:slug/:category/:file_name",
            get(review_audio).delete(delete_review_clip),
        )
        .route(
            "/synth/:slug",
            axum::routing::delete(delete_synth),
        )
        .route("/synth/:slug/sample", get(synthetic_samples))
        .route("/synth/:slug/audio/:file_name", get(synthetic_audio))
        .route("/synth/:slug/generate", post(generate_synth))
        .route("/synth/:slug/generate/status", get(generate_synth_status))
        .layer(DefaultBodyLimit::max(512 * 1024 * 1024))
        .with_state(state.clone());

    // Drive the training queue: reconcile finished runs and dispatch the next
    // queued job on a timer, independent of any request.
    if state.host_repo_root.is_some() {
        tokio::spawn(training_scheduler(state.clone()));
    }

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
