use anyhow::Result;
use axum::{
    extract::{Json, Query, State},
    http::{header, HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::info;
use voice_core::audio::AudioBuffer;

#[derive(Parser, Debug)]
#[command(name = "voice-server")]
#[command(about = "Ultra-low latency real-time Voice & Audio server in Rust")]
struct Args {
    #[arg(short, long, default_value = "127.0.0.1")]
    host: String,

    #[arg(short, long, default_value_t = 8088)]
    port: u16,

    #[arg(long, default_value = "models")]
    models_dir: PathBuf,
}

#[derive(Clone)]
struct AppState {
    models_dir: PathBuf,
    start_time: Instant,
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    engine: &'static str,
    uptime_sec: u64,
    cpu_cores: usize,
    models_detected: usize,
}

#[derive(Serialize)]
struct ModelInfo {
    name: String,
    has_model_pth: bool,
    has_config: bool,
    has_vocab: bool,
    path: String,
}

#[derive(Deserialize)]
struct TtsRequest {
    text: String,
    #[serde(default = "default_speaker")]
    speaker: String,
    #[serde(default = "default_speed")]
    speed: f32,
    #[serde(default = "default_lang")]
    language: String,
}

fn default_speaker() -> String {
    "Kei_voice".to_string()
}
fn default_speed() -> f32 {
    1.0
}
fn default_lang() -> String {
    "ja".to_string()
}

#[derive(Deserialize)]
struct AudioStatsRequest {
    path: String,
}

#[derive(Serialize)]
struct AudioStatsResponse {
    path: String,
    duration_sec: f32,
    sample_rate: u32,
    peak_linear: f32,
    peak_dbfs: f32,
    rms_linear: f32,
    rms_dbfs: f32,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "voice_server=info,tower_http=info".into()),
        )
        .init();

    let args = Args::parse();
    let state = Arc::new(AppState {
        models_dir: args.models_dir.clone(),
        start_time: Instant::now(),
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/api/v1/models", get(models_handler))
        .route("/api/v1/stats", post(stats_handler))
        .route("/api/v1/clean", post(clean_handler))
        .route("/api/v1/tts", post(tts_handler))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    info!("🚀 Rust Voice Server listening on http://{}", addr);
    info!("📁 Serving VoiceStudio models from: {:?}", args.models_dir);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_handler(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let models_detected = count_models(&state.models_dir);
    Json(HealthResponse {
        status: "ok",
        engine: "Rust Axum Voice Engine v0.1.0",
        uptime_sec: state.start_time.elapsed().as_secs(),
        cpu_cores: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4),
        models_detected,
    })
}

fn count_models(dir: &Path) -> usize {
    if !dir.exists() {
        return 0;
    }
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter(|e| e.path().is_dir())
                .count()
        })
        .unwrap_or(0)
}

async fn models_handler(State(state): State<Arc<AppState>>) -> Json<Vec<ModelInfo>> {
    let mut list = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&state.models_dir) {
        for entry in entries.filter_map(Result::ok) {
            let p = entry.path();
            if p.is_dir() {
                let name = p.file_name().unwrap().to_string_lossy().to_string();
                let has_model_pth = p.join("model.pth").exists();
                let has_config = p.join("config.json").exists();
                let has_vocab = p.join("vocab.json").exists();

                list.push(ModelInfo {
                    name,
                    has_model_pth,
                    has_config,
                    has_vocab,
                    path: p.to_string_lossy().to_string(),
                });
            }
        }
    }
    Json(list)
}

async fn stats_handler(Json(req): Json<AudioStatsRequest>) -> Result<Json<AudioStatsResponse>, StatusCode> {
    let audio = AudioBuffer::load_wav(&req.path).map_err(|_| StatusCode::NOT_FOUND)?;
    let peak = audio.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    let rms = (audio.samples.iter().map(|x| x * x).sum::<f32>() / audio.samples.len().max(1) as f32).sqrt();
    let peak_dbfs = 20.0 * (peak + 1e-7).log10();
    let rms_dbfs = 20.0 * (rms + 1e-7).log10();

    Ok(Json(AudioStatsResponse {
        path: req.path,
        duration_sec: audio.duration_sec(),
        sample_rate: audio.sample_rate,
        peak_linear: peak,
        peak_dbfs,
        rms_linear: rms,
        rms_dbfs,
    }))
}

#[derive(Deserialize)]
struct CleanRequest {
    input_path: String,
    output_path: String,
    #[serde(default = "default_sr")]
    target_sr: u32,
    #[serde(default = "default_top_db")]
    top_db: f32,
}

fn default_sr() -> u32 { 22050 }
fn default_top_db() -> f32 { 30.0 }

async fn clean_handler(Json(req): Json<CleanRequest>) -> Result<Json<serde_json::Value>, StatusCode> {
    let t0 = Instant::now();
    let mut audio = AudioBuffer::load_wav(&req.input_path).map_err(|_| StatusCode::NOT_FOUND)?;
    
    if audio.sample_rate != req.target_sr {
        audio = audio.resample(req.target_sr).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    }
    audio.trim_silence(req.top_db, 50.0);
    audio.normalize_peak(0.707);
    audio.save_wav(&req.output_path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let elapsed = t0.elapsed();
    Ok(Json(serde_json::json!({
        "status": "success",
        "output_path": req.output_path,
        "duration_sec": audio.duration_sec(),
        "sample_rate": audio.sample_rate,
        "processing_time_ms": elapsed.as_secs_f64() * 1000.0
    })))
}

async fn tts_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TtsRequest>,
) -> impl IntoResponse {
    let t0 = Instant::now();
    info!("[TTS] text='{}' speaker='{}' speed={}", req.text, req.speaker, req.speed);

    let model_path = state.models_dir.join(&req.speaker);
    if !model_path.exists() {
        return (
            StatusCode::NOT_FOUND,
            HeaderMap::new(),
            format!("Speaker model '{}' not found in {:?}", req.speaker, state.models_dir).into_bytes(),
        );
    }

    // High performance tone/synthesizer preview:
    // Generates a clean preview chime / audio stream at 22050 Hz
    let sample_rate = 22050;
    let duration = (req.text.chars().count() as f32 * 0.15 / req.speed).clamp(0.5, 5.0);
    let num_samples = (duration * sample_rate as f32) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    let freq = match req.speaker.as_str() {
        "Kei_voice" => 330.0,
        "japanese_mix" => 261.63,
        _ => 440.0,
    };

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let env = (1.0 - t / duration).max(0.0);
        let s = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.3 * env;
        samples.push(s);
    }

    let audio = AudioBuffer::new(samples, sample_rate);
    let mut wav_bytes = Vec::new();
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = std::io::Cursor::new(&mut wav_bytes);
    if let Ok(mut writer) = hound::WavWriter::new(&mut cursor, spec) {
        for &s in &audio.samples {
            let val = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
            let _ = writer.write_sample(val);
        }
        let _ = writer.finalize();
    }

    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, "audio/wav".parse().unwrap());
    headers.insert(
        header::HeaderName::from_static("x-latency-ms"),
        format!("{:.2}", t0.elapsed().as_secs_f64() * 1000.0).parse().unwrap(),
    );

    (StatusCode::OK, headers, wav_bytes)
}
