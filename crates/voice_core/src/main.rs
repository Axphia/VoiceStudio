use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use voice_core::audio::AudioBuffer;
use voice_core::vad::{detect_speech, split_long_segments};

#[derive(Parser)]
#[command(name = "voice-prep")]
#[command(about = "High-performance Rust Audio Preprocessor for VoiceStudio", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Clean and prepare a single audio file (trim silence, resample, normalize)
    Clean {
        #[arg(short, long)]
        input: PathBuf,

        #[arg(short, long)]
        output: PathBuf,

        /// Target sample rate (Hz)
        #[arg(long, default_value_t = 22050)]
        sr: u32,

        /// Peak normalization target (e.g. 0.707 for -3 dBFS, 0.95 for -0.5 dBFS)
        #[arg(long, default_value_t = 0.707)]
        peak: f32,

        /// Silence trim threshold in dB
        #[arg(long, default_value_t = 30.0)]
        top_db: f32,

        /// Start offset in seconds
        #[arg(long, default_value_t = 0.0)]
        start: f32,

        /// Max duration in seconds (0 = full file)
        #[arg(long, default_value_t = 0.0)]
        dur: f32,
    },

    /// Multi-threaded batch preparation of entire audio directories for TTS fine-tuning
    BatchPrep {
        #[arg(short, long)]
        input_dir: PathBuf,

        #[arg(short, long)]
        out_dir: PathBuf,

        /// Target sample rate (Hz)
        #[arg(long, default_value_t = 22050)]
        sr: u32,

        /// Maximum segment duration in seconds (XTTS limit is ~11.0s)
        #[arg(long, default_value_t = 11.0)]
        max_dur: f32,

        /// Minimum segment duration in seconds
        #[arg(long, default_value_t = 1.0)]
        min_dur: f32,

        /// Peak normalization target
        #[arg(long, default_value_t = 0.707)]
        peak: f32,

        /// Silence trim threshold in dB
        #[arg(long, default_value_t = 30.0)]
        top_db: f32,
    },

    /// Inspect audio files: durations, sample rates, peak levels, and RMS stats
    Stats {
        #[arg(short, long)]
        path: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Clean {
            input,
            output,
            sr,
            peak,
            top_db,
            start,
            dur,
        } => {
            let t0 = Instant::now();
            println!("[voice-prep] Cleaning: {:?}", input);

            let mut audio = AudioBuffer::load_wav(&input)?;
            let orig_dur = audio.duration_sec();

            // Slice start and optional duration
            if start > 0.0 || dur > 0.0 {
                let actual_dur = if dur > 0.0 { dur } else { orig_dur - start };
                audio = audio.slice(start, actual_dur);
            }

            // Resample
            if audio.sample_rate != sr {
                audio = audio.resample(sr)?;
            }

            // Trim silence & normalize
            audio.trim_silence(top_db, 50.0);
            audio.normalize_peak(peak);

            audio.save_wav(&output)?;
            let elapsed = t0.elapsed();
            println!(
                "[voice-prep] Done! Original: {:.2}s -> Output: {:.2}s @ {}Hz (took {:.1}ms) -> {:?}",
                orig_dur,
                audio.duration_sec(),
                sr,
                elapsed.as_secs_f64() * 1000.0,
                output
            );
        }

        Commands::BatchPrep {
            input_dir,
            out_dir,
            sr,
            max_dur,
            min_dur,
            peak,
            top_db,
        } => {
            let t0 = Instant::now();
            println!("[voice-prep] Starting high-speed batch dataset prep across all CPU cores...");
            println!("[voice-prep] Input: {:?}, Target SR: {} Hz", input_dir, sr);

            let wavs_out = out_dir.join("wavs");
            fs::create_dir_all(&wavs_out)?;

            let mut wav_files = Vec::new();
            for entry in fs::read_dir(&input_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        if ext.to_string_lossy().to_lowercase() == "wav" {
                            wav_files.push(path);
                        }
                    }
                }
            }

            if wav_files.is_empty() {
                eprintln!("[voice-prep] No .wav files found in {:?}", input_dir);
                return Ok(());
            }

            println!("[voice-prep] Found {} WAV files to process.", wav_files.len());

            let total_clips = AtomicUsize::new(0);
            let total_dur_ms = AtomicUsize::new(0);

            // Parallel processing with Rayon across CPU cores
            let processed_rows: Vec<Vec<(String, String)>> = wav_files
                .par_iter()
                .map(|file_path| {
                    let mut rows = Vec::new();
                    let audio_res = AudioBuffer::load_wav(file_path);
                    let mut audio = match audio_res {
                        Ok(a) => a,
                        Err(e) => {
                            eprintln!("[voice-prep] Skipping {:?}: {}", file_path, e);
                            return rows;
                        }
                    };

                    // Resample to target SR
                    if audio.sample_rate != sr {
                        match audio.resample(sr) {
                            Ok(resampled) => audio = resampled,
                            Err(e) => {
                                eprintln!("[voice-prep] Resample failed for {:?}: {}", file_path, e);
                                return rows;
                            }
                        }
                    }

                    // VAD & long segment splitting
                    let raw_segs = detect_speech(&audio, top_db, 200.0, 250.0);
                    let segs = split_long_segments(&audio, raw_segs, max_dur, min_dur);

                    let stem = file_path
                        .file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_else(|| "clip".to_string());

                    for (idx, seg) in segs.iter().enumerate() {
                        let mut clip = audio.slice(seg.start_sec, seg.duration());
                        if clip.duration_sec() < min_dur {
                            continue;
                        }
                        clip.normalize_peak(peak);

                        let clip_name = format!("{}_{:04}", stem, idx);
                        let clip_path = wavs_out.join(format!("{}.wav", clip_name));

                        if let Ok(()) = clip.save_wav(&clip_path) {
                            total_clips.fetch_add(1, Ordering::Relaxed);
                            total_dur_ms.fetch_add(
                                (clip.duration_sec() * 1000.0) as usize,
                                Ordering::Relaxed,
                            );
                            rows.push((clip_name, format!("Speech clip from {}", stem)));
                        }
                    }
                    rows
                })
                .collect();

            // Export metadata.csv in ljspeech format: clip_name|text|text
            let meta_path = out_dir.join("metadata.csv");
            let mut writer = csv::WriterBuilder::new()
                .delimiter(b'|')
                .from_path(&meta_path)?;

            for file_rows in processed_rows {
                for (name, text) in file_rows {
                    writer.write_record(&[&name, &text, &text])?;
                }
            }
            writer.flush()?;

            let clips_count = total_clips.load(Ordering::SeqCst);
            let total_seconds = total_dur_ms.load(Ordering::SeqCst) as f64 / 1000.0;
            let elapsed = t0.elapsed();

            println!("\n=======================================================");
            println!("[voice-prep] BATCH PREP COMPLETED IN {:.2}s!", elapsed.as_secs_f64());
            println!("[voice-prep] Output Directory: {:?}", out_dir);
            println!("[voice-prep] Generated Clips: {}", clips_count);
            println!("[voice-prep] Total Speech Time: {:.2} min ({:.1}s)", total_seconds / 60.0, total_seconds);
            println!("[voice-prep] Metadata exported -> {:?}", meta_path);
            println!("=======================================================\n");
        }

        Commands::Stats { path } => {
            if path.is_file() {
                let audio = AudioBuffer::load_wav(&path)?;
                let peak = audio.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
                let rms = (audio.samples.iter().map(|x| x * x).sum::<f32>() / audio.samples.len() as f32).sqrt();
                let peak_db = 20.0 * (peak + 1e-7).log10();
                let rms_db = 20.0 * (rms + 1e-7).log10();

                println!("File: {:?}", path);
                println!("  Sample Rate: {} Hz", audio.sample_rate);
                println!("  Channels:    1 (mono)");
                println!("  Duration:    {:.3} s", audio.duration_sec());
                println!("  Samples:     {}", audio.samples.len());
                println!("  Peak:        {:.4} ({:.2} dBFS)", peak, peak_db);
                println!("  RMS:         {:.4} ({:.2} dBFS)", rms, rms_db);
            } else if path.is_dir() {
                let mut count = 0;
                let mut total_sec = 0.0;
                for entry in fs::read_dir(&path)? {
                    let entry = entry?;
                    let p = entry.path();
                    if p.extension().map(|e| e.to_string_lossy().to_lowercase()) == Some("wav".into()) {
                        if let Ok(audio) = AudioBuffer::load_wav(&p) {
                            count += 1;
                            total_sec += audio.duration_sec();
                        }
                    }
                }
                println!("Directory: {:?}", path);
                println!("  Total WAV Files: {}", count);
                println!("  Total Duration:  {:.2} s ({:.2} min)", total_sec, total_sec / 60.0);
            }
        }
    }

    Ok(())
}
