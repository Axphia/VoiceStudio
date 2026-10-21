use anyhow::{bail, Context, Result};
use hound::{WavReader, WavSpec, WavWriter};
use rubato::{FftFixedInOut, Resampler};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct AudioBuffer {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl AudioBuffer {
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self { samples, sample_rate }
    }

    pub fn duration_sec(&self) -> f32 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.samples.len() as f32 / self.sample_rate as f32
        }
    }

    pub fn load_wav<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut reader = WavReader::open(path.as_ref())
            .with_context(|| format!("Failed to open WAV: {:?}", path.as_ref()))?;
        let spec = reader.spec();
        let channels = spec.channels as usize;
        let sample_rate = spec.sample_rate;

        let raw_samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
            (hound::SampleFormat::Int, 16) => reader
                .samples::<i16>()
                .filter_map(Result::ok)
                .map(|s| s as f32 / 32768.0)
                .collect(),
            (hound::SampleFormat::Int, 24) => reader
                .samples::<i32>()
                .filter_map(Result::ok)
                .map(|s| s as f32 / 8388608.0)
                .collect(),
            (hound::SampleFormat::Int, 32) => reader
                .samples::<i32>()
                .filter_map(Result::ok)
                .map(|s| s as f32 / 2147483648.0)
                .collect(),
            (hound::SampleFormat::Float, 32) => reader
                .samples::<f32>()
                .filter_map(Result::ok)
                .collect(),
            (fmt, bits) => bail!("Unsupported WAV format: {:?} at {} bits", fmt, bits),
        };

        let mono_samples = if channels > 1 {
            let num_frames = raw_samples.len() / channels;
            let mut mono = Vec::with_capacity(num_frames);
            for i in 0..num_frames {
                let mut sum = 0.0f32;
                for ch in 0..channels {
                    sum += raw_samples[i * channels + ch];
                }
                mono.push(sum / channels as f32);
            }
            mono
        } else {
            raw_samples
        };

        Ok(Self {
            samples: mono_samples,
            sample_rate,
        })
    }

    pub fn save_wav<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }
        let spec = WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = WavWriter::create(path.as_ref(), spec)
            .with_context(|| format!("Failed to create WAV: {:?}", path.as_ref()))?;

        for &s in &self.samples {
            let clamped = s.clamp(-1.0, 1.0);
            let val = (clamped * 32767.0).round() as i16;
            writer.write_sample(val)?;
        }
        writer.finalize()?;
        Ok(())
    }

    /// Resample audio to target sample rate using rubato
    pub fn resample(&self, target_sr: u32) -> Result<Self> {
        if self.sample_rate == target_sr || self.samples.is_empty() {
            return Ok(self.clone());
        }

        let chunk_size = 1024;
        let mut resampler = FftFixedInOut::<f32>::new(
            self.sample_rate as usize,
            target_sr as usize,
            chunk_size,
            1,
        ).context("Failed to initialize Rubato resampler")?;

        let mut output_samples = Vec::new();
        let mut pos = 0;

        while pos < self.samples.len() {
            let needed = resampler.input_frames_next();
            let available = self.samples.len() - pos;
            let take = needed.min(available);

            let mut in_buffer = vec![0.0f32; needed];
            in_buffer[..take].copy_from_slice(&self.samples[pos..pos + take]);
            pos += take;

            let waves_in = vec![in_buffer];
            let waves_out = resampler.process(&waves_in, None)
                .context("Error during rubato resampling step")?;
            output_samples.extend_from_slice(&waves_out[0]);
        }

        let expected_len = (self.samples.len() as f64 * target_sr as f64 / self.sample_rate as f64).round() as usize;
        output_samples.truncate(expected_len);

        Ok(Self {
            samples: output_samples,
            sample_rate: target_sr,
        })
    }

    pub fn normalize_peak(&mut self, target_peak: f32) {
        let max_val = self.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if max_val > 1e-7 {
            let gain = target_peak / max_val;
            for s in &mut self.samples {
                *s *= gain;
            }
        }
    }

    pub fn trim_silence(&mut self, top_db: f32, pad_ms: f32) {
        if self.samples.is_empty() {
            return;
        }
        let peak = self.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if peak < 1e-7 {
            self.samples.clear();
            return;
        }

        let threshold = peak * 10.0f32.powf(-top_db / 20.0);
        let frame_size = (self.sample_rate as f32 * 0.02) as usize;
        if frame_size == 0 {
            return;
        }

        let mut start_idx = 0;
        for (i, window) in self.samples.chunks(frame_size).enumerate() {
            let rms = (window.iter().map(|x| x * x).sum::<f32>() / window.len() as f32).sqrt();
            if rms >= threshold {
                start_idx = i * frame_size;
                break;
            }
        }

        let mut end_idx = self.samples.len();
        for (i, window) in self.samples.rchunks(frame_size).enumerate() {
            let rms = (window.iter().map(|x| x * x).sum::<f32>() / window.len() as f32).sqrt();
            if rms >= threshold {
                end_idx = self.samples.len() - (i * frame_size);
                break;
            }
        }

        let pad_samples = (self.sample_rate as f32 * (pad_ms / 1000.0)) as usize;
        start_idx = start_idx.saturating_sub(pad_samples);
        end_idx = (end_idx + pad_samples).min(self.samples.len());

        if start_idx < end_idx {
            self.samples = self.samples[start_idx..end_idx].to_vec();
        } else {
            self.samples.clear();
        }
    }

    pub fn slice(&self, start_sec: f32, dur_sec: f32) -> Self {
        let start = (start_sec * self.sample_rate as f32).round() as usize;
        let len = (dur_sec * self.sample_rate as f32).round() as usize;
        let end = (start + len).min(self.samples.len());

        let slice_samples = if start < self.samples.len() {
            self.samples[start..end].to_vec()
        } else {
            Vec::new()
        };

        Self {
            samples: slice_samples,
            sample_rate: self.sample_rate,
        }
    }
}
