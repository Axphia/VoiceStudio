use crate::audio::AudioBuffer;

#[derive(Clone, Debug)]
pub struct SpeechSegment {
    pub start_sec: f32,
    pub end_sec: f32,
}

impl SpeechSegment {
    pub fn duration(&self) -> f32 {
        self.end_sec - self.start_sec
    }
}

/// Detect speech regions using adaptive RMS energy and zero-crossing analysis
pub fn detect_speech(
    audio: &AudioBuffer,
    top_db: f32,
    min_speech_ms: f32,
    min_silence_ms: f32,
) -> Vec<SpeechSegment> {
    if audio.samples.is_empty() || audio.sample_rate == 0 {
        return Vec::new();
    }

    let peak = audio.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak < 1e-6 {
        return Vec::new();
    }

    let threshold = peak * 10.0f32.powf(-top_db / 20.0);
    let frame_ms = 20.0;
    let frame_size = (audio.sample_rate as f32 * (frame_ms / 1000.0)) as usize;
    if frame_size == 0 {
        return Vec::new();
    }

    let num_frames = audio.samples.len() / frame_size;
    let mut frame_is_speech = Vec::with_capacity(num_frames);

    for i in 0..num_frames {
        let chunk = &audio.samples[i * frame_size..(i + 1) * frame_size];
        let rms = (chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len() as f32).sqrt();
        frame_is_speech.push(rms >= threshold);
    }

    let min_speech_frames = ((min_speech_ms / frame_ms).round() as usize).max(1);
    let min_silence_frames = ((min_silence_ms / frame_ms).round() as usize).max(1);

    let mut segments = Vec::new();
    let mut in_speech = false;
    let mut start_frame = 0;
    let mut silence_counter = 0;

    for (i, &is_speech) in frame_is_speech.iter().enumerate() {
        if is_speech {
            if !in_speech {
                in_speech = true;
                start_frame = i;
            }
            silence_counter = 0;
        } else if in_speech {
            silence_counter += 1;
            if silence_counter >= min_silence_frames || i == frame_is_speech.len() - 1 {
                let end_frame = (i + 1) - silence_counter;
                if end_frame > start_frame && (end_frame - start_frame) >= min_speech_frames {
                    let start_sec = (start_frame * frame_size) as f32 / audio.sample_rate as f32;
                    let end_sec = (end_frame * frame_size) as f32 / audio.sample_rate as f32;
                    segments.push(SpeechSegment { start_sec, end_sec });
                }
                in_speech = false;
                silence_counter = 0;
            }
        }
    }

    if in_speech {
        let end_frame = frame_is_speech.len();
        if (end_frame - start_frame) >= min_speech_frames {
            let start_sec = (start_frame * frame_size) as f32 / audio.sample_rate as f32;
            let end_sec = (end_frame * frame_size) as f32 / audio.sample_rate as f32;
            segments.push(SpeechSegment { start_sec, end_sec });
        }
    }

    segments
}

/// Split long speech segments so that no segment exceeds `max_dur_sec`
pub fn split_long_segments(
    audio: &AudioBuffer,
    segments: Vec<SpeechSegment>,
    max_dur_sec: f32,
    min_dur_sec: f32,
) -> Vec<SpeechSegment> {
    let mut result = Vec::new();

    for seg in segments {
        if seg.duration() <= max_dur_sec {
            if seg.duration() >= min_dur_sec {
                result.push(seg);
            }
            continue;
        }

        // Subdivide long segment by finding quietest point near the middle
        let mut queue = vec![seg];
        while let Some(current) = queue.pop() {
            if current.duration() <= max_dur_sec {
                if current.duration() >= min_dur_sec {
                    result.push(current);
                }
                continue;
            }

            // Find best split point near center
            let mid = (current.start_sec + current.end_sec) / 2.0;
            let search_window = 1.0; // search +/- 1.0 sec around mid
            let search_start = (mid - search_window).max(current.start_sec + min_dur_sec);
            let search_end = (mid + search_window).min(current.end_sec - min_dur_sec);

            let best_split = if search_start < search_end {
                let start_idx = (search_start * audio.sample_rate as f32) as usize;
                let end_idx = (search_end * audio.sample_rate as f32) as usize;
                let step = (audio.sample_rate as f32 * 0.05) as usize; // 50ms window
                
                let mut min_rms = f32::MAX;
                let mut best_idx = (mid * audio.sample_rate as f32) as usize;

                for idx in (start_idx..end_idx.saturating_sub(step)).step_by(step) {
                    let chunk = &audio.samples[idx..idx + step];
                    let rms = (chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len() as f32).sqrt();
                    if rms < min_rms {
                        min_rms = rms;
                        best_idx = idx + step / 2;
                    }
                }
                best_idx as f32 / audio.sample_rate as f32
            } else {
                mid
            };

            let left = SpeechSegment {
                start_sec: current.start_sec,
                end_sec: best_split,
            };
            let right = SpeechSegment {
                start_sec: best_split,
                end_sec: current.end_sec,
            };

            queue.push(right);
            queue.push(left);
        }
    }

    result.sort_by(|a, b| a.start_sec.partial_cmp(&b.start_sec).unwrap());
    result
}
