use anyhow::{anyhow, Result};
use std::{fs::File, path::Path};
use symphonia::core::{
    audio::SampleBuffer, codecs::DecoderOptions, formats::FormatOptions, io::MediaSourceStream,
    meta::MetadataOptions, probe::Hint,
};

/// Open any audio container (e.g., mp3, m4a, ogg, wav).
/// Read its packets (compressed codec frames: mp3/aac/flac/etc).
/// Decode packets to PCM i16 samples (raw waveform).
/// Serialize to little-endian bytes (ready to chunk into Solana tx payloads).
///
pub fn decode_audio(input: &Path) -> Result<(Vec<u8>, u32, u8)> {
    // Decode to PCM i16 samples
    let PcmData {
        samples,
        sample_rate,
        channels,
    } = decode_to_pcm_i16(input)?;

    // Convert i16 samples to bytes (2 bytes per sample)
    let pcm_bytes = i16_to_le_bytes_vec(&samples);

    Ok((pcm_bytes, sample_rate, channels))
}

struct PcmData {
    samples: Vec<i16>, // interleaved:
    sample_rate: u32,
    channels: u8, // 1 = mono, >=2 = interleaved multi-channel
}

/// Decode any supported file into **PCM i16 mono**.
/// Notes:
/// - Containers (mp3/m4a/ogg) store compressed packets for space.
/// - We need raw PCM bytes to stream via on-chain chunks and reconstruct on playback.
///
/// Steps:
/// - Probe container → find default audio track.
/// - For each *packet* (codec frame), run the decoder → decoded buffer (PCM).
fn decode_to_pcm_i16(path: &Path) -> Result<PcmData> {
    let file = File::open(path)?;

    // Create the media source stream
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Provide a hint (file extension) to help the format probe choose a demuxer.
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }

    // Container demuxer (format) — parses headers/metadata and yields packets.
    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;

    let mut format = probed.format;

    // Pick the default audio track (there can be multiple streams).
    let track = format
        .default_track()
        .ok_or_else(|| anyhow!("no default audio track found"))?;

    // Get the track ID for later use.
    let track_id = track.id;

    // Codec decoder — turns *packets* → decoded **PCM** frames.
    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    // We keep the source sample rate and optionally resample later.
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| anyhow!("missing sample_rate"))?;

    // Output PCM samples (mono or stereo i16)
    let mut pcm_output: Vec<i16> = Vec::new();
    let mut out_channels: u8 = 0;

    //  Read packets sequentially (each packet is a small time-slice of compressed audio).
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(err) => {
                // Done reading the stream (EOF) or container ended.
                match err {
                    symphonia::core::errors::Error::ResetRequired => {
                        return Err(anyhow!("decoder reset required; unsupported stream change"));
                    }
                    symphonia::core::errors::Error::IoError(_) => break, // EOF
                    _ => break,
                }
            }
        };

        // Skip packets that belong to other tracks.
        if packet.track_id() != track_id {
            continue;
        }

        // Decode this compressed packet → decoded buffer (PCM samples).
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(e) => {
                // Some formats may have recoverable errors; skip bad packets.
                match e {
                    symphonia::core::errors::Error::DecodeError(_) => continue,
                    _ => return Err(anyhow!("decode error: {e}")),
                }
            }
        };

        // Convert decoded buffer to interleaved f32 once
        let spec = *decoded.spec();
        let frames = decoded.frames();
        let ch = spec.channels.count();
        let mut buf = SampleBuffer::<f32>::new(frames as u64, spec);
        buf.copy_interleaved_ref(decoded);
        let s = buf.samples(); // interleaved f32

        // If mono, downmix (average channels); else keep original channels.
        if ch == 1 {
            if out_channels == 0 {
                out_channels = 1;
            }
            for f in 0..frames {
                let mut acc = 0.0f32;
                //
                for c in 0..ch {
                    //
                    acc += s[f * ch + c];
                }
                let avg = acc / (ch as f32);
                pcm_output.push(float_to_i16(avg));
            }
        } else {
            // Keep original channels, convert f32 -> i16, keep interleaved layout.
            if out_channels == 0 {
                out_channels = ch as u8;
            }
            pcm_output.reserve(frames * ch);
            for sample in s.iter().copied() {
                pcm_output.push(float_to_i16(sample));
            }
        }
    }

    if pcm_output.is_empty() {
        return Err(anyhow!("no audio decoded"));
    }

    Ok(PcmData {
        samples: pcm_output,
        sample_rate,
        channels: out_channels,
    })
}

#[inline(always)]
fn float_to_i16(v: f32) -> i16 {
    // Treat NaN as silence, clamp to [-1.0, 1.0], then scale and round.
    let v = if v.is_nan() { 0.0 } else { v.clamp(-1.0, 1.0) };
    (v * i16::MAX as f32).round() as i16
}

/// Convert i16 samples to little-endian bytes.
fn i16_to_le_bytes_vec(samples: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Chunk ensuring we end on frame boundaries.
/// frame = all channels for one time index = channels * 2 bytes (i16le)
pub fn chunk_pcm_aligned(buf: &[u8], channels: u8, max_chunk: usize) -> Vec<Vec<u8>> {
    assert!(channels >= 1);
    let bytes_per_frame = (channels as usize) * 2;
    assert!(
        max_chunk >= bytes_per_frame,
        "max_chunk too small for one frame"
    );

    // Make chunk size a multiple of frame size
    let effective = max_chunk - (max_chunk % bytes_per_frame);
    let mut chunks = Vec::with_capacity(buf.len().div_ceil(effective));

    let mut i = 0;
    while i < buf.len() {
        let end = (i + effective).min(buf.len());
        // Also trim tail to a frame boundary for the final chunk
        let end_aligned = end - (end - i) % bytes_per_frame;
        if end_aligned == i {
            break; // should not happen unless buf tail is misaligned
        }
        chunks.push(buf[i..end_aligned].to_vec());
        i = end_aligned;
    }
    chunks
}

pub fn play_pcm_i16_mono_rodio(pcm_le_bytes: &[u8], sample_rate: u32) -> anyhow::Result<()> {
    // Write to a temporary WAV file
    let temp_path = std::env::temp_dir().join("temp_audio.wav");

    // Create WAV file
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(&temp_path, spec)?;

    // Convert bytes back to i16 samples and write
    for chunk in pcm_le_bytes.chunks_exact(2) {
        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
        writer.write_sample(sample)?;
    }
    writer.finalize()?;

    // Use system command to play the audio
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("afplay")
            .arg(&temp_path)
            .output()?;

        if !output.status.success() {
            anyhow::bail!(
                "Failed to play audio with afplay: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    #[cfg(target_os = "linux")]
    {
        let output = std::process::Command::new("aplay")
            .arg(&temp_path)
            .output()?;

        if !output.status.success() {
            anyhow::bail!(
                "Failed to play audio with aplay: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    #[cfg(target_os = "windows")]
    {
        let output = std::process::Command::new("powershell")
            .args(&[
                "-c",
                &format!(
                    "(New-Object Media.SoundPlayer '{}'').PlaySync()",
                    temp_path.display()
                ),
            ])
            .output()?;

        if !output.status.success() {
            anyhow::bail!(
                "Failed to play audio with PowerShell: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    // Clean up temp file
    let _ = std::fs::remove_file(&temp_path);

    Ok(())
}

#[cfg(test)]
mod tests {
    use hound::{SampleFormat, WavSpec, WavWriter};
    use std::io::Write;
    use std::path::Path;
    use tempfile::NamedTempFile;

    fn write_wav_mono_16bit(
        path: &Path,
        sample_rate_hz: u32,
        duration_secs: f32,
        frequency_hz: f32,
        amplitude_i16: i16,
    ) {
        let spec = WavSpec {
            channels: 1,
            sample_rate: sample_rate_hz,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(path, spec).unwrap();
        let total_samples = (sample_rate_hz as f32 * duration_secs) as usize;

        for sample_index in 0..total_samples {
            let time_secs = sample_index as f32 / sample_rate_hz as f32;
            let sample_value = ((2.0 * std::f32::consts::PI * frequency_hz * time_secs).sin()
                * (amplitude_i16 as f32))
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16;

            writer.write_sample(sample_value).unwrap();
        }
        writer.finalize().unwrap();
    }

    fn write_wav_stereo_16bit(
        path: &Path,
        sample_rate_hz: u32,
        duration_secs: f32,
        left_frequency_hz: f32,
        right_frequency_hz: f32,
        amplitude_i16: i16,
    ) {
        let spec = WavSpec {
            channels: 2,
            sample_rate: sample_rate_hz,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(path, spec).unwrap();
        let total_samples = (sample_rate_hz as f32 * duration_secs) as usize;

        for sample_index in 0..total_samples {
            let time_secs = sample_index as f32 / sample_rate_hz as f32;

            let left_sample = ((2.0 * std::f32::consts::PI * left_frequency_hz * time_secs).sin()
                * (amplitude_i16 as f32))
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16;

            let right_sample = ((2.0 * std::f32::consts::PI * right_frequency_hz * time_secs).sin()
                * (amplitude_i16 as f32))
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16;

            writer.write_sample(left_sample).unwrap();
            writer.write_sample(right_sample).unwrap();
        }
        writer.finalize().unwrap();
    }

    #[test]
    fn decode_audio_mono_has_nonzero_bytes() {
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "").unwrap(); // ensure file exists
        let temp_path = temp_file.into_temp_path();

        write_wav_mono_16bit(&temp_path, 44_100, 0.20, 440.0, 8_000);

        let (pcm_bytes, sample_rate_hz, channel_count) =
            super::decode_audio(&temp_path.to_path_buf()).unwrap();

        assert_eq!(sample_rate_hz, 44_100);
        assert_eq!(channel_count, 1);

        assert!(!pcm_bytes.is_empty(), "PCM is empty");
        assert!(
            pcm_bytes
                .chunks_exact(2)
                .any(|two_bytes| two_bytes != [0, 0]),
            "all samples are zero"
        );
    }

    #[test]
    fn decode_audio_stereo_has_nonzero_bytes() {
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "").unwrap();
        let temp_path = temp_file.into_temp_path();

        write_wav_stereo_16bit(&temp_path, 48_000, 0.20, 440.0, 660.0, 7_000);

        let (pcm_bytes, sample_rate_hz, channel_count) =
            super::decode_audio(&temp_path.to_path_buf()).unwrap();

        assert_eq!(sample_rate_hz, 48_000);
        assert_eq!(channel_count, 2);

        // Interleaved i16 → at least some bytes non-zero
        assert!(
            pcm_bytes
                .chunks_exact(2)
                .any(|two_bytes| two_bytes != [0, 0]),
            "all samples are zero"
        );
    }

    #[test]
    fn chunk_pcm_alignment_and_sizes() {
        // Make a synthetic non-zero buffer: 2-channel, 16-bit → 4 bytes/frame
        let channel_count_u8 = 2u8;
        let bytes_per_frame = (channel_count_u8 as usize) * 2;
        let frame_count = 1234usize;

        let mut buffer = Vec::with_capacity(frame_count * bytes_per_frame);
        for byte_index in 0..(frame_count * bytes_per_frame) {
            buffer.push((byte_index % 251) as u8); // non-zero-ish content
        }

        // Choose a max_chunk_size that is not a multiple of frame size
        let max_chunk_size = 750usize; // not multiple of 4
        let chunks = super::chunk_pcm_aligned(&buffer, channel_count_u8, max_chunk_size);

        assert!(!chunks.is_empty());

        // All chunks must be multiples of bytes_per_frame
        for chunk in &chunks {
            assert_eq!(chunk.len() % bytes_per_frame, 0, "chunk not frame-aligned");
        }

        // Reassemble and ensure it equals the original
        let reassembled: Vec<u8> = chunks.iter().flat_map(|chunk| chunk.clone()).collect();
        assert_eq!(reassembled, buffer, "reassembled differs");
    }
}
