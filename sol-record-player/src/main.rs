pub mod audio;
pub mod utils;

use anyhow::{anyhow, Context, Result};
use clap::{Args, Parser, Subcommand};
use futures::future::join_all;
use hex::ToHex;
use solana_client::nonblocking::rpc_client::RpcClient;
use solana_commitment_config::CommitmentConfig;
use std::{
    collections::{hash_map, HashMap},
    io::Cursor,
    path::PathBuf,
    str::FromStr,
    time::Duration,
};
use tokio::time::sleep;
use tracing_subscriber::{fmt, EnvFilter};
use zstd::stream::{decode_all, encode_all};

use sha2::{Digest, Sha256};
use solana_sdk::{pubkey::Pubkey, signature::Signature, signer::Signer, transaction::Transaction};

use crate::{
    audio::{chunk_pcm_aligned, decode_audio, play_pcm_i16_mono_rodio},
    utils::{
        build_write_ix, extract_program_ix_for_pda, fetch_signatures_desc, fetch_tx_retry,
        load_keypair, parse_write_ix_bytes, send_and_confirm_with_latest_blockhash, ParsedChunk,
    },
};

const PROGRAM_ID: &str = "H2cBhhLFgQg9isScbr7wxL57AayzREEi5Jd5C1bj1J5D";

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let _ = fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse().unwrap()))
        .try_init();

    let cli = Cli::parse();
    match cli.command {
        Command::Upload(cmd) => cmd.run().await?,
        Command::Play(cmd) => cmd.run().await?,
    }
    Ok(())
}

/// Sol Records client
#[derive(Debug, Parser)]
#[command(name = "sol-records", author, version, about = "Upload and play on-chain audio chunks", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Load a file, chunk it, and send sequential transactions
    Upload(UploadCmd),
    /// Given a PDA, stream tx history/states and play as data arrives
    Play(PlayCmd),
}

#[derive(Debug, Args)]
struct UploadCmd {
    /// Input audio file (any format supported by symphonia; normalized to PCM)
    #[arg(short = 'i', long = "input", value_name = "FILE")]
    input: PathBuf,

    /// Display name to store in the account (truncated to 32 bytes)
    #[arg(short = 'n', long = "name")]
    name: Option<String>,

    /// RPC URL (e.g. https://api.devnet.solana.com)
    #[arg(long = "rpc-url", default_value = "https://api.devnet.solana.com")]
    rpc_url: String,

    /// Payer keypair path (JSON)
    #[arg(long = "keypair")]
    keypair: Option<PathBuf>,

    /// Max chunk size (bytes) for payload
    #[arg(long = "max-chunk", default_value_t = 1000)]
    max_chunk: usize,

    /// Commitment level
    #[arg(long = "commitment", default_value = "confirmed", value_parser = ["processed","confirmed","finalized"])]
    commitment: String,

    /// Dry run without sending transactions
    #[arg(long = "dry-run", default_value_t = false)]
    dry_run: bool,
}

impl UploadCmd {
    pub async fn run(self) -> Result<()> {
        // Decode input with symphonia -> PCM (mono, 16-bit, 44.1k ideally).
        let (pcm_bytes, sample_rate, channels) = decode_audio(&self.input)?;

        // Build chunk payloads (<= max_chunk).
        let chunks = chunk_pcm_aligned(&pcm_bytes, channels, self.max_chunk);
        tracing::info!(
            "chunks = {}, first lens = {:?}",
            chunks.len(),
            chunks.iter().take(5).map(|c| c.len()).collect::<Vec<_>>()
        );

        // Derive audio_id = first 8 bytes of sha256(file bytes).
        let mut hasher = Sha256::new();

        // since we alreaddy hace chunks in memory we can hash them directly
        for chunk in &chunks {
            hasher.update(chunk);
        }
        let hash_result = hasher.finalize();
        let audio_id: [u8; 8] = hash_result[0..8]
            .try_into()
            .expect("slice with incorrect length");

        let payer = load_keypair(self.keypair.as_ref())?;
        let creator_pubkey = payer.pubkey();

        let program_id = Pubkey::from_str(PROGRAM_ID)
            .map_err(|_| anyhow!("PROGRAM_ID is not a valid pubkey: {PROGRAM_ID}"))?;

        let seeds: &[&[u8]] = &[
            b"sol_record",
            &audio_id,               // fixed 8 bytes
            creator_pubkey.as_ref(), // 32 bytes
        ];
        let (record_pda, _bump) = Pubkey::find_program_address(seeds, &program_id);

        tracing::info!("creator  = {}", creator_pubkey);
        tracing::info!("audio_id = 0x{:02x?}", audio_id);
        tracing::info!("pda (client) = {}", record_pda);

        let rpc_url = if self.rpc_url.is_empty() {
            "https://api.devnet.solana.com".to_string()
        } else {
            self.rpc_url.clone()
        };
        let rpc = RpcClient::new_with_commitment(rpc_url.clone(), CommitmentConfig::confirmed());

        let compressed_chunks: Vec<Vec<u8>> = chunks
            .into_iter()
            .map(|c| encode_all(Cursor::new(c), 3).context("zstd compress chunk"))
            .collect::<Result<_, _>>()?;

        let total_chunks = compressed_chunks.len() as u32;
        let first_ix = build_write_ix(
            program_id,
            creator_pubkey,
            record_pda,
            sample_rate,
            channels,
            audio_id,
            self.name.as_deref(),
            0,
            total_chunks,
            total_chunks == 1,
            &compressed_chunks[0],
        );

        let mut first_tx = Transaction::new_with_payer(&[first_ix], Some(&creator_pubkey));
        let sig_0 = send_and_confirm_with_latest_blockhash(&rpc, &mut first_tx, &[&payer]).await?;
        tracing::info!("first chunk sent: {} (index 0/{})", sig_0, total_chunks - 1);

        for (i, payload) in compressed_chunks.iter().enumerate().skip(1) {
            let is_final = i == compressed_chunks.len() - 1;
            tracing::info!(
                "chunk {} len={}, head={}",
                i,
                payload.len(),
                payload.get(0..32).unwrap_or(&[]).encode_hex::<String>()
            );

            let mut attempt = 0;

            // Retry loop
            loop {
                attempt += 1;

                let ix = build_write_ix(
                    program_id,
                    creator_pubkey,
                    record_pda,
                    sample_rate,
                    channels,
                    audio_id,
                    self.name.as_deref(),
                    i as u32,
                    total_chunks,
                    is_final,
                    payload,
                );
                let mut tx = Transaction::new_with_payer(&[ix], Some(&creator_pubkey));
                match send_and_confirm_with_latest_blockhash(&rpc, &mut tx, &[&payer]).await {
                    Ok(sig) => {
                        tracing::info!(
                            "chunk {}/{} ok (attempt {}), sig={}",
                            i,
                            total_chunks - 1,
                            attempt,
                            sig
                        );
                        break;
                    }
                    Err(e) => {
                        if attempt >= 3 {
                            tracing::error!("chunk {} failed after {} attempts: {}", i, attempt, e);
                            // Error: abort the whole upload since we exceeded retries and we can proceed from where we left off
                            return Err(anyhow!(
                                "chunk {} failed after {} attempts: {}",
                                i,
                                attempt,
                                e
                            ));
                        }
                        let delay_ms = (75 * attempt as u64).min(500);
                        tracing::warn!(
                            "chunk {} send failed (attempt {}): {} — retrying in {}ms",
                            i,
                            attempt,
                            e,
                            delay_ms
                        );
                        sleep(Duration::from_millis(delay_ms)).await;
                    }
                }
            }
        }

        // Logging for tracking but we could save the manifest in to a JSON.
        tracing::info!(
            "Upload (stub) — input={:?}, name={}, pda={}, max_chunk={}, dry_run={}",
            self.input,
            self.name.unwrap_or("unnamed".to_string()),
            record_pda.to_string(),
            self.max_chunk,
            self.dry_run
        );
        Ok(())
    }
}

#[derive(Debug, Args)]
struct PlayCmd {
    /// Record PDA pubkey to stream from
    #[arg(short = 'k', long = "pda", value_name = "PUBKEY")]
    pda: String,

    /// RPC URL (e.g. https://api.devnet.solana.com)
    #[arg(long = "rpc-url", default_value = "https://api.devnet.solana.com")]
    rpc_url: String,

    /// Optional cache directory to persist fetched chunks between sessions
    #[arg(long = "cache-dir")]
    cache_dir: Option<PathBuf>,
}

impl PlayCmd {
    pub async fn run(self) -> Result<()> {
        let program_id = Pubkey::from_str(PROGRAM_ID).map_err(|_| anyhow!("PROGRAM_ID invalid"))?;
        let pda = Pubkey::from_str(&self.pda).map_err(|_| anyhow!("PDA is not a valid pubkey"))?;

        let rpc = RpcClient::new(self.rpc_url.clone());

        tracing::info!("Play pda={}", pda,);

        let mut before: Option<Signature> = None;
        let mut found_final: Option<ParsedChunk> = None;
        let mut total_chunks: Option<usize> = None;
        let mut expected_audio_id: Option<[u8; 8]> = None;

        let mut chunks_map: HashMap<u32, Vec<u8>> = HashMap::new();

        'paging: loop {
            let page = fetch_signatures_desc(&rpc, &pda, before, 1000).await?;
            if page.is_empty() {
                break;
            }

            let sigs: Vec<Signature> = page
                .iter()
                .filter_map(|s| Signature::from_str(&s.signature).ok())
                .collect();

            let mut tx_count = 0usize;

            // batch fetch 10 transactions at a time
            // increase if your RPC doesnt rate limit you
            for chunk_sigs in sigs.chunks(10) {
                let futures = chunk_sigs
                    .iter()
                    .map(|sig| fetch_tx_retry(&rpc, sig))
                    .collect::<Vec<_>>();

                let txs = join_all(futures).await;

                tx_count += txs.len();

                tracing::info!("fetched {} txs", tx_count);

                // For each fetched transaction, extract and parse the Sol Record write ix
                for (_sig, tx_result) in chunk_sigs.iter().zip(txs.into_iter()) {
                    let tx = match tx_result {
                        Ok(confirmed_tx) => confirmed_tx,
                        Err(_) => continue,
                    };

                    if let Some(ix_bytes) = extract_program_ix_for_pda(&tx, &program_id, &pda) {
                        if let Ok(parsed) = parse_write_ix_bytes(&ix_bytes) {
                            if let Some(id) = expected_audio_id {
                                if id != parsed.audio_id {
                                    return Err(anyhow!(
                                        "Audio file corrupted: mismatched audio_id in chunk"
                                    ));
                                }
                            } else {
                                expected_audio_id = Some(parsed.audio_id);
                            }

                            if let Some(total_chunk_size) = total_chunks {
                                if total_chunk_size != parsed.total_chunks as usize {
                                    return Err(anyhow!(
                                        "Audio file corrupted: Inconsistent total_chunks in stream"
                                    ));
                                }
                            } else {
                                total_chunks = Some(parsed.total_chunks as usize);
                            }

                            if let Some(total_chunk_size) = total_chunks {
                                let idx = parsed.chunk_idx as usize;
                                if idx >= total_chunk_size {
                                    continue;
                                }
                            }

                            // Insert chunk if not already present (we shouldn't see duplicates, but just in case)
                            if let hash_map::Entry::Vacant(e) = chunks_map.entry(parsed.chunk_idx) {
                                e.insert(parsed.data.clone());
                            }
                            if parsed.is_final {
                                found_final = Some(parsed.clone());
                            }

                            if let Some(chunks) = total_chunks {
                                // If we already have them all, we can stop right now
                                if chunks_map.len() == chunks {
                                    tracing::info!("Collected all {} chunks; stopping.", chunks);
                                    break 'paging;
                                }
                            }
                        }
                    }
                }
            }

            // Advance paging cursor.
            before = page
                .last()
                .and_then(|s| Signature::from_str(&s.signature).ok());
        }

        let final_chunk =
            found_final.ok_or_else(|| anyhow!("could not find final chunk for PDA {}", pda))?;
        let total_chunks_found =
            total_chunks.ok_or_else(|| anyhow!("final chunk missing total_chunks"))?;

        if chunks_map.len() < total_chunks_found {
            return Err(anyhow!(
                "incomplete recording: have {}/{} chunks",
                chunks_map.len(),
                total_chunks_found
            ));
        }

        // Build ordered PCM buffer
        let mut ordered_compressed_chunks = Vec::with_capacity(total_chunks_found);
        for i in 0..(total_chunks_found as u32) {
            if let Some(data) = chunks_map.remove(&i) {
                ordered_compressed_chunks.push(data);
            } else {
                return Err(anyhow!("Missing chunk: {}.", i));
            }
        }

        let mut pcm_bytes = Vec::new();
        for (i, chunk) in ordered_compressed_chunks.iter().enumerate() {
            let decompressed = decode_all(Cursor::new(&chunk))
                .with_context(|| format!("zstd decompress chunk {}", i))?;
            pcm_bytes.extend_from_slice(&decompressed);
        }

        let audio_name = String::from_utf8_lossy(&final_chunk.name)
            .trim_matches(char::from(0))
            .to_string();

        // Start playback once buffer_chunks ready.
        tracing::info!(
            "Reassembled PCM: {} bytes ({} KB). {} ready to stream/play.",
            pcm_bytes.len(),
            pcm_bytes.len() / 1024,
            audio_name
        );

        // Play via rodio
        play_pcm_i16_mono_rodio(&pcm_bytes, final_chunk.sample_rate)?;

        Ok(())
    }
}

#[cfg(test)]
mod real_file_tests {
    use std::path::{Path, PathBuf};

    use std::io::Cursor;
    use zstd::stream::{decode_all, encode_all};

    // Pull functions from the audio module explicitly.
    use crate::audio::{chunk_pcm_aligned, decode_audio};

    /// Resolve a file relative to the crate root (where Cargo.toml lives).
    fn root_file(rel: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
    }

    /// Set this to a file you actually have in the repo root, e.g. "sound.wav".
    /// You can also override with: `AUDIO_FIXTURE=./myfile.m4a cargo test -- --nocapture`
    fn fixture_path() -> PathBuf {
        if let Ok(p) = std::env::var("AUDIO_FIXTURE") {
            PathBuf::from(p)
        } else {
            root_file("sound2.wav")
        }
    }

    #[test]
    fn decode_and_chunk_real_file() {
        let path = fixture_path();
        assert!(
            path.exists(),
            "Test audio not found at {:?}. Put a file there or set AUDIO_FIXTURE=path",
            path
        );

        // Decode input with symphonia -> PCM bytes
        let (pcm_bytes, sample_rate, channels) = decode_audio(&path).expect("decode_audio failed");

        println!(
            "decoded: samples(bytes)={}, sample_rate={}Hz, channels={}",
            pcm_bytes.len(),
            sample_rate,
            channels
        );

        // Quick sanity: PCM not empty, and at least *some* bytes non-zero (depends on the file!)
        assert!(!pcm_bytes.is_empty(), "PCM output is empty");
        let any_nz = pcm_bytes.chunks_exact(2).any(|b| b != [0, 0]);
        println!("decoded_any_non_zero_sample={}", any_nz);

        // Chunk payloads (<= max_chunk), keeping frame alignment
        let max_chunk = 750usize;
        let chunks = chunk_pcm_aligned(&pcm_bytes, channels, max_chunk);

        // Mirror your runtime log
        let first_lens: Vec<usize> = chunks.iter().take(5).map(|c| c.len()).collect();
        println!("chunks = {}, first lens = {:?}", chunks.len(), first_lens);

        // Assertions: non-empty, each chunk <= max_chunk and aligned to frame size
        assert!(!chunks.is_empty(), "No chunks produced");
        let bytes_per_frame = (channels as usize) * 2; // i16 per channel
        for (i, c) in chunks.iter().enumerate() {
            assert!(
                c.len() <= max_chunk,
                "chunk[{i}] len {} exceeds max_chunk {}",
                c.len(),
                max_chunk
            );
            assert_eq!(
                c.len() % bytes_per_frame,
                0,
                "chunk[{i}] is not frame-aligned ({} % {} != 0)",
                c.len(),
                bytes_per_frame
            );
        }

        // Reassemble and compare exactly with input bytes (chunker must be lossless)
        let reassembled: Vec<u8> = chunks.concat();
        assert_eq!(
            reassembled, pcm_bytes,
            "Reassembled bytes differ from original PCM"
        );

        // Print first-non-zero frame indices for quick visual sanity
        let first_pcm_nz = pcm_bytes
            .chunks_exact(2)
            .position(|b| b != [0, 0])
            .unwrap_or(usize::MAX);
        println!("first_non_zero_frame_in_pcm={}", first_pcm_nz);

        for (i, c) in chunks.iter().take(3).enumerate() {
            let nz = c
                .chunks_exact(2)
                .position(|b| b != [0, 0])
                .unwrap_or(usize::MAX);
            println!("chunk[{i}] len={}, first_non_zero_frame={}", c.len(), nz);
        }
    }

    #[test]
    fn decode_chunk_and_roundtrip_zstd() {
        let path = fixture_path();
        assert!(
            path.exists(),
            "Test audio not found at {:?}. Put a file there or set AUDIO_FIXTURE=path",
            path
        );

        // Decode → PCM
        let (pcm_bytes, sample_rate, channels) = decode_audio(&path).expect("decode_audio failed");
        assert!(!pcm_bytes.is_empty(), "PCM output is empty");

        // Chunk PCM with frame alignment
        let max_chunk = 750usize;
        let chunks = chunk_pcm_aligned(&pcm_bytes, channels, max_chunk);
        assert!(!chunks.is_empty(), "No chunks produced");

        let bytes_per_frame = (channels as usize) * 2; // i16 per channel
        for (i, c) in chunks.iter().enumerate() {
            assert!(
                c.len() <= max_chunk,
                "chunk[{i}] len {} exceeds max_chunk {}",
                c.len(),
                max_chunk
            );
            assert_eq!(
                c.len() % bytes_per_frame,
                0,
                "chunk[{i}] not frame-aligned ({} % {} != 0)",
                c.len(),
                bytes_per_frame
            );
        }

        // Compress each chunk independently (zstd level 3)
        let compressed: Vec<Vec<u8>> = chunks
            .iter()
            .map(|c| encode_all(Cursor::new(c), 3).expect("zstd encode chunk"))
            .collect();

        // Decompress each chunk and reassemble
        let mut roundtrip_pcm = Vec::with_capacity(pcm_bytes.len());
        for (i, cc) in compressed.iter().enumerate() {
            let dec = decode_all(Cursor::new(cc)).expect("zstd decode chunk");
            assert_eq!(
                dec.len() % bytes_per_frame,
                0,
                "decompressed chunk[{i}] not frame-aligned"
            );
            roundtrip_pcm.extend_from_slice(&dec);
        }

        // Exact match with original PCM
        assert_eq!(
            roundtrip_pcm, pcm_bytes,
            "Round-trip (zstd per-chunk) differs from original PCM"
        );

        let total_compressed: usize = compressed.iter().map(|c| c.len()).sum();
        println!(
            "zstd per-chunk roundtrip ok — sample_rate={}Hz, channels={}, original={} bytes, compressed={} bytes ({:.2}x)",
            sample_rate,
            channels,
            pcm_bytes.len(),
            total_compressed,
            (pcm_bytes.len() as f64) / (total_compressed as f64)
        );
    }
}
