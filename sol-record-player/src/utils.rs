use anyhow::{anyhow, Result};
use solana_client::{
    nonblocking::rpc_client::RpcClient, rpc_client::GetConfirmedSignaturesForAddress2Config,
    rpc_response::RpcConfirmedTransactionStatusWithSignature,
};
use solana_commitment_config::CommitmentConfig;
use solana_transaction_status::{
    EncodedConfirmedTransactionWithStatusMeta, EncodedTransaction, UiCompiledInstruction,
    UiInstruction, UiMessage, UiParsedInstruction, UiTransactionEncoding,
};
use std::{path::PathBuf, str::FromStr, time::Duration};
use tokio::time::sleep;

use solana_sdk::{
    bs58,
    instruction::{AccountMeta, Instruction},
    pubkey::Pubkey,
    signature::{read_keypair_file, Keypair, Signature},
    transaction::Transaction,
};

const SYSTEM_PROGRAM_ID: Pubkey = Pubkey::from_str_const("11111111111111111111111111111111");

pub fn load_keypair(maybe_path: Option<&PathBuf>) -> Result<Keypair> {
    let path = if let Some(p) = maybe_path {
        p.clone()
    } else {
        // Default Solana CLI keypair location
        dirs::home_dir()
            .map(|path| path.join(".config/solana/id.json"))
            .ok_or_else(|| anyhow!("could not resolve home directory for default keypair"))?
    };
    let payer_keypair =
        read_keypair_file(path).map_err(|e| anyhow!("failed to read keypair file: {e}"))?;
    Ok(payer_keypair)
}

fn pack_name(name: Option<&str>) -> [u8; 32] {
    let mut out = [0u8; 32];
    if let Some(s) = name {
        let b = s.as_bytes();
        let n = b.len().min(32);
        out[..n].copy_from_slice(&b[..n]);
    }
    out
}

/// Packs the exact SolRecordIxData layout (no discriminator).
#[allow(clippy::too_many_arguments)]
pub fn build_write_ix(
    program_id: Pubkey,
    payer: Pubkey,
    record_pda: Pubkey,
    sample_rate: u32,
    channels: u8,
    audio_id: [u8; 8],
    name: Option<&str>,
    chunk_idx: u32,
    total_chunks: u32,
    is_final: bool,
    data_bytes: &[u8],
) -> Instruction {
    // Pre-size roughly to avoid reallocs
    let mut data = Vec::with_capacity(4 + 1 + 8 + 32 + 4 + 4 + 1 + data_bytes.len());

    // sample_rate [4, LE]
    data.extend_from_slice(&sample_rate.to_le_bytes());
    // channels [1]
    data.push(channels);
    // audio_id [8]
    data.extend_from_slice(&audio_id);
    // name [NAME_CAP]
    data.extend_from_slice(&pack_name(name));
    // chunk_idx [4, LE]
    data.extend_from_slice(&chunk_idx.to_le_bytes());
    // total_chunks [4, LE]
    data.extend_from_slice(&total_chunks.to_le_bytes());
    // is_final [1]
    data.push(u8::from(is_final));
    // data [...]
    data.extend_from_slice(data_bytes);

    Instruction {
        program_id,
        accounts: vec![
            AccountMeta::new(payer, true),
            AccountMeta::new(record_pda, false),
            AccountMeta::new_readonly(SYSTEM_PROGRAM_ID, false),
        ],
        data,
    }
}

/// Wait-and-confirm helper (basic).
pub async fn send_and_confirm_with_latest_blockhash(
    rpc: &RpcClient,
    tx: &mut Transaction,
    signers: &[&Keypair],
) -> Result<Signature, anyhow::Error> {
    let recent_blockhash = rpc.get_latest_blockhash().await?;
    tx.try_sign(signers, recent_blockhash)?;

    let res = rpc.send_and_confirm_transaction(tx).await;
    match res {
        Err(e) => Err(anyhow!("Transaction failed: {}", e)),
        Ok(sig) => Ok(sig),
    }
}

/// Sends+confirms one tx, retrying on transient errors, keeping order.
/// Optional throttle_after_ok_ms lets you add a tiny delay only after success.
pub async fn send_confirm_sequential_one(
    rpc: &RpcClient,
    tx: &mut Transaction,
    signers: &[&Keypair],
    throttle_after_ok_ms: u64, // 0 = no throttle
) -> Result<Signature> {
    let mut attempts = 0usize;

    loop {
        attempts += 1;

        // Always refresh blockhash before (re)signing
        let (bh, _) = rpc
            .get_latest_blockhash_with_commitment(CommitmentConfig::processed())
            .await?;
        tx.try_sign(signers, bh)?;

        match rpc.send_and_confirm_transaction(tx).await {
            Ok(sig) => {
                if throttle_after_ok_ms > 0 {
                    sleep(Duration::from_millis(throttle_after_ok_ms)).await;
                }
                return Ok(sig);
            }
            Err(e) => {
                let msg = e.to_string();

                // Adaptive backoff based on common transient errors
                // (string matching is pragmatic; you can refine by error types if you prefer)
                let backoff_ms = if msg.contains("429")
                    || msg.contains("TooManyRequests")
                    || msg.contains("Rate limit")
                {
                    150 * attempts as u64
                } else if msg.contains("BlockhashNotFound") || msg.contains("expired blockhash") {
                    // Just loop and re-sign with a fresh blockhash; tiny sleep to avoid tight spin
                    40
                } else if msg.contains("AccountInUse")
                    || msg.contains("node is behind")
                    || msg.contains("Transaction was discarded")
                {
                    80 * attempts as u64
                } else {
                    // Unknown/transient: small exponential backoff
                    75 * attempts as u64
                };

                if attempts >= 6 {
                    // Give up after a few tries; surface the error
                    return Err(anyhow::anyhow!(e));
                }

                sleep(Duration::from_millis(backoff_ms.min(800))).await;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParsedChunk {
    pub sample_rate: u32,
    pub channels: u8,
    pub audio_id: [u8; 8],
    pub name: [u8; 32],
    pub chunk_idx: u32,
    pub total_chunks: u32,
    pub is_final: bool,
    pub data: Vec<u8>,
}

pub fn parse_write_ix_bytes(ix: &[u8]) -> anyhow::Result<ParsedChunk> {
    // header = 54 bytes
    if ix.len() < 54 {
        anyhow::bail!("ix too short: {}", ix.len());
    }
    let mut off = 0usize;
    let mut take = |n: usize| {
        let s = &ix[off..off + n];
        off += n;
        s
    };

    let mut sr4 = [0u8; 4];
    sr4.copy_from_slice(take(4));
    let sample_rate = u32::from_le_bytes(sr4);

    let channels = take(1)[0];

    let mut audio_id = [0u8; 8];
    audio_id.copy_from_slice(take(8));

    let mut name = [0u8; 32];
    name.copy_from_slice(take(32));

    let mut ci4 = [0u8; 4];
    ci4.copy_from_slice(take(4));
    let chunk_idx = u32::from_le_bytes(ci4);

    let mut tc4 = [0u8; 4];
    tc4.copy_from_slice(take(4));
    let total_chunks = u32::from_le_bytes(tc4);

    let is_final = take(1)[0] != 0;

    let data = ix[off..].to_vec();

    Ok(ParsedChunk {
        sample_rate,
        channels,
        audio_id,
        name,
        chunk_idx,
        total_chunks,
        is_final,
        data,
    })
}

pub fn extract_program_ix_for_pda(
    tx: &EncodedConfirmedTransactionWithStatusMeta,
    program_id: &Pubkey,
    pda: &Pubkey,
) -> Option<Vec<u8>> {
    let EncodedTransaction::Json(json_tx) = &tx.transaction.transaction else {
        return None;
    };

    match &json_tx.message {
        UiMessage::Raw(raw) => {
            let keys = &raw.account_keys;

            let pid_str = program_id.to_string();
            let prog_idx = keys.iter().position(|k| k == &pid_str)?;

            for ix in &raw.instructions {
                if ix.program_id_index as usize != prog_idx {
                    continue;
                }
                // This is check should always pass (defensive)
                let has_pda = ix.accounts.iter().any(|&acct_idx| {
                    keys.get(acct_idx as usize)
                        .and_then(|s| Pubkey::from_str(s).ok())
                        .map(|pk| pk == *pda)
                        .unwrap_or(false)
                });
                if !has_pda {
                    continue;
                }

                if let Ok(bytes) = bs58::decode(&ix.data).into_vec() {
                    return Some(bytes);
                }
            }
            None
        }
        UiMessage::Parsed(parsed) => {
            // account keys are objects with .pubkey field
            let key_strs: Vec<&str> = parsed
                .account_keys
                .iter()
                .map(|k| k.pubkey.as_str())
                .collect();

            for ui_ix in &parsed.instructions {
                match ui_ix {
                    // Compiled inside a Parsed message: same as Raw but through parsed.account_keys
                    UiInstruction::Compiled(UiCompiledInstruction {
                        program_id_index,
                        accounts,
                        data,
                        stack_height: _,
                    }) => {
                        let Some(prog_key) = prog_key_at(*program_id_index, &key_strs) else {
                            continue;
                        };
                        if &prog_key != program_id {
                            continue;
                        }
                        // PDA present?
                        let has_pda = accounts.iter().any(|&acct_idx| {
                            key_strs
                                .get(acct_idx as usize)
                                .and_then(|s| Pubkey::from_str(s).ok())
                                .map(|pk| pk == *pda)
                                .unwrap_or(false)
                        });
                        if !has_pda {
                            continue;
                        }

                        if let Ok(bytes) = bs58::decode(data).into_vec() {
                            return Some(bytes);
                        }
                    }
                    UiInstruction::Parsed(UiParsedInstruction::PartiallyDecoded(
                        partially_decoded,
                    )) => {
                        let Ok(prog_key) = Pubkey::from_str(&partially_decoded.program_id) else {
                            continue;
                        };
                        if &prog_key != program_id {
                            continue;
                        }

                        let has_pda = partially_decoded.accounts.iter().any(|acc| {
                            Pubkey::from_str(acc)
                                .map(|pubkey| pubkey == *pda)
                                .unwrap_or(false)
                        });
                        if !has_pda {
                            continue;
                        }

                        if let Ok(bytes) = bs58::decode(&partially_decoded.data).into_vec() {
                            return Some(bytes);
                        }
                    }

                    // Fully parsed instructions do not expose raw bytes — skip
                    UiInstruction::Parsed(UiParsedInstruction::Parsed(_)) => continue,
                }
            }
            None
        }
    }
}

pub async fn fetch_signatures_desc(
    rpc: &RpcClient,
    addr: &Pubkey,
    before: Option<Signature>,
    limit: usize,
) -> anyhow::Result<Vec<RpcConfirmedTransactionStatusWithSignature>> {
    let cfg = GetConfirmedSignaturesForAddress2Config {
        before,
        until: None,
        limit: Some(limit.min(1000)),
        ..GetConfirmedSignaturesForAddress2Config::default()
    };
    let sigs = rpc
        .get_signatures_for_address_with_config(addr, cfg)
        .await?;
    Ok(sigs)
}

pub async fn fetch_tx(
    rpc: &RpcClient,
    sig: &Signature,
) -> anyhow::Result<EncodedConfirmedTransactionWithStatusMeta> {
    let cfg = UiTransactionEncoding::Json;
    let tx = rpc.get_transaction(sig, cfg).await?;
    Ok(tx)
}

pub async fn fetch_tx_retry(
    rpc: &RpcClient,
    sig: &Signature,
) -> anyhow::Result<EncodedConfirmedTransactionWithStatusMeta> {
    let attempts = 3usize;
    for attempt in 1..=attempts {
        match fetch_tx(rpc, sig).await {
            Ok(tx) => return Ok(tx),
            Err(e) if attempt < attempts => {
                // exponential backoff
                let ms = (500 * attempt as u64).min(400);
                tracing::debug!(
                    "get_transaction({sig}) failed (attempt {attempt}/{attempts}): {e}; retrying in {ms}ms"
                );
                tokio::time::sleep(Duration::from_millis(ms)).await;
                continue;
            }
            Err(e) => {
                tracing::warn!("get_transaction({sig}) failed after {attempts} attempts: {e}");
                return Err(e);
            }
        }
    }
    unreachable!()
}

pub fn prog_key_at(idx: u8, key_strs: &Vec<&str>) -> Option<Pubkey> {
    key_strs
        .get(idx as usize)
        .and_then(|s| Pubkey::from_str(s).ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hex::FromHex;

    #[test]
    fn build_write_ix_layout_is_correct() {
        let program_id = Pubkey::new_unique();
        let payer = Pubkey::new_unique();
        let pda = Pubkey::new_unique();

        let sample_rate = 44_100u32;
        let channels = 2u8;
        let audio_id = <[u8; 8]>::from_hex("0102030405060708").unwrap();
        let name = Some("Test Name");
        let chunk_idx = 3u32;
        let total_chunks = 10u32;
        let is_final = false;
        let payload: Vec<u8> = (0u8..32).collect(); // 0..31

        let ix = build_write_ix(
            program_id,
            payer,
            pda,
            sample_rate,
            channels,
            audio_id,
            name,
            chunk_idx,
            total_chunks,
            is_final,
            &payload,
        );

        // Accounts order
        assert_eq!(ix.accounts.len(), 3);
        assert_eq!(ix.accounts[0].pubkey, payer);
        assert!(ix.accounts[0].is_signer);
        assert!(ix.accounts[0].is_writable);
        assert_eq!(ix.accounts[1].pubkey, pda);
        assert!(ix.accounts[1].is_writable);
        assert!(!ix.accounts[1].is_signer);
        assert_eq!(ix.accounts[2].pubkey, super::SYSTEM_PROGRAM_ID);

        // Data layout: 54 header + payload
        assert_eq!(ix.data.len(), 54 + payload.len());
        let d = &ix.data;

        assert_eq!(&d[0..4], &sample_rate.to_le_bytes());
        assert_eq!(d[4], channels);
        assert_eq!(&d[5..13], &audio_id);

        // name (padded / truncated to 32)
        let mut expect_name = [0u8; 32];
        let name_bytes = name.unwrap().as_bytes();
        expect_name[..name_bytes.len()].copy_from_slice(name_bytes);
        assert_eq!(&d[13..45], &expect_name);

        assert_eq!(&d[45..49], &chunk_idx.to_le_bytes());
        assert_eq!(&d[49..53], &total_chunks.to_le_bytes());
        assert_eq!(d[53], 0); // is_final
        assert_eq!(&d[54..], &payload[..]);
    }
}
