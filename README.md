# 🎵 Sol-Aud(io)

**Sol-Aud(io)** is a proof-of-concept Solana dApp that uploads and plays back
**on-chain audio** — one chunked transaction at a time.  
It uses a Pinocchio program and a Rust CLI for uploading and playback.

> ⚠️ Disclaimer This project is a proof of concept (POC) meant to demonstrate
> the feasibility of on-chain audio storage and playback on Solana. It is not
> optimized for production use . There are far more efficient approaches, for
> examle using Audio compression before chunking. For the best results during
> testing, try using short audio clips (up to ~10 seconds) to keep upload and
> playback smooth. Expect slower performance and higher transaction counts for
> longer files. Please take in consideration RPC limits when uploading and
> fetching a file.

---

## Architecture

| Component            | Description                                                                                                 |
| -------------------- | ----------------------------------------------------------------------------------------------------------- |
| `sol-record/`        | Solana on-chain program (Pinocchio). Stores audio chunks in PDA accounts.                                   |
| `sol-record-player/` | CLI client to upload (`upload`) or play (`play`) audio. Uses Symphonia for decoding and Rodio for playback. |

---

## Uploading Audio

1. Build the CLI:

   ```bash
   cargo build -p sol-record-player --release
   ```

2. Run upload (replace values accordingly):

```bash
RUST_LOG=info ./target/release/sol-record-player upload \
  --input ./sound.wav \
  --name "My First Track" \
  --keypair ./keypair.json \
  --rpc-url "https://devnet.YOUR_RPC.com/?api-key=YOUR_KEY" \
  --max-chunk 750
```

This will:

- Decode your audio to raw PCM.
- Split it into 750-byte aligned chunks.
- Upload each chunk as a Solana transaction.
- Derive a deterministic PDA for the track.

3. Playing Audio

Once uploaded, you can fetch and play a record directly from its PDA.

PDA key of a record uploaded: FnGWo9cs4ZXZ5gpvEBj38Tyd5tihuCn5ehYvzT1JsjRK

```bash
./target/release/sol-record-player play \
  --pda 2d15eQs9j9JVgDQYWFGg91CH28KxBrrpwxNXqTH4pQTd \
  --rpc-url "https://devnet.helius-rpc.com/?api-key=fc421f09-346e-447f-92e6-045a45a55301"
```

The CLI will:

- Stream chunk transactions from chain.
- Reassemble and verify by audio_id.
- Start playback once the buffer is ready.

## Notes

- Uploads are sequential to preserve order.
- Each transaction stores exactly one chunk in the PDA.
