#![allow(unexpected_cfgs)]

use core::mem::size_of;
use pinocchio::{
    account_info::AccountInfo,
    instruction::{Seed, Signer},
    no_allocator, nostd_panic_handler, program_entrypoint,
    program_error::ProgramError,
    pubkey::{find_program_address, Pubkey},
    sysvars::{clock::Clock, rent::Rent, Sysvar},
    ProgramResult,
};
use pinocchio_system::instructions::CreateAccount;

/// Constants
const MAX_CHUNK: usize = 1000;
const NAME_CAP: usize = 32;

program_entrypoint!(process_instruction);
no_allocator!();
nostd_panic_handler!();

#[inline(always)]
fn process_instruction(
    _program_id: &Pubkey,
    accounts: &[AccountInfo],
    instruction_data: &[u8],
) -> ProgramResult {
    write_chunk(accounts, instruction_data)?;
    Ok(())
}

/// Write a chunk of data to the on-chain record
/// Accounts:
/// 0. [signer] The creator of the record
/// 1. [writable] The record account
/// 2. System program
///
/// Instruction Data:
/// - audio_id: [u8; 8] - Unique audio identifier
/// - sample_rate: u32 - Sample rate of the audio
/// - channels: u8 - Number of audio channels
/// - name: [u8; 32] - Name of the audio (UTF-8, truncated)
/// - chunk_idx: u32 - Index of the chunk being written (0-based)
/// - total_chunks: u32 - Total number of chunks
/// - data: [u8; N] - The chunk data (N <= max_chunk)
/// - is_final: u8 - 1 if this is the final chunk, 0 otherwise
fn write_chunk(accounts: &[AccountInfo], _instruction_data: &[u8]) -> Result<(), ProgramError> {
    let [account_info_creator, account_info_record, system_program] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };

    if !account_info_creator.is_signer() {
        return Err(ProgramError::MissingRequiredSignature);
    }

    if !account_info_record.is_writable() {
        return Err(ProgramError::InvalidAccountData);
    }

    let sol_record_ix_data = parse_instruction_data(_instruction_data)?;

    let (record_pda, record_bump) = find_program_address(
        &[
            b"sol_record",
            &sol_record_ix_data.audio_id[..],
            account_info_creator.key().as_ref(),
        ],
        &crate::id(),
    );

    if record_pda != *account_info_record.key() {
        return Err(ProgramError::InvalidSeeds);
    }

    // Check if account has been initialized
    if account_info_record.owner() == system_program.key() && account_info_record.lamports() == 0 {
        // If not initialized, create new account

        let bump = [record_bump];
        // Construct signer seeds for creating the Sol Record PDA with signature
        let record_signer_seeds = [
            Seed::from("sol_record".as_bytes()),
            Seed::from(&sol_record_ix_data.audio_id[..]),
            Seed::from(account_info_creator.key().as_ref()),
            Seed::from(&bump),
        ];
        let record_signers = [Signer::from(&record_signer_seeds[..])];

        CreateAccount {
            from: &account_info_creator,
            to: &account_info_record,
            lamports: Rent::get().unwrap().minimum_balance(ACCOUNT_DATA_LEN),
            space: ACCOUNT_DATA_LEN as u64,
            owner: &crate::id(),
        }
        .invoke_signed(&record_signers)?;

        let mut data_ref = account_info_record.try_borrow_mut_data()?;
        let state: &mut SolRecState = unsafe { &mut *(data_ref.as_mut_ptr() as *mut SolRecState) };

        // double check initialization values
        if state.initialized != 0 {
            return Err(SolRecordsError::AccountAlreadyInitialized.into());
        }

        if sol_record_ix_data.data.len() > MAX_CHUNK {
            return Err(SolRecordsError::DataTooLarge.into());
        }

        // Initialize the SolRecState in the account data
        state.initialized = 1;
        state.creator = *account_info_creator.key();
        state.name = sol_record_ix_data.name;
        state.sample_rate = sol_record_ix_data.sample_rate;
        state.channels = sol_record_ix_data.channels;
        state.audio_id = sol_record_ix_data.audio_id;
        state.chunk_idx = sol_record_ix_data.chunk_idx;
        state.total_chunks = sol_record_ix_data.total_chunks;
        state.uploaded = if sol_record_ix_data.is_final == 1 {
            1
        } else {
            0
        };
        state.last_updated_slot = Clock::get()?.slot.to_le_bytes();

        // write chunk data to buffer
        data_ref[HEADER_SIZE..HEADER_SIZE + sol_record_ix_data.data.len()]
            .copy_from_slice(sol_record_ix_data.data);

        Ok(())
    } else {
        if account_info_record.owner() != &crate::id() {
            return Err(ProgramError::IncorrectProgramId);
        }
        // Attempt to load existing state and error if not found
        let mut data_ref = account_info_record.try_borrow_mut_data()?;

        if data_ref.len() < ACCOUNT_DATA_LEN {
            return Err(ProgramError::AccountDataTooSmall);
        }

        // Initialize the SolRecState in the account data
        let (header_bytes, buf_bytes) = data_ref.split_at_mut(HEADER_SIZE);

        // Load account data
        let state: &mut SolRecState =
            unsafe { &mut *(header_bytes.as_mut_ptr() as *mut SolRecState) };

        // Validate instruction data against existing state
        validate_ix_data(state, &sol_record_ix_data)?;

        state.chunk_idx = sol_record_ix_data.chunk_idx;
        state.last_updated_slot = Clock::get()?.slot.to_le_bytes();
        if sol_record_ix_data.is_final == 1 {
            state.uploaded = 1u8;
        }

        // Write chunk data to buffer
        buf_bytes[0..sol_record_ix_data.data.len()].copy_from_slice(sol_record_ix_data.data);

        // Zero the tail (in case its the last chunk and smaller than MAX_CHUNK)
        for b in &mut buf_bytes[sol_record_ix_data.data.len()..] {
            *b = 0;
        }

        Ok(())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SolRecState {
    pub initialized: u8,
    pub creator: Pubkey,
    pub name: [u8; 32],
    pub sample_rate: [u8; 4],
    pub channels: u8,
    pub audio_id: [u8; 8],
    pub chunk_idx: [u8; 4],
    pub total_chunks: [u8; 4],
    pub uploaded: u8,
    pub last_updated_slot: [u8; 8],
    // followed by chunk buffer of MAX_CHUNK bytes
}

const HEADER_SIZE: usize = size_of::<SolRecState>();
const ACCOUNT_DATA_LEN: usize = HEADER_SIZE + MAX_CHUNK;

/// Parsed instruction data for writing a chunk
///
/// Many fields can be optional but will leave them all required for simplicity
pub struct SolRecordIxData<'a> {
    pub sample_rate: [u8; 4],
    pub channels: u8,
    pub audio_id: [u8; 8],
    pub name: [u8; NAME_CAP],
    pub chunk_idx: [u8; 4],
    pub total_chunks: [u8; 4],
    pub is_final: u8,
    pub data: &'a [u8],
}

#[inline(always)]
fn parse_instruction_data<'a>(ix: &'a [u8]) -> Result<SolRecordIxData<'a>, ProgramError> {
    if ix.len() < 54 {
        return Err(ProgramError::InvalidInstructionData);
    }

    let mut sample_rate = [0u8; 4];
    sample_rate.copy_from_slice(&ix[0..4]);

    let channels = ix[4];

    let mut audio_id = [0u8; 8];
    audio_id.copy_from_slice(&ix[5..13]);

    let mut name = [0u8; NAME_CAP];
    name.copy_from_slice(&ix[13..45]);

    let mut chunk_idx = [0u8; 4];
    chunk_idx.copy_from_slice(&ix[45..49]);
    let mut total_chunks = [0u8; 4];
    total_chunks.copy_from_slice(&ix[49..53]);

    let is_final = ix[53];
    let data = &ix[54..];

    if data.len() > MAX_CHUNK {
        return Err(ProgramError::InvalidInstructionData);
    }
    Ok(SolRecordIxData {
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

fn validate_ix_data(state: &SolRecState, ix: &SolRecordIxData) -> Result<(), ProgramError> {
    if state.audio_id != ix.audio_id {
        return Err(ProgramError::InvalidInstructionData);
    }
    if state.name != ix.name {
        return Err(ProgramError::InvalidInstructionData);
    }

    let ix_chunk_idx = le_u32(ix.chunk_idx);
    let state_chunk_total = le_u32(state.total_chunks);

    if ix_chunk_idx >= state_chunk_total {
        return Err(ProgramError::InvalidInstructionData);
    }

    if ix.is_final == 1 && ix_chunk_idx + 1 != state_chunk_total {
        return Err(ProgramError::InvalidInstructionData);
    }

    if ix.data.len() > MAX_CHUNK {
        return Err(ProgramError::InvalidInstructionData);
    }
    if state.uploaded != 0 {
        return Err(ProgramError::InvalidAccountData);
    }
    Ok(())
}

#[inline(always)]
fn le_u32(bytes: [u8; 4]) -> u32 {
    u32::from_le_bytes(bytes)
}

#[derive(Clone, PartialEq)]
pub enum SolRecordsError {
    // x01 Index > total_chunks
    IndexOutOfBounds,
    // x02 Data > MAX_CHUNK
    DataTooLarge,
    // x03 Invalid instruction data
    InvalidInstructionData,
    // x04 Account already initialized
    AccountAlreadyInitialized,
}

impl From<SolRecordsError> for ProgramError {
    fn from(e: SolRecordsError) -> Self {
        Self::Custom(e as u32)
    }
}
