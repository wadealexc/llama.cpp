#!/usr/bin/env node

const fs = await import('fs');

/**
 * Simple test script to interact with llama-server
 * 
 * Usage:
 *   node test-server.mjs
 * 
 * Assumes server is running on http://localhost:8090
 */

const BASE_URL = 'http://localhost:8090';
// const BASE_URL = 'http://192.168.87.30:8070'; // KITSU
const SLOT_IDX = 0;
const BEFORE_FNAME = `slot${SLOT_IDX}-before.state`;
const AFTER_FNAME = `slot${SLOT_IDX}-after.state`;

const SLOT_SAVE_PATH = './slot-saves/';

// Check for --props flag
const args = process.argv.slice(2);
const SHOW_PROPS_ONLY = args.includes('--props');
const READ_FILE_INFO_ONLY = args.includes('--read');
const PRINT_TENSOR_INFO = args.includes('--tensors');

async function getProps() {
    console.log('\n=== GET /props ===');
    try {
        const response = await fetch(`${BASE_URL}/props`);
        const data = await response.json();
        console.log(JSON.stringify(data, null, 2));
        return data;
    } catch (error) {
        console.error('Error fetching props:', error.message);
        return null;
    }
}

async function getSlots() {
    console.log('\n=== GET /slots ===');
    try {
        const response = await fetch(`${BASE_URL}/slots`);
        const data = await response.json();
        console.log(JSON.stringify(data, null, 2));
        return data;
    } catch (error) {
        console.error('Error fetching slots:', error.message);
        return null;
    }
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

async function saveSlot(id, filename, action = 'save') {
    console.log(`\n=== POST /slots/${id}?action=${action} ===`);
    console.log(`Filename: ${filename}`);

    try {
        const response = await fetch(`${BASE_URL}/slots/${id}?action=${action}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                filename: filename,
            }),
        });

        const data = await response.json();

        // Handle both save and restore responses
        if (action === 'save') {
            console.log(`Slot save successful:`);
            console.log(`  - Tokens saved: ${data.n_saved}`);
            console.log(`  - File size: ${formatBytes(data.n_written)}`);
            console.log(`  - Time taken: ${data.timings.save_ms.toFixed(2)} ms`);
        } else {
            console.log(`Slot restore successful:`);
            console.log(`  - Tokens restored: ${data.n_restored}`);
            console.log(`  - File size: ${formatBytes(data.n_read)}`);
            console.log(`  - Time taken: ${data.timings.restore_ms.toFixed(2)} ms`);
        }
        return data;
    } catch (error) {
        console.error(`Error ${action}ing slot ${id}:`, error.message);
        return null;
    }
}

async function sendCompletion(prompt, id_slot = -1) {
    console.log('\n=== POST /completion ===');
    console.log('Prompt:', prompt);
    console.log('Target slot:', id_slot === -1 ? 'auto (any idle)' : id_slot);

    try {
        const response = await fetch(`${BASE_URL}/completion`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: prompt,
                stream: false,
                id_slot: id_slot,  // Assign to specific slot (-1 = auto)
                n_predict: 128,    // Generate up to 128 tokens
                temperature: 0.7,
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('\n--- Assistant Response ---');
        console.log(data.content);
        console.log('--- End Response ---');
        console.log('Processed by slot:', data.id_slot);
        console.log('Tokens generated:', data.tokens_predicted);
        console.log('Tokens evaluated:', data.tokens_evaluated);
        console.log('');

        return data;
    } catch (error) {
        console.error('Error sending completion:', error.message);
        return null;
    }
}

function ggmlTypeToString(ggmlType) {
    const typeMap = {
        0:  'F32',
        1:  'F16',
        2:  'Q4_0',
        3:  'Q4_1',
        6:  'Q5_0',
        7:  'Q5_1',
        8:  'Q8_0',
        9:  'Q8_1',
        10: 'Q2_K',
        11: 'Q3_K',
        12: 'Q4_K',
        13: 'Q5_K',
        14: 'Q6_K',
        15: 'Q8_K',
        16: 'IQ2_XXS',
        17: 'IQ2_XS',
        18: 'IQ3_XXS',
        19: 'IQ1_S',
        20: 'IQ4_NL',
        21: 'IQ3_S',
        22: 'IQ2_S',
        23: 'IQ4_XS',
        24: 'I8',
        25: 'I16',
        26: 'I32',
        27: 'I64',
        28: 'F64',
        29: 'IQ1_M',
        30: 'BF16',
        34: 'TQ1_0',
        35: 'TQ2_0',
        39: 'MXFP4',
        40: 'NVFP4',
        41: 'Q1_0',
    };

    return typeMap[ggmlType] || 'UNKNOWN';
}

function readSlotFile(file) {
    const path = SLOT_SAVE_PATH + file;
    console.log(`Reading slot file: ${file} at path ${path}`);

    // 1. Check file exists, console log if it doesn't
    if (!fs.existsSync(path)) {
        console.log(`File does not exist: ${path}`);
        return;
    }

    // 2. Read magic, version
    const buffer = fs.readFileSync(path);
    const magic = buffer.readUInt32LE(0);
    const version = buffer.readUInt32LE(4);

    // 3. Read prompt from file
    const nTokenCount = buffer.readUInt32LE(8);
    const tokens = [];
    for (let i = 0; i < nTokenCount; i++) {
        const offset = 12 + (i * 4);
        tokens.push(buffer.readInt32LE(offset));
    }

    // 4. Read number of streams (state_read)
    let offset = 12 + (nTokenCount * 4);
    const numStreams = buffer.readUInt32LE(offset);
    offset += 4;

    // Log magic, version, and nTokenCount. Don't do anything to `tokens` for now.
    console.log(`Magic: 0x${magic.toString(16)}`);
    console.log(`Version: ${version}`);
    console.log(`Token count: ${nTokenCount}`);
    console.log(`File size: ${formatBytes(buffer.length)} (${buffer.length} bytes)`);
    console.log(`Streams: ${numStreams}`);

    // 5. Read each stream's metadata and data
    for (let i = 0; i < numStreams; i++) {
        // state_read: number of cells in this stream
        const cellCount = buffer.readUInt32LE(offset);
        offset += 4;

        console.log(` - stream ${i} has cells: ${cellCount}`);
        if (cellCount == 0) continue;

        // 6. state_read_meta
        let prevNSeqId = -1;
        let prevSeqId = -1;
        let prevPos = -1;
        for (let j = 0; j < cellCount; j++) {
            const pos = buffer.readInt32LE(offset);
            offset += 4;
            const numSeqId = buffer.readUInt32LE(offset);
            offset += 4;

            // NOTE:
            // If we were using Qwen3.5, we would read llama_kv_cell_ext here
            // However, for this script, we're using Qwen3, which uses rope type NEOX.
            // llama_kv_cell_ext has an x and y - both int32 that encode a spatial pos
            // used for M-RoPE.

            // Read sequence id
            const seqId = buffer.readInt32LE(offset);
            offset += 4;

            // Log first few entries, then only log if we get unexpected values
            if (
                j >= 5 &&
                pos === prevPos + 1 &&
                numSeqId === prevNSeqId &&
                seqId === prevSeqId
            ) {
                // Expected values; don't log
            } else {
                console.log(` --- st ${i} | cell ${j}: pos ${pos} | nSeqId ${numSeqId} | seqId ${seqId}`);
            }

            prevPos = pos;
            prevNSeqId = numSeqId;
            prevSeqId = seqId;
        }

        // 7. state_read_data
        //
        // vTrans indicates whether the value tensor is stored in transposed layout
        // This is controlled by flash attention (-fa). If fa is set, v_trans is false.
        // Since fa is on by default, we'll assume it's set.
        const vTrans = buffer.readUInt32LE(offset) !== 0;
        const nLayers = buffer.readUInt32LE(offset + 4);
        offset += 8;

        console.log(` - Value tensor in transposed layout? ${vTrans ? 'yes' : 'no'}`);
        console.log(` - N Layers: ${nLayers}`);

        // Read K rows
        const kOffsetStart = offset;
        for (let j = 0; j < nLayers; j++) {
            // Read ggml_type
            const keyType = buffer.readInt32LE(offset);
            offset += 4;

            // Read row size of key
            const keyRowSize = buffer.readBigUInt64LE(offset);
            offset += 8;

            let keyTypeStr = ggmlTypeToString(keyType);
            if (keyTypeStr === "UNKNOWN") keyTypeStr += `(${keyType})`;
            
            // Assume contiguous cells; calculate tensor size and increment offset
            const tensorSize = BigInt(cellCount) * keyRowSize;
            offset += Number(tensorSize);

            if (PRINT_TENSOR_INFO)
                console.log(` --- layer ${j} | kType: ${keyTypeStr} | rowSize: ${keyRowSize} | tensorSize: ${formatBytes(Number(tensorSize))}`);
        }
        
        const kTotalSize = offset - kOffsetStart;
        console.log(` - Total K tensor size: ${formatBytes(kTotalSize)}`);

        // Read V rows
        const vOffsetStart = offset;
        for (let j = 0; j < nLayers; j++) {
            // Read ggml_type
            const vType = buffer.readInt32LE(offset);
            offset += 4;

            // Read row size of value
            const vRowSize = buffer.readBigUInt64LE(offset);
            offset += 8;

            let vTypeStr = ggmlTypeToString(vType);
            if (vTypeStr === "UNKNOWN") vTypeStr += `(${vType})`;

            // Assume contiguous cells; calculate tensor size and increment offset
            const tensorSize = BigInt(cellCount) * vRowSize;
            offset += Number(tensorSize);

            if (PRINT_TENSOR_INFO)
                console.log(` --- layer ${j} | vType: ${vTypeStr} | rowSize: ${vRowSize} | tensorSize: ${formatBytes(Number(tensorSize))}`);
        }

        const vTotalSize = offset - vOffsetStart;
        console.log(` - Total V tensor size: ${formatBytes(vTotalSize)}`);
    }
}

async function main() {
    console.log('Testing llama-server on', BASE_URL);
    console.log('='.repeat(50));

    // If --props flag is set, just show props and exit
    if (SHOW_PROPS_ONLY) {
        await getProps();
        return;
    } else if (READ_FILE_INFO_ONLY) {
        readSlotFile(BEFORE_FNAME);
        readSlotFile(AFTER_FNAME);
        return;
    }

    // Step 1: Get initial slot state
    await getSlots();

    // Step 2: Save slot state (before)
    await saveSlot(SLOT_IDX, BEFORE_FNAME, 'save');

    // Step 3: Send completion request (targeting SLOT_IDX)
    const prompt = 'hi how are you';
    await sendCompletion(prompt, SLOT_IDX);

    // Step 4: Get slot state after completion
    await getSlots();

    // Step 5: Save slot state (after)
    await saveSlot(SLOT_IDX, AFTER_FNAME, 'save');

    console.log('\n' + '='.repeat(50));
    console.log('Test complete!');
    console.log('\nSaved files:');
    console.log(`  - ${BEFORE_FNAME} (binary KV cache)`);
    console.log(`  - ${AFTER_FNAME} (binary KV cache)`);
}

main().catch(console.error);
