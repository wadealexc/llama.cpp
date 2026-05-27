#!/usr/bin/env node

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

// Check for --props flag
const args = process.argv.slice(2);
const SHOW_PROPS_ONLY = args.includes('--props');

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

async function main() {
    console.log('Testing llama-server on', BASE_URL);
    console.log('='.repeat(50));

    // If --props flag is set, just show props and exit
    if (SHOW_PROPS_ONLY) {
        await getProps();
        return;
    }

    const BEFORE_FNAME = `slot${SLOT_IDX}-before.state`;
    const AFTER_FNAME = `slot${SLOT_IDX}-after.state`;
    
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
