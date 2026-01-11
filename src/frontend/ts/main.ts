/** Main application entry point */
import { loadConfig, saveConfig, populateConfigInputs, getConfigFromInputs } from './config.js';
import { SSEManager } from './sse.js';
import { runDetection } from './detection.js';
import { ImageViewer, updatePaddleViz, updateMergeViz } from './ui.js';
import { ExtractionResponse } from './types.js';

// Global instances
let imageViewer: ImageViewer | undefined;
let sseManager: SSEManager | undefined;

// Initialize on page load
window.addEventListener('DOMContentLoaded', async () => {
    await initialize();
});

async function initialize(): Promise<void> {
    try {
        // Load configuration
        await loadConfig();
        const config = await loadConfig();
        populateConfigInputs(config);
        
        // Initialize UI components
        imageViewer = new ImageViewer();
        updatePaddleViz();
        updateMergeViz();
        
        // Setup SSE for auto-updating screenshots
        sseManager = new SSEManager(async () => {
            if (imageViewer) {
                await imageViewer.loadScreenshot();
            }
        });
        sseManager.connect();
        
        // Setup event listeners
        setupEventListeners();
        
        // Export for global access if needed
        (window as any).imageViewer = imageViewer;
        (window as any).sseManager = sseManager;
        
    } catch (e) {
        console.error('Error initializing:', e);
        const outputText = document.getElementById('output-text');
        if (outputText) {
            outputText.innerText = 'Error loading config: ' + (e as Error).message;
        }
    }
}

function setupEventListeners(): void {
    // Config save button
    const saveConfigBtn = document.getElementById('save-config-btn');
    if (saveConfigBtn) {
        saveConfigBtn.addEventListener('click', handleSaveConfig);
    }
    
    // Paddle settings sliders
    const minConfInput = document.getElementById('min_confidence') as HTMLInputElement;
    const minWidthInput = document.getElementById('min_width') as HTMLInputElement;
    const minHeightInput = document.getElementById('min_height') as HTMLInputElement;
    
    if (minConfInput) minConfInput.addEventListener('input', updatePaddleViz);
    if (minWidthInput) {
        minWidthInput.addEventListener('input', () => {
            updatePaddleViz();
            // Live update size filtering
            if (imageViewer) imageViewer.updateSizeFilter();
        });
    }
    if (minHeightInput) {
        minHeightInput.addEventListener('input', () => {
            updatePaddleViz();
            // Live update size filtering
            if (imageViewer) imageViewer.updateSizeFilter();
        });
    }
    
    // Merge settings sliders
    const vTolInput = document.getElementById('v_tol') as HTMLInputElement;
    const hTolInput = document.getElementById('h_tol') as HTMLInputElement;
    const wRatioInput = document.getElementById('w_ratio') as HTMLInputElement;
    
    if (vTolInput) vTolInput.addEventListener('input', () => {
        updateMergeViz();
        if (imageViewer) imageViewer.runLiveMerge();
    });
    if (hTolInput) hTolInput.addEventListener('input', () => {
        updateMergeViz();
        if (imageViewer) imageViewer.runLiveMerge();
    });
    if (wRatioInput) wRatioInput.addEventListener('input', () => {
        updateMergeViz();
        if (imageViewer) imageViewer.runLiveMerge();
    });
    
    // Action buttons
    const captureBtn = document.getElementById('capture-btn');
    const detectBtn = document.getElementById('detect-btn');
    const extractBtn = document.getElementById('extract-btn');
    
    if (captureBtn) captureBtn.addEventListener('click', handleCapture);
    if (detectBtn) detectBtn.addEventListener('click', handleDetection);
    if (extractBtn) extractBtn.addEventListener('click', handleExtraction);
    
    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
        if (sseManager) {
            sseManager.disconnect();
        }
    });
}

async function handleSaveConfig(): Promise<void> {
    const outputText = document.getElementById('output-text') as HTMLElement;
    try {
        const config = getConfigFromInputs();
        const success = await saveConfig(config);
        if (outputText) {
            outputText.innerText = success 
                ? '✅ Config saved successfully!' 
                : '❌ Error saving config';
        }
    } catch (e) {
        if (outputText) {
            outputText.innerText = '❌ Error saving config: ' + (e as Error).message;
        }
    }
}

async function handleCapture(): Promise<void> {
    const outputText = document.getElementById('output-text') as HTMLElement;
    if (outputText) outputText.innerText = "📸 Capturing screenshot...";
    
    try {
        const res = await fetch('/capture', { method: 'POST' });
        const data = await res.json();
        
        if (data.status === "success") {
            if (sseManager) {
                sseManager.setCurrentVersion(data.version || 0);
            }
            if (imageViewer) {
                await imageViewer.loadScreenshot();
            }
        } else {
            if (outputText) outputText.innerText = "❌ Error: " + data.message;
        }
    } catch (e) {
        if (outputText) {
            outputText.innerText = "❌ Error capturing: " + (e as Error).message;
        }
    }
}

async function handleDetection(): Promise<void> {
    const outputText = document.getElementById('output-text') as HTMLElement;
    if (outputText) outputText.innerText = "🔍 Detecting text regions (PaddleOCR)... This may take a few seconds...";
    
    try {
        const minConfInput = document.getElementById('min_confidence') as HTMLInputElement;
        const minWidthInput = document.getElementById('min_width') as HTMLInputElement;
        const minHeightInput = document.getElementById('min_height') as HTMLInputElement;
        
        const regions = await runDetection(
            parseFloat(minConfInput?.value || '0.6'),
            parseInt(minWidthInput?.value || '30'),
            parseInt(minHeightInput?.value || '30')
        );
        
        if (imageViewer) {
            imageViewer.setRawDetections(regions);
        }
        if (outputText) {
            outputText.innerText = `✅ Detected ${regions.length} regions. Adjust merge settings to group them.`;
        }
    } catch (e) {
        if (outputText) {
            outputText.innerText = "❌ Error: " + (e as Error).message;
        }
    }
}

async function handleExtraction(): Promise<void> {
    const outputText = document.getElementById('output-text') as HTMLElement;
    
    // Save config first
    await handleSaveConfig();
    
    if (outputText) outputText.innerText = "🚀 Extracting text with AI... (this may take a moment)";
    
    try {
        const res = await fetch('/run_extraction', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({})
        });
        
        const data: ExtractionResponse = await res.json();
        if (outputText) {
            if (data.status === "success") {
                outputText.innerText = data.text || "No text extracted.";
            } else {
                outputText.innerText = "❌ Error: " + (data.message || "Unknown error");
            }
        }
    } catch (e) {
        if (outputText) {
            outputText.innerText = "❌ Error: " + (e as Error).message;
        }
    }
}

// Export for global access if needed (after initialization)
// These are set in the initialize() function

