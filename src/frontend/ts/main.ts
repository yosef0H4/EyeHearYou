/** Main application entry point */
import { loadConfig, saveConfig, populateConfigInputs, getConfigFromInputs } from './config.js';
import { SSEManager } from './sse.js';
import { runDetection } from './detection.js';
import { ImageViewer, updatePaddleViz, updateMergeViz } from './ui.js';
import { ExtractionResponse } from './types.js';

// Global instances
let imageViewer: ImageViewer | undefined;
let sseManager: SSEManager | undefined;
let previewAbortController: AbortController | null = null;  // For canceling in-flight preview requests

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
        sseManager = new SSEManager(async (event) => {
            if (imageViewer) {
                await imageViewer.loadScreenshot();
                // Auto-run detection and load extraction when new screenshot arrives
                await autoRefreshOnNewScreenshot(event.version);
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
    const minWidthTextInput = document.getElementById('min_width_input') as HTMLInputElement;
    const minHeightTextInput = document.getElementById('min_height_input') as HTMLInputElement;
    
    // Update visualizers on input (no server call)
    if (minConfInput) minConfInput.addEventListener('input', updatePaddleViz);
    
    // Sync min width slider and text input
    if (minWidthInput && minWidthTextInput) {
        minWidthInput.addEventListener('input', () => {
            minWidthTextInput.value = minWidthInput.value;
            updatePaddleViz();
        });
        minWidthInput.addEventListener('change', () => {
            refreshPreview(false);  // On release, refresh preview
        });
        minWidthInput.addEventListener('pointerup', () => {
            refreshPreview(false);  // On release, refresh preview
        });
        minWidthTextInput.addEventListener('input', () => {
            const value = parseInt(minWidthTextInput.value) || 5;
            const clampedValue = Math.max(5, Math.min(1000, value));
            minWidthInput.value = String(clampedValue);
            if (minWidthTextInput.value !== String(clampedValue)) {
                minWidthTextInput.value = String(clampedValue);
            }
            updatePaddleViz();
        });
        minWidthTextInput.addEventListener('blur', () => {
            const value = parseInt(minWidthTextInput.value) || 5;
            const clampedValue = Math.max(5, Math.min(1000, value));
            minWidthInput.value = String(clampedValue);
            minWidthTextInput.value = String(clampedValue);
            updatePaddleViz();
            refreshPreview(false);  // On blur, refresh preview
        });
    }
    
    // Sync min height slider and text input
    if (minHeightInput && minHeightTextInput) {
        minHeightInput.addEventListener('input', () => {
            minHeightTextInput.value = minHeightInput.value;
            updatePaddleViz();
        });
        minHeightInput.addEventListener('change', () => {
            refreshPreview(false);  // On release, refresh preview
        });
        minHeightInput.addEventListener('pointerup', () => {
            refreshPreview(false);  // On release, refresh preview
        });
        minHeightTextInput.addEventListener('input', () => {
            const value = parseInt(minHeightTextInput.value) || 5;
            const clampedValue = Math.max(5, Math.min(1000, value));
            minHeightInput.value = String(clampedValue);
            if (minHeightTextInput.value !== String(clampedValue)) {
                minHeightTextInput.value = String(clampedValue);
            }
            updatePaddleViz();
        });
        minHeightTextInput.addEventListener('blur', () => {
            const value = parseInt(minHeightTextInput.value) || 5;
            const clampedValue = Math.max(5, Math.min(1000, value));
            minHeightInput.value = String(clampedValue);
            minHeightTextInput.value = String(clampedValue);
            updatePaddleViz();
            refreshPreview(false);  // On blur, refresh preview
        });
    }
    
    // Merge settings sliders
    const vTolInput = document.getElementById('v_tol') as HTMLInputElement;
    const hTolInput = document.getElementById('h_tol') as HTMLInputElement;
    const wRatioInput = document.getElementById('w_ratio') as HTMLInputElement;
    const vTolTextInput = document.getElementById('v_tol_input') as HTMLInputElement;
    const hTolTextInput = document.getElementById('h_tol_input') as HTMLInputElement;
    
    // Sync vertical tolerance slider and text input
    if (vTolInput && vTolTextInput) {
        vTolInput.addEventListener('input', () => {
            vTolTextInput.value = vTolInput.value;
            updateMergeViz();
        });
        vTolInput.addEventListener('change', () => {
            refreshPreview(false);  // On release, refresh preview
        });
        vTolInput.addEventListener('pointerup', () => {
            refreshPreview(false);  // On release, refresh preview
        });
        vTolTextInput.addEventListener('input', () => {
            const value = parseInt(vTolTextInput.value) || 0;
            const clampedValue = Math.max(0, Math.min(1000, value));
            vTolInput.value = String(clampedValue);
            if (vTolTextInput.value !== String(clampedValue)) {
                vTolTextInput.value = String(clampedValue);
            }
            updateMergeViz();
        });
        vTolTextInput.addEventListener('blur', () => {
            const value = parseInt(vTolTextInput.value) || 0;
            const clampedValue = Math.max(0, Math.min(1000, value));
            vTolInput.value = String(clampedValue);
            vTolTextInput.value = String(clampedValue);
            updateMergeViz();
            refreshPreview(false);  // On blur, refresh preview
        });
    }
    
    // Sync horizontal tolerance slider and text input
    if (hTolInput && hTolTextInput) {
        hTolInput.addEventListener('input', () => {
            hTolTextInput.value = hTolInput.value;
            updateMergeViz();
        });
        hTolInput.addEventListener('change', () => {
            refreshPreview(false);  // On release, refresh preview
        });
        hTolInput.addEventListener('pointerup', () => {
            refreshPreview(false);  // On release, refresh preview
        });
        hTolTextInput.addEventListener('input', () => {
            const value = parseInt(hTolTextInput.value) || 0;
            const clampedValue = Math.max(0, Math.min(1000, value));
            hTolInput.value = String(clampedValue);
            if (hTolTextInput.value !== String(clampedValue)) {
                hTolTextInput.value = String(clampedValue);
            }
            updateMergeViz();
        });
        hTolTextInput.addEventListener('blur', () => {
            const value = parseInt(hTolTextInput.value) || 0;
            const clampedValue = Math.max(0, Math.min(1000, value));
            hTolInput.value = String(clampedValue);
            hTolTextInput.value = String(clampedValue);
            updateMergeViz();
            refreshPreview(false);  // On blur, refresh preview
        });
    }
    
    if (wRatioInput) {
        wRatioInput.addEventListener('input', () => {
            updateMergeViz();
        });
        wRatioInput.addEventListener('change', () => {
            refreshPreview(false);  // On release, refresh preview
        });
        wRatioInput.addEventListener('pointerup', () => {
            refreshPreview(false);  // On release, refresh preview
        });
    }
    
    // Min confidence also triggers preview refresh on release
    if (minConfInput) {
        minConfInput.addEventListener('change', () => {
            refreshPreview(false);  // On release, refresh preview (uses cached scores)
        });
        minConfInput.addEventListener('pointerup', () => {
            refreshPreview(false);  // On release, refresh preview
        });
    }
    
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
    
    // Run detection (this will cache raw detections with scores)
    await refreshPreview(true);
}

/**
 * Refresh preview by calling backend with current settings
 * @param shouldRunDetection - If true, run new Paddle detection. If false, use cached detections.
 */
async function refreshPreview(shouldRunDetection: boolean): Promise<void> {
    // Cancel any in-flight preview request
    if (previewAbortController) {
        previewAbortController.abort();
    }
    previewAbortController = new AbortController();
    
    try {
        const config = getConfigFromInputs();
        const result = await runDetection(config.text_detection, shouldRunDetection, previewAbortController.signal);
        
        if (imageViewer) {
            imageViewer.setDetectionResults(result.filtered, result.merged);
        }
        
        const outputText = document.getElementById('output-text') as HTMLElement;
        if (outputText) {
            if (shouldRunDetection) {
                outputText.innerText = `✅ Detected ${result.filtered.length} filtered regions. Adjust merge settings to group them.`;
            } else {
                outputText.innerText = `✅ Preview updated: ${result.filtered.length} filtered, ${result.merged.length} merged regions.`;
            }
        }
    } catch (e) {
        // Ignore aborted requests
        if ((e as Error).name === 'AbortError') {
            return;
        }
        const outputText = document.getElementById('output-text') as HTMLElement;
        if (outputText) {
            outputText.innerText = "❌ Error refreshing preview: " + (e as Error).message;
        }
        console.error('Error refreshing preview:', e);
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

/**
 * Automatically refresh detection and extraction when new screenshot arrives via hotkey
 */
async function autoRefreshOnNewScreenshot(version: number): Promise<void> {
    const outputText = document.getElementById('output-text') as HTMLElement;
    if (outputText) {
        outputText.innerText = "🔄 New screenshot detected. Running detection and OCR...";
    }
    
    try {
        // Run detection with current settings (uses cached if available, otherwise runs new)
        const config = getConfigFromInputs();
        const result = await runDetection(config.text_detection, false, previewAbortController?.signal);
        
        if (imageViewer) {
            imageViewer.setDetectionResults(result.filtered, result.merged);
        }
        
        // Check for last extraction (from hotkey)
        const extractionRes = await fetch('/last_extraction');
        const extractionData = await extractionRes.json();
        
        if (extractionData.status === "success" && extractionData.text && extractionData.version === version) {
            // Hotkey already ran extraction, just display it
            if (outputText) {
                outputText.innerText = extractionData.text;
            }
        } else {
            // No extraction yet, run it now
            await handleExtraction();
        }
    } catch (e) {
        const outputText = document.getElementById('output-text') as HTMLElement;
        if (outputText) {
            outputText.innerText = "❌ Error auto-refreshing: " + (e as Error).message;
        }
        console.error('Error auto-refreshing:', e);
    }
}

// Export for global access if needed (after initialization)
// These are set in the initialize() function

