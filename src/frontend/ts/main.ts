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
let autoSaveTimeout: number | null = null;  // For debouncing auto-save
let previewRefreshTimeout: number | null = null;  // For debouncing preview refresh from text inputs

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
        
        // Setup SSE for auto-updating screenshots and status
        sseManager = new SSEManager(async (event) => {
            if (event.type === "status") {
                updateStatusUI(event);
            } 
            else if (event.type === "screenshot" || event.version) {
                // Backward compatibility if backend sends old format
                if (imageViewer && event.version) {
                    await imageViewer.loadScreenshot();
                    // Auto-run detection and load extraction when new screenshot arrives
                    await autoRefreshOnNewScreenshot(event.version);
                }
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
    
    // Detection settings sliders
    const minConfInput = document.getElementById('min_confidence') as HTMLInputElement;
    const minWidthInput = document.getElementById('min_width') as HTMLInputElement;
    const minHeightInput = document.getElementById('min_height') as HTMLInputElement;
    const minWidthTextInput = document.getElementById('min_width_input') as HTMLInputElement;
    const minHeightTextInput = document.getElementById('min_height_input') as HTMLInputElement;
    
    // Update visualizers on input (no server call)
    if (minConfInput) {
        minConfInput.addEventListener('input', () => {
            updatePaddleViz();
            autoSaveConfig();  // Auto-save on change
        });
    }
    
    // Sync min width slider and text input
    if (minWidthInput && minWidthTextInput) {
        minWidthInput.addEventListener('input', () => {
            minWidthTextInput.value = minWidthInput.value;
            updatePaddleViz();
            autoSaveConfig();  // Auto-save on change
        });
        minWidthInput.addEventListener('change', () => {
            autoSaveConfig();  // Save on release
            refreshPreview(false);  // On release, refresh preview
        });
        minWidthInput.addEventListener('pointerup', () => {
            autoSaveConfig();  // Save on release
            refreshPreview(false);  // On release, refresh preview
        });
        minWidthTextInput.addEventListener('input', () => {
            const value = parseInt(minWidthTextInput.value) || 5;
            const minValue = Math.max(5, value);
            // Clamp slider to 300, but allow text input to go higher
            const sliderValue = Math.min(300, minValue);
            minWidthInput.value = String(sliderValue);
            // Don't clamp text input - allow values > 300
            if (minWidthTextInput.value !== String(minValue) && minValue >= 5) {
                minWidthTextInput.value = String(minValue);
            }
            updatePaddleViz();
            autoSaveConfig();  // Auto-save on change
            debouncedPreviewRefresh();  // Live preview update
        });
        minWidthTextInput.addEventListener('blur', () => {
            const value = parseInt(minWidthTextInput.value) || 5;
            const minValue = Math.max(5, value);
            // Clamp slider to 300, but allow text input to go higher
            const sliderValue = Math.min(300, minValue);
            minWidthInput.value = String(sliderValue);
            // Don't clamp text input - allow values > 300
            minWidthTextInput.value = String(minValue);
            updatePaddleViz();
            autoSaveConfig();  // Save on blur
            refreshPreview(false);  // On blur, refresh preview
        });
    }
    
    // Sync min height slider and text input
    if (minHeightInput && minHeightTextInput) {
        minHeightInput.addEventListener('input', () => {
            minHeightTextInput.value = minHeightInput.value;
            updatePaddleViz();
            autoSaveConfig();  // Auto-save on change
        });
        minHeightInput.addEventListener('change', () => {
            autoSaveConfig();  // Save on release
            refreshPreview(false);  // On release, refresh preview
        });
        minHeightInput.addEventListener('pointerup', () => {
            autoSaveConfig();  // Save on release
            refreshPreview(false);  // On release, refresh preview
        });
        minHeightTextInput.addEventListener('input', () => {
            const value = parseInt(minHeightTextInput.value) || 5;
            const minValue = Math.max(5, value);
            // Clamp slider to 300, but allow text input to go higher
            const sliderValue = Math.min(300, minValue);
            minHeightInput.value = String(sliderValue);
            // Don't clamp text input - allow values > 300
            if (minHeightTextInput.value !== String(minValue) && minValue >= 5) {
                minHeightTextInput.value = String(minValue);
            }
            updatePaddleViz();
            autoSaveConfig();  // Auto-save on change
            debouncedPreviewRefresh();  // Live preview update
        });
        minHeightTextInput.addEventListener('blur', () => {
            const value = parseInt(minHeightTextInput.value) || 5;
            const minValue = Math.max(5, value);
            // Clamp slider to 300, but allow text input to go higher
            const sliderValue = Math.min(300, minValue);
            minHeightInput.value = String(sliderValue);
            // Don't clamp text input - allow values > 300
            minHeightTextInput.value = String(minValue);
            updatePaddleViz();
            autoSaveConfig();  // Save on blur
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
            autoSaveConfig();  // Auto-save on change
        });
        vTolInput.addEventListener('change', () => {
            autoSaveConfig();  // Save on release
            refreshPreview(false);  // On release, refresh preview
        });
        vTolInput.addEventListener('pointerup', () => {
            autoSaveConfig();  // Save on release
            refreshPreview(false);  // On release, refresh preview
        });
        vTolTextInput.addEventListener('input', () => {
            const value = parseInt(vTolTextInput.value) || 0;
            const minValue = Math.max(0, value);
            // Clamp slider to 300, but allow text input to go higher
            const sliderValue = Math.min(300, minValue);
            vTolInput.value = String(sliderValue);
            // Don't clamp text input - allow values > 300
            if (vTolTextInput.value !== String(minValue) && minValue >= 0) {
                vTolTextInput.value = String(minValue);
            }
            updateMergeViz();
            autoSaveConfig();  // Auto-save on change
            debouncedPreviewRefresh();  // Live preview update
        });
        vTolTextInput.addEventListener('blur', () => {
            const value = parseInt(vTolTextInput.value) || 0;
            const minValue = Math.max(0, value);
            // Clamp slider to 300, but allow text input to go higher
            const sliderValue = Math.min(300, minValue);
            vTolInput.value = String(sliderValue);
            // Don't clamp text input - allow values > 300
            vTolTextInput.value = String(minValue);
            updateMergeViz();
            autoSaveConfig();  // Save on blur
            refreshPreview(false);  // On blur, refresh preview
        });
    }
    
    // Sync horizontal tolerance slider and text input
    if (hTolInput && hTolTextInput) {
        hTolInput.addEventListener('input', () => {
            hTolTextInput.value = hTolInput.value;
            updateMergeViz();
            autoSaveConfig();  // Auto-save on change
        });
        hTolInput.addEventListener('change', () => {
            autoSaveConfig();  // Save on release
            refreshPreview(false);  // On release, refresh preview
        });
        hTolInput.addEventListener('pointerup', () => {
            autoSaveConfig();  // Save on release
            refreshPreview(false);  // On release, refresh preview
        });
        hTolTextInput.addEventListener('input', () => {
            const value = parseInt(hTolTextInput.value) || 0;
            const minValue = Math.max(0, value);
            // Clamp slider to 300, but allow text input to go higher
            const sliderValue = Math.min(300, minValue);
            hTolInput.value = String(sliderValue);
            // Don't clamp text input - allow values > 300
            if (hTolTextInput.value !== String(minValue) && minValue >= 0) {
                hTolTextInput.value = String(minValue);
            }
            updateMergeViz();
            autoSaveConfig();  // Auto-save on change
            debouncedPreviewRefresh();  // Live preview update
        });
        hTolTextInput.addEventListener('blur', () => {
            const value = parseInt(hTolTextInput.value) || 0;
            const minValue = Math.max(0, value);
            // Clamp slider to 300, but allow text input to go higher
            const sliderValue = Math.min(300, minValue);
            hTolInput.value = String(sliderValue);
            // Don't clamp text input - allow values > 300
            hTolTextInput.value = String(minValue);
            updateMergeViz();
            autoSaveConfig();  // Save on blur
            refreshPreview(false);  // On blur, refresh preview
        });
    }
    
    if (wRatioInput) {
        wRatioInput.addEventListener('input', () => {
            updateMergeViz();
            autoSaveConfig();  // Auto-save on change
        });
        wRatioInput.addEventListener('change', () => {
            autoSaveConfig();  // Save on release
            refreshPreview(false);  // On release, refresh preview
        });
        wRatioInput.addEventListener('pointerup', () => {
            autoSaveConfig();  // Save on release
            refreshPreview(false);  // On release, refresh preview
        });
    }
    
    // Min confidence also triggers preview refresh on release
    if (minConfInput) {
        minConfInput.addEventListener('change', () => {
            autoSaveConfig();  // Save on release
            refreshPreview(false);  // On release, refresh preview (uses cached scores)
        });
        minConfInput.addEventListener('pointerup', () => {
            autoSaveConfig();  // Save on release
            refreshPreview(false);  // On release, refresh preview
        });
    }
    
    // API settings also auto-save
    const apiUrlInput = document.getElementById('api_url') as HTMLInputElement;
    const apiKeyInput = document.getElementById('api_key') as HTMLInputElement;
    const modelInput = document.getElementById('model') as HTMLInputElement;
    
    if (apiUrlInput) {
        apiUrlInput.addEventListener('blur', () => autoSaveConfig());
        apiUrlInput.addEventListener('change', () => autoSaveConfig());
    }
    if (apiKeyInput) {
        apiKeyInput.addEventListener('blur', () => autoSaveConfig());
        apiKeyInput.addEventListener('change', () => autoSaveConfig());
    }
    if (modelInput) {
        modelInput.addEventListener('blur', () => autoSaveConfig());
        modelInput.addEventListener('change', () => autoSaveConfig());
    }
    
    // Action buttons
    const captureBtn = document.getElementById('capture-btn');
    const detectBtn = document.getElementById('detect-btn');
    const extractBtn = document.getElementById('extract-btn');
    const cancelBtn = document.getElementById('cancel-btn');
    
    if (captureBtn) captureBtn.addEventListener('click', handleCapture);
    if (detectBtn) detectBtn.addEventListener('click', handleDetection);
    if (extractBtn) extractBtn.addEventListener('click', handleExtraction);
    if (cancelBtn) cancelBtn.addEventListener('click', handleCancel);
    
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
        if (outputText && !success) {
            outputText.innerText = '❌ Error saving config';
        }
        // Don't show success message for auto-save to avoid spam
    } catch (e) {
        if (outputText) {
            outputText.innerText = '❌ Error saving config: ' + (e as Error).message;
        }
    }
}

/**
 * Auto-save config to config.json whenever settings change
 * Debounced to avoid saving on every input event
 */
function autoSaveConfig(): void {
    // Clear existing timeout
    if (autoSaveTimeout !== null) {
        clearTimeout(autoSaveTimeout);
    }
    
    // Debounce: save 500ms after last change
    autoSaveTimeout = window.setTimeout(async () => {
        try {
            const config = getConfigFromInputs();
            await saveConfig(config);
            console.log('Config auto-saved');
        } catch (e) {
            console.error('Error auto-saving config:', e);
        }
    }, 500);
}

/**
 * Debounced preview refresh for text inputs
 * Updates preview after user stops typing for a moment
 */
function debouncedPreviewRefresh(): void {
    // Clear existing timeout
    if (previewRefreshTimeout !== null) {
        clearTimeout(previewRefreshTimeout);
    }
    
    // Debounce: refresh 800ms after last change (slightly longer than auto-save)
    previewRefreshTimeout = window.setTimeout(() => {
        refreshPreview(false);
    }, 800);
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
    if (outputText) outputText.innerText = "🔍 Detecting text regions (RapidOCR)... This may take a few seconds...";
    
    // Run detection (this will cache raw detections with scores)
    await refreshPreview(true);
}

/**
 * Refresh preview by calling backend with current settings
 * @param shouldRunDetection - If true, run new RapidOCR detection. If false, use cached detections.
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

function updateStatusUI(event: any): void {
    const container = document.getElementById('status-container');
    const msgEl = document.getElementById('status-message');
    const barEl = document.getElementById('progress-bar');
    const percentEl = document.getElementById('status-percent');
    const extractBtn = document.getElementById('extract-btn') as HTMLButtonElement;
    const captureBtn = document.getElementById('capture-btn') as HTMLButtonElement;
    const detectBtn = document.getElementById('detect-btn') as HTMLButtonElement;

    if (!container || !msgEl || !barEl) return;

    if (event.isLoading) {
        container.style.display = 'block';
        msgEl.innerText = event.message || "Processing...";
        
        if (event.progress !== undefined) {
            barEl.style.width = `${event.progress}%`;
            if (percentEl) percentEl.innerText = `${Math.round(event.progress)}%`;
        }
        
        // Disable buttons while processing
        if (extractBtn) extractBtn.disabled = true;
        if (captureBtn) captureBtn.disabled = true;
        if (detectBtn) detectBtn.disabled = true;
    } else {
        // Finished or Cancelled
        msgEl.innerText = event.message || "Ready";
        if (percentEl) percentEl.innerText = "";
        
        // Short delay before hiding if successful
        setTimeout(() => {
            container.style.display = 'none';
            barEl.style.width = '0%';
        }, 2000);

        // Re-enable buttons
        if (extractBtn) extractBtn.disabled = false;
        if (captureBtn) captureBtn.disabled = false;
        if (detectBtn) detectBtn.disabled = false;
        
        // Refresh extraction text if finished
        if (event.message && (event.message.includes("Copied") || event.message.includes("Extracted"))) {
            loadLastExtraction();
        }
    }
}

async function handleCancel(): Promise<void> {
    try {
        await fetch('/cancel', { method: 'POST' });
        const msgEl = document.getElementById('status-message');
        if (msgEl) msgEl.innerText = "Cancelling...";
    } catch (e) {
        console.error("Error canceling:", e);
    }
}

async function loadLastExtraction(): Promise<void> {
    try {
        const res = await fetch('/last_extraction');
        const data = await res.json();
        const outputText = document.getElementById('output-text');
        if (data.status === "success" && data.text && outputText) {
            outputText.innerText = data.text;
        }
    } catch (e) {
        console.error(e);
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

