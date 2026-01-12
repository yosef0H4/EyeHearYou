/** Configuration management */
import { Config, TextDetectionConfig } from './types.js';

let globalConfig: Config | null = null;

export async function loadConfig(): Promise<Config> {
    try {
        const res = await fetch('/config');
        const config = await res.json() as Config;
        globalConfig = config;
        return config;
    } catch (e) {
        console.error('Error loading config:', e);
        throw e;
    }
}

export function getConfig(): Config | null {
    return globalConfig;
}

export async function saveConfig(config: Config): Promise<boolean> {
    try {
        const res = await fetch('/config', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(config)
        });
        const data = await res.json();
        if (data.status === "success") {
            globalConfig = config;
            return true;
        }
        return false;
    } catch (e) {
        console.error('Error saving config:', e);
        return false;
    }
}

export function populateConfigInputs(config: Config): void {
    const apiUrlInput = document.getElementById('api_url') as HTMLInputElement;
    const apiKeyInput = document.getElementById('api_key') as HTMLInputElement;
    const modelInput = document.getElementById('model') as HTMLInputElement;
    
    if (apiUrlInput) apiUrlInput.value = config.api_url || '';
    if (apiKeyInput) apiKeyInput.value = config.api_key || '';
    if (modelInput) modelInput.value = config.model || '';
    
    const td = config.text_detection || {} as TextDetectionConfig;
    const minConfInput = document.getElementById('min_confidence') as HTMLInputElement;
    const minWidthInput = document.getElementById('min_width') as HTMLInputElement;
    const minHeightInput = document.getElementById('min_height') as HTMLInputElement;
    const minWidthTextInput = document.getElementById('min_width_input') as HTMLInputElement;
    const minHeightTextInput = document.getElementById('min_height_input') as HTMLInputElement;
    const vTolInput = document.getElementById('v_tol') as HTMLInputElement;
    const hTolInput = document.getElementById('h_tol') as HTMLInputElement;
    const vTolTextInput = document.getElementById('v_tol_input') as HTMLInputElement;
    const hTolTextInput = document.getElementById('h_tol_input') as HTMLInputElement;
    const wRatioInput = document.getElementById('w_ratio') as HTMLInputElement;
    
    if (minConfInput) minConfInput.value = String(td.min_confidence || 0.6);
    if (minWidthInput) {
        minWidthInput.value = String(td.min_width || 30);
        if (minWidthTextInput) minWidthTextInput.value = String(td.min_width || 30);
    }
    if (minHeightInput) {
        minHeightInput.value = String(td.min_height || 30);
        if (minHeightTextInput) minHeightTextInput.value = String(td.min_height || 30);
    }
    if (vTolInput) {
        vTolInput.value = String(td.merge_vertical_tolerance || 30);
        if (vTolTextInput) vTolTextInput.value = String(td.merge_vertical_tolerance || 30);
    }
    if (hTolInput) {
        hTolInput.value = String(td.merge_horizontal_tolerance || 50);
        if (hTolTextInput) hTolTextInput.value = String(td.merge_horizontal_tolerance || 50);
    }
    if (wRatioInput) wRatioInput.value = String(td.merge_width_ratio_threshold || 0.3);
}

export function getConfigFromInputs(): Config {
    const apiUrlInput = document.getElementById('api_url') as HTMLInputElement;
    const apiKeyInput = document.getElementById('api_key') as HTMLInputElement;
    const modelInput = document.getElementById('model') as HTMLInputElement;
    const minConfInput = document.getElementById('min_confidence') as HTMLInputElement;
    // Prefer text inputs if they exist, fall back to sliders
    const minWidthTextInput = document.getElementById('min_width_input') as HTMLInputElement;
    const minHeightTextInput = document.getElementById('min_height_input') as HTMLInputElement;
    const minWidthInput = document.getElementById('min_width') as HTMLInputElement;
    const minHeightInput = document.getElementById('min_height') as HTMLInputElement;
    const vTolTextInput = document.getElementById('v_tol_input') as HTMLInputElement;
    const hTolTextInput = document.getElementById('h_tol_input') as HTMLInputElement;
    const vTolInput = document.getElementById('v_tol') as HTMLInputElement;
    const hTolInput = document.getElementById('h_tol') as HTMLInputElement;
    const wRatioInput = document.getElementById('w_ratio') as HTMLInputElement;
    
    return {
        api_url: apiUrlInput?.value || '',
        api_key: apiKeyInput?.value || '',
        model: modelInput?.value || '',
        max_image_dimension: 1080,
        text_detection: {
            min_confidence: parseFloat(minConfInput?.value || '0.6'),
            min_width: parseInt(minWidthTextInput?.value || minWidthInput?.value || '30'),
            min_height: parseInt(minHeightTextInput?.value || minHeightInput?.value || '30'),
            merge_vertical_tolerance: parseInt(vTolTextInput?.value || vTolInput?.value || '30'),
            merge_horizontal_tolerance: parseInt(hTolTextInput?.value || hTolInput?.value || '50'),
            merge_width_ratio_threshold: parseFloat(wRatioInput?.value || '0.3')
        }
    };
}

