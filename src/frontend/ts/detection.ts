/** Text detection functionality */
import { DetectionResponse, TextDetectionConfig } from './types.js';

export interface DetectionPreviewResult {
    raw: Array<{ bbox: [number, number, number, number], score: number }>;
    filtered: [number, number, number, number][];
    merged: Array<{
        rect: [number, number, number, number];
        count: number;
        originalBoxes: [number, number, number, number][];
    }>;
}

export async function runDetection(
    settings: TextDetectionConfig,
    runDetection: boolean = true,
    signal?: AbortSignal
): Promise<DetectionPreviewResult> {
    const res = await fetch('/detect_preview', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            text_detection: settings,
            run_detection: runDetection
        }),
        signal: signal
    });
    
    const data: DetectionResponse = await res.json();
    
    if (data.status === "success") {
        // New format
        if (data.raw !== undefined && data.filtered !== undefined && data.merged !== undefined) {
            return {
                raw: data.raw,
                filtered: data.filtered,
                merged: data.merged
            };
        }
        // Legacy format (for backward compatibility)
        if (data.regions) {
            return {
                raw: [],
                filtered: data.regions,
                merged: data.regions.map(rect => ({
                    rect,
                    count: 1,
                    originalBoxes: [rect]
                }))
            };
        }
        // Empty result
        return {
            raw: [],
            filtered: [],
            merged: []
        };
    } else {
        throw new Error(data.message || "Detection failed");
    }
}

