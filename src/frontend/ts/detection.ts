/** Text detection functionality */
import { DetectionResponse, BoundingBoxTuple } from './types.js';

export async function runDetection(
    minConfidence: number,
    minWidth: number,
    minHeight: number
): Promise<BoundingBoxTuple[]> {
    const settings = {
        min_confidence: minConfidence,
        min_width: minWidth,
        min_height: minHeight
    };

    const res = await fetch('/detect_preview', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ text_detection: settings })
    });
    
    const data: DetectionResponse = await res.json();
    
    if (data.status === "success" && data.regions) {
        return data.regions;
    } else {
        throw new Error(data.message || "Detection failed");
    }
}

