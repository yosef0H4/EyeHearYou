/** Type definitions for the OCR UI */

export interface Config {
    api_url: string;
    api_key: string;
    model: string;
    max_image_dimension: number;
    text_detection: TextDetectionConfig;
}

export interface TextDetectionConfig {
    min_confidence: number;
    min_width: number;
    min_height: number;
    merge_vertical_tolerance: number;
    merge_horizontal_tolerance: number;
    merge_width_ratio_threshold: number;
}

export interface BoundingBox {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
}

export type BoundingBoxTuple = [number, number, number, number];

export interface ScreenshotResponse {
    status: "success" | "error";
    image?: string;
    width?: number;
    height?: number;
    version?: number;
    message?: string;
}

export interface DetectionResponse {
    status: "success" | "error";
    regions?: BoundingBoxTuple[];
    message?: string;
}

export interface ExtractionResponse {
    status: "success" | "error";
    text?: string;
    message?: string;
}

export interface MergedBox {
    rect: BoundingBoxTuple;
    count: number;
    originalBoxes: BoundingBoxTuple[];
}

export interface ScreenshotEvent {
    version: number;
    width: number;
    height: number;
}

