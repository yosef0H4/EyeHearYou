/** UI rendering and interaction */
import { BoundingBoxTuple, MergedBox, ScreenshotResponse } from './types.js';
import { mergeCloseTextBoxes } from './merge.js';

export class ImageViewer {
    private imageNaturalWidth: number = 0;
    private imageNaturalHeight: number = 0;
    private imageDisplayWidth: number = 0;
    private imageDisplayHeight: number = 0;
    private scaleX: number = 1;
    private scaleY: number = 1;
    private rawDetections: BoundingBoxTuple[] = [];

    constructor() {
        this.setupResizeHandler();
    }

    private setupResizeHandler(): void {
        let resizeTimeout: number;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = window.setTimeout(() => {
                this.updateScales();
                if (this.rawDetections.length > 0) {
                    this.drawRawBoxes();
                    this.runLiveMerge();
                }
            }, 250);
        });
    }

    private updateScales(): void {
        const img = document.getElementById('preview-img') as HTMLImageElement;
        if (img && img.complete && this.imageNaturalWidth > 0) {
            this.imageDisplayWidth = img.clientWidth;
            this.imageDisplayHeight = img.clientHeight;
            this.scaleX = this.imageDisplayWidth / this.imageNaturalWidth;
            this.scaleY = this.imageDisplayHeight / this.imageNaturalHeight;
            this.updateOverlaySizes();
        }
    }

    async loadScreenshot(): Promise<void> {
        try {
            const res = await fetch('/screenshot');
            const data: ScreenshotResponse = await res.json();
            
            if (data.status === "success" && data.image) {
                const img = document.getElementById('preview-img') as HTMLImageElement;
                const wrapper = document.getElementById('image-wrapper') as HTMLElement;
                const placeholder = document.getElementById('placeholder') as HTMLElement;
                
                if (img && wrapper && placeholder) {
                    img.src = "data:image/png;base64," + data.image;
                    
                    img.onload = () => {
                        wrapper.style.display = 'block';
                        placeholder.style.display = 'none';
                        
                        this.imageNaturalWidth = img.naturalWidth;
                        this.imageNaturalHeight = img.naturalHeight;
                        this.imageDisplayWidth = img.clientWidth;
                        this.imageDisplayHeight = img.clientHeight;
                        
                        this.scaleX = this.imageDisplayWidth / this.imageNaturalWidth;
                        this.scaleY = this.imageDisplayHeight / this.imageNaturalHeight;
                        
                        this.updateOverlaySizes();
                        this.clearBoxes();
                        
                        const outputText = document.getElementById('output-text') as HTMLElement;
                        if (outputText) {
                            outputText.innerText = `✅ Screenshot updated (version ${data.version || 0}). Ready to detect text regions.`;
                        }
                    };
                }
            }
        } catch (e) {
            console.error('Error loading screenshot:', e);
        }
    }

    private updateOverlaySizes(): void {
        const overlayRaw = document.getElementById('overlay-raw') as HTMLElement;
        const overlayTolerance = document.getElementById('overlay-tolerance') as HTMLElement;
        const overlayMerged = document.getElementById('overlay-merged') as HTMLElement;
        
        if (overlayRaw) {
            overlayRaw.style.width = this.imageDisplayWidth + 'px';
            overlayRaw.style.height = this.imageDisplayHeight + 'px';
        }
        if (overlayTolerance) {
            overlayTolerance.style.width = this.imageDisplayWidth + 'px';
            overlayTolerance.style.height = this.imageDisplayHeight + 'px';
        }
        if (overlayMerged) {
            overlayMerged.style.width = this.imageDisplayWidth + 'px';
            overlayMerged.style.height = this.imageDisplayHeight + 'px';
        }
    }

    setRawDetections(detections: BoundingBoxTuple[]): void {
        this.rawDetections = detections;
        this.drawRawBoxes();
        this.runLiveMerge();
    }

    private drawRawBoxes(): void {
        const container = document.getElementById('overlay-raw');
        if (!container) return;
        
        container.innerHTML = '';
        
        this.rawDetections.forEach((rect) => {
            const [x1, y1, x2, y2] = rect;
            const div = document.createElement('div');
            div.className = 'box box-raw';
            div.style.left = (x1 * this.scaleX) + 'px';
            div.style.top = (y1 * this.scaleY) + 'px';
            div.style.width = ((x2 - x1) * this.scaleX) + 'px';
            div.style.height = ((y2 - y1) * this.scaleY) + 'px';
            container.appendChild(div);
        });
    }

    runLiveMerge(): void {
        if (!this.rawDetections || this.rawDetections.length === 0) return;

        const vTolInput = document.getElementById('v_tol') as HTMLInputElement;
        const hTolInput = document.getElementById('h_tol') as HTMLInputElement;
        const wRatioInput = document.getElementById('w_ratio') as HTMLInputElement;

        const v_tol = parseInt(vTolInput?.value || '30');
        const h_tol = parseInt(hTolInput?.value || '50');
        const ratio_thresh = parseFloat(wRatioInput?.value || '0.3');

        const merged = mergeCloseTextBoxes(this.rawDetections, v_tol, h_tol, ratio_thresh);
        this.drawMergedBoxes(merged);
    }

    private drawMergedBoxes(mergedList: MergedBox[]): void {
        const containerMerged = document.getElementById('overlay-merged');
        const containerTolerance = document.getElementById('overlay-tolerance');
        
        if (containerMerged) containerMerged.innerHTML = '';
        if (containerTolerance) containerTolerance.innerHTML = '';

        const vTolInput = document.getElementById('v_tol') as HTMLInputElement;
        const hTolInput = document.getElementById('h_tol') as HTMLInputElement;
        const v_tol = parseInt(vTolInput?.value || '30');
        const h_tol = parseInt(hTolInput?.value || '50');

        mergedList.forEach((item) => {
            const [x1, y1, x2, y2] = item.rect;
            
            // Draw merged box (blue)
            if (containerMerged) {
                const div = document.createElement('div');
                div.className = 'box box-merged';
                div.style.left = (x1 * this.scaleX) + 'px';
                div.style.top = (y1 * this.scaleY) + 'px';
                div.style.width = ((x2 - x1) * this.scaleX) + 'px';
                div.style.height = ((y2 - y1) * this.scaleY) + 'px';
                
                if (item.count > 1) {
                    const label = document.createElement('div');
                    label.className = 'box-label';
                    label.innerText = String(item.count);
                    div.appendChild(label);
                }

                containerMerged.appendChild(div);
            }

            // Draw tolerance zones (yellow) for merged boxes with multiple original boxes
            if (item.count > 1 && item.originalBoxes && containerTolerance) {
                item.originalBoxes.forEach((origBox) => {
                    const [ox1, oy1, ox2, oy2] = origBox;
                    const tolDiv = document.createElement('div');
                    tolDiv.className = 'box box-tolerance';
                    tolDiv.style.left = ((ox1 - h_tol) * this.scaleX) + 'px';
                    tolDiv.style.top = ((oy1 - v_tol) * this.scaleY) + 'px';
                    tolDiv.style.width = ((ox2 - ox1 + 2 * h_tol) * this.scaleX) + 'px';
                    tolDiv.style.height = ((oy2 - oy1 + 2 * v_tol) * this.scaleY) + 'px';
                    containerTolerance.appendChild(tolDiv);
                });
            }
        });
    }

    private clearBoxes(): void {
        this.rawDetections = [];
        const overlayRaw = document.getElementById('overlay-raw');
        const overlayTolerance = document.getElementById('overlay-tolerance');
        const overlayMerged = document.getElementById('overlay-merged');
        
        if (overlayRaw) overlayRaw.innerHTML = '';
        if (overlayTolerance) overlayTolerance.innerHTML = '';
        if (overlayMerged) overlayMerged.innerHTML = '';
    }
}

export function updatePaddleViz(): void {
    const conf = parseFloat((document.getElementById('min_confidence') as HTMLInputElement)?.value || '0.6');
    const w = parseInt((document.getElementById('min_width') as HTMLInputElement)?.value || '30');
    const h = parseInt((document.getElementById('min_height') as HTMLInputElement)?.value || '30');

    const valConf = document.getElementById('val_conf');
    const valWidth = document.getElementById('val_width');
    const valHeight = document.getElementById('val_height');
    
    if (valConf) valConf.innerText = conf.toFixed(2);
    if (valWidth) valWidth.innerText = String(w);
    if (valHeight) valHeight.innerText = String(h);

    const box = document.getElementById('viz-paddle-box') as HTMLElement;
    if (box) {
        box.style.opacity = String(conf);
        box.style.width = Math.max(20, Math.min(100, w)) + 'px';
        box.style.height = Math.max(20, Math.min(100, h)) + 'px';
    }
}

export function updateMergeViz(): void {
    const v = parseInt((document.getElementById('v_tol') as HTMLInputElement)?.value || '30');
    const h = parseInt((document.getElementById('h_tol') as HTMLInputElement)?.value || '50');
    const r = parseFloat((document.getElementById('w_ratio') as HTMLInputElement)?.value || '0.3');

    const valVTol = document.getElementById('val_vtol');
    const valHTol = document.getElementById('val_htol');
    const valRatio = document.getElementById('val_ratio');
    
    if (valVTol) valVTol.innerText = String(v);
    if (valHTol) valHTol.innerText = String(h);
    if (valRatio) valRatio.innerText = r.toFixed(2);

    const baseW = 50;
    const baseH = 25;
    const zone = document.getElementById('viz-merge-zone') as HTMLElement;
    
    if (zone) {
        const scale = 0.4;
        zone.style.width = (baseW + (h * 2 * scale)) + 'px';
        zone.style.height = (baseH + (v * 2 * scale)) + 'px';
    }
}

