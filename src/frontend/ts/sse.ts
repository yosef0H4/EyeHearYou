/** Server-Sent Events for screenshot updates */
import { ScreenshotEvent } from './types.js';

export class SSEManager {
    private eventSource: EventSource | null = null;
    private currentVersion: number = 0;
    private onScreenshotUpdate: (event: ScreenshotEvent) => Promise<void>;

    constructor(onScreenshotUpdate: (event: ScreenshotEvent) => Promise<void>) {
        this.onScreenshotUpdate = onScreenshotUpdate;
    }

    connect(): void {
        if (this.eventSource) {
            this.eventSource.close();
        }
        
        this.eventSource = new EventSource('/screenshot/events');
        
        this.eventSource.onmessage = async (event) => {
            try {
                const data: ScreenshotEvent = JSON.parse(event.data);
                if (data.version && data.version > this.currentVersion) {
                    this.currentVersion = data.version;
                    await this.onScreenshotUpdate(data);
                }
            } catch (e) {
                console.error('Error processing SSE event:', e);
            }
        };
        
        this.eventSource.onerror = (error) => {
            console.error('SSE connection error:', error);
            // Try to reconnect after a delay
            setTimeout(() => {
                if (this.eventSource && this.eventSource.readyState === EventSource.CLOSED) {
                    this.connect();
                }
            }, 3000);
        };
    }

    disconnect(): void {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
    }

    getCurrentVersion(): number {
        return this.currentVersion;
    }

    setCurrentVersion(version: number): void {
        this.currentVersion = version;
    }
}

