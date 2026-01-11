/** Merge logic for text boxes (ported from Python) */
import { BoundingBoxTuple, MergedBox } from './types.js';

export function mergeCloseTextBoxes(
    textRegions: BoundingBoxTuple[],
    verticalTolerance: number,
    horizontalTolerance: number,
    widthRatioThreshold: number
): MergedBox[] {
    if (!textRegions || textRegions.length === 0) {
        return [];
    }

    // Deep copy detections
    const boxes = JSON.parse(JSON.stringify(textRegions)) as BoundingBoxTuple[];
    const merged: MergedBox[] = [];
    const used = new Array(boxes.length).fill(false);

    // Iterate and merge (same algorithm as Python)
    for (let i = 0; i < boxes.length; i++) {
        if (used[i]) continue;

        let group: BoundingBoxTuple[] = [boxes[i]];
        used[i] = true;
        let changed = true;

        while (changed) {
            changed = false;
            for (let j = 0; j < boxes.length; j++) {
                if (used[j] || j === i) continue;
                
                const box2 = boxes[j];
                let canMerge = false;

                // Check box2 against every box currently in the group
                for (const gBox of group) {
                    const [x1_1, y1_1, x2_1, y2_1] = gBox;
                    const [x1_2, y1_2, x2_2, y2_2] = box2;

                    const w1 = x2_1 - x1_1;
                    const w2 = x2_2 - x1_2;

                    // Vertical Gap check
                    const v_gap1 = Math.abs(y2_1 - y1_2);
                    const v_gap2 = Math.abs(y2_2 - y1_1);
                    const is_v_adj = (v_gap1 < verticalTolerance || v_gap2 < verticalTolerance);

                    if (!is_v_adj) continue;

                    // Horizontal Alignment check
                    const x_align = Math.abs(x1_1 - x1_2);
                    if (x_align > horizontalTolerance) continue;

                    // Width Ratio check
                    if (w1 > 0 && w2 > 0) {
                        const ratio = Math.min(w1, w2) / Math.max(w1, w2);
                        if (ratio < widthRatioThreshold) continue;
                    }

                    canMerge = true;
                    break;
                }

                if (canMerge) {
                    group.push(box2);
                    used[j] = true;
                    changed = true;
                }
            }
        }

        // Create merged bbox from group
        if (group.length > 0) {
            const min_x = Math.min(...group.map(b => b[0]));
            const min_y = Math.min(...group.map(b => b[1]));
            const max_x = Math.max(...group.map(b => b[2]));
            const max_y = Math.max(...group.map(b => b[3]));
            
            merged.push({
                rect: [min_x, min_y, max_x, max_y],
                count: group.length,
                originalBoxes: group
            });
        }
    }

    return merged;
}

