/**
 * alignmentTracker.js
 * Logic for calculating the time-based percentage of COG alignment.
 */

let alignedFrames = 0;
let totalFrames = 0;
const HORIZONTAL_THRESHOLD = 0.05; // 5% of screen width tolerance

/**
 * Updates the frame count and calculates the current alignment percentage.
 * @param {Object} currentCog - {x, y} normalized coordinates of current COG
 * @param {Object} optimalData - {x, y} normalized coordinates of optimal COG
 * @returns {number} The current percentage of time aligned (0-100)
 */
export function calculateAlignmentPercentage(currentCog, optimalData) {
    if (!currentCog || !optimalData) return 0;

    totalFrames++;

    // Compare the horizontal distance between current and optimal COG
    const diff = Math.abs(currentCog.x - optimalData.x);

    if (diff <= HORIZONTAL_THRESHOLD) {
        alignedFrames++;
    }

    return (alignedFrames / totalFrames) * 100;
}

/**
 * Resets the tracking counters for a new video or session.
 */
export function resetAlignmentStats() {
    alignedFrames = 0;
    totalFrames = 0;
}
