// dataAnalysis.js — Performance Analytics
// 
// Metrics are body-height-normalized so scores (should be) independent of video
// resolution, zoom level, and camera distance.

// --- ANALYTICS CONFIGURATION ---
const ANALYTICS_WINDOW = 30;        // Rolling window for jerk RMS (Root mean square to get mean) (~1s at 30fps)
const JERK_SIGMOID_K = 8.0;         // Sigmoid steepness (better controlling how rapidly the metric changes) — tuned so typical climbing maps to mid-range
const SCORE_EMA_ALPHA = 0.08;       // EMA (exponential moving average) improved smoothing for the displayed smoothness score
const STABILITY_SIGMA = 0.04;       // Gaussian σ (use in normal distribution) for instant stability (~4% body width)
const STABILITY_EMA_ALPHA = 0.05;   // EMA smoothing for session stability

// --- STATE ---
const velocityHistory = [];         // Normalized velocities (body-heights/frame)
const accelHistory = [];            // Normalized accelerations for jerk calculation
let currentSmoothnessScore = 100;   // EMA-smoothed display score
let peakSmoothnessScore = 0;        // Best score seen this session
let minSmoothnessScore = 100;       // Worst score seen this session
let currentVelocity = 0;            // Current frame velocity (normalized)

// Stability state
let sessionStabilityEMA = 0;        // Cumulative EMA of instant accuracy scores
let totalFrames = 0;
let stabilityScoreSum = 0;          // Running sum for true average


//reset
export function resetAnalytics() {
    velocityHistory.length = 0;
    accelHistory.length = 0;
    currentSmoothnessScore = 100;
    peakSmoothnessScore = 0;
    minSmoothnessScore = 100;
    currentVelocity = 0;
    sessionStabilityEMA = 0;
    totalFrames = 0;
    stabilityScoreSum = 0;
}

/**
 * Calculates detailed metrics using a Gaussian decay / normal distribution to improve general accuracy
 * Being 3% off scores ~75, being 6% off scores ~32, etc.
 * 
 * Instant: Gaussian of horizontal gap → 0-100
 * Session: EMA of instant scores (responsive to recent trend, reflects history)
 */
export function calculateDetailedMetrics(currentCog, optimalData) {
    if (!currentCog || !optimalData) {
        return { instant: 0, session: 0 };
    }

    totalFrames++;

    // 1. Horizontal offset between current and optimal COG (normalized 0–1 coords)
    const gap = Math.abs(currentCog.x - optimalData.x);

    // 2. Gaussian decay (basically a normal distribution): exp(-(gap²) / (2σ²)) * 100
    //    σ = STABILITY_SIGMA (0.04), so:
    //      gap=0    → 100.0 (perfect)
    //      gap=0.03 → ~75   (good)
    //      gap=0.04 → ~60   (moderate)
    //      gap=0.06 → ~32   (bad)
    //      gap=0.10 → ~4    (very bad)
    const instantScore = Math.exp(-(gap * gap) / (2 * STABILITY_SIGMA * STABILITY_SIGMA)) * 100;

    // 3. Session stability: EMA (exponential moving average) combine history with recent values
    //    First frame seeds the EMA; subsequent frames blend
    if (totalFrames === 1) {
        sessionStabilityEMA = instantScore;
    } else {
        sessionStabilityEMA = STABILITY_EMA_ALPHA * instantScore + (1 - STABILITY_EMA_ALPHA) * sessionStabilityEMA;
    }

    // Also track a running average for the session summary
    stabilityScoreSum += instantScore;

    return {
        instant: instantScore.toFixed(1),
        session: sessionStabilityEMA.toFixed(1)
    };
}

export function getCurrentVelocity() {
    return currentVelocity;
}

export function getCurrentSmoothnessScore() {
    return currentSmoothnessScore;
}

/**
 * smoothness analysis. 
 *
 * Pipeline:
 *   1. Normalize velocity by body height ()
 *   2. Compute acceleration (Δvelocity)
 *   3. Compute jerk (Δacceleration) — 3rd derivative of position
 *   4. RMS jerk over period (accel history window)
 *   5. Sigmoid mapping (in order to map to bounded range)→ 0–100
 *   6. EMA (exponential moving average) on displayed score for visual stability
 *
 * @param {Array} cogHistory - recent COG positions (normalized 0–1 coords)
 * @param {CanvasRenderingContext2D} canvasContext - for drawing the HUD
 * @param {number} canvasWidth
 * @param {number} canvasHeight
 * @param {number} s - scale factor for drawing (authored for 720p)
 * @param {number} bodyHeight - distance from head to ankle midpoint (normalized coords, 0–1)
 */
export function analyzeSmoothness(cogHistory, canvasContext, canvasWidth, canvasHeight, s = 1, bodyHeight = 0.5) {
    // Guard: need at least 2 points
    if (cogHistory.length > 1) {
        const currentPos = cogHistory[cogHistory.length - 1];
        const previousPos = cogHistory[cogHistory.length - 2];

        // 1. Calculate displacement in normalized coordinates
        const dx = currentPos.x - previousPos.x;
        const dy = currentPos.y - previousPos.y;
        const rawVelocity = Math.sqrt(dx * dx + dy * dy);

        // 2. Normalize by body height
        const safeBodyHeight = Math.max(bodyHeight, 0.05); // prevent division by near-zero
        const normalizedVelocity = rawVelocity / safeBodyHeight;
        currentVelocity = normalizedVelocity;

        velocityHistory.push(normalizedVelocity);
        if (velocityHistory.length > ANALYTICS_WINDOW) velocityHistory.shift();

        // 3. Compute acceleration (Δvelocity) — needs ≥2 velocity samples
        if (velocityHistory.length >= 2) {
            const accel = Math.abs(
                velocityHistory[velocityHistory.length - 1] - velocityHistory[velocityHistory.length - 2]
            );
            accelHistory.push(accel);
            if (accelHistory.length > ANALYTICS_WINDOW) accelHistory.shift();
        }

        // 4. Compute jerk (derivative of accel) RMS over a period — needs ≥2 acceleration samples
        if (accelHistory.length >= 2) {
            let jerkSumSq = 0;
            let jerkCount = 0;
            for (let i = 1; i < accelHistory.length; i++) {
                const jerk = Math.abs(accelHistory[i] - accelHistory[i - 1]);
                jerkSumSq += jerk * jerk;
                jerkCount++;
            }
            const rmsJerk = Math.sqrt(jerkSumSq / jerkCount);

            // 5. Sigmoid mapping (converting room meansquare jerk into a normalize 0-100 score): score = 100 / (1 + k * rmsJerk)
            //    rmsJerk=0    → 100  (perfectly smooth)
            //    rmsJerk=0.01 → ~93  (very smooth)
            //    rmsJerk=0.05 → ~71  (moderate)
            //    rmsJerk=0.15 → ~45  (jerky)
            //    rmsJerk=0.50 → ~20  (very jerky)
            const rawScore = 100 / (1 + JERK_SIGMOID_K * rmsJerk);

            // 6. EMA smoothing on the displayed score
            currentSmoothnessScore = SCORE_EMA_ALPHA * rawScore + (1 - SCORE_EMA_ALPHA) * currentSmoothnessScore;

            // Track session extremes
            peakSmoothnessScore = Math.max(peakSmoothnessScore, currentSmoothnessScore);
            minSmoothnessScore = Math.min(minSmoothnessScore, currentSmoothnessScore);
        }
    }

    // --- DRAW THE SCORE ON THE OVERLAY HUD ---
    const boxX = 20 * s;
    const boxY = 20 * s;
    const boxW = 220 * s;
    const boxH = 80 * s;

    canvasContext.fillStyle = 'rgba(9, 18, 30, 0.7)';
    canvasContext.roundRect(boxX, boxY, boxW, boxH, 12 * s);
    canvasContext.fill();
    canvasContext.strokeStyle = 'rgba(131, 160, 194, 0.3)';
    canvasContext.lineWidth = 1 * s;
    canvasContext.stroke();

    // Color code: Green if smooth, Orange if jerky
    canvasContext.fillStyle = currentSmoothnessScore > 75 ? '#67f2c4' : '#ffd166';
    canvasContext.font = `bold ${Math.round(32 * s)}px Inter, sans-serif`;
    canvasContext.fillText(currentSmoothnessScore.toFixed(0), boxX + 20 * s, boxY + 40 * s);

    canvasContext.fillStyle = '#8fa6c2';
    canvasContext.font = `${Math.round(14 * s)}px Inter, sans-serif`;
    canvasContext.fillText('Smoothness Score', boxX + 20 * s, boxY + 62 * s);
}


//Velocity stuff

//Draws the velocity graph
export function drawVelocityChart(chartContext, chartWidth, chartHeight) {
    chartContext.clearRect(0, 0, chartWidth, chartHeight);

    // Panel background
    chartContext.fillStyle = '#081018';
    chartContext.fillRect(0, 0, chartWidth, chartHeight);

    // Grid
    chartContext.strokeStyle = 'rgba(131, 160, 194, 0.25)';
    chartContext.lineWidth = 1;
    const gridLines = 4;
    for (let i = 1; i <= gridLines; i++) {
        const y = (chartHeight / (gridLines + 1)) * i;
        chartContext.beginPath();
        chartContext.moveTo(0, y);
        chartContext.lineTo(chartWidth, y);
        chartContext.stroke();
    }

    if (velocityHistory.length < 2) {
        return;
    }

    const maxVelocity = Math.max(...velocityHistory, 0.001);
    const minVelocity = Math.min(...velocityHistory, 0);
    const range = Math.max(maxVelocity - minVelocity, 0.001);

    chartContext.beginPath();
    chartContext.strokeStyle = '#7ec8ff';
    chartContext.lineWidth = 2;

    for (let i = 0; i < velocityHistory.length; i++) {
        const x = (i / (velocityHistory.length - 1)) * chartWidth;
        const normalized = (velocityHistory[i] - minVelocity) / range;
        const y = chartHeight - normalized * chartHeight;

        if (i === 0) {
            chartContext.moveTo(x, y);
        } else {
            chartContext.lineTo(x, y);
        }
    }

    chartContext.stroke();
}

// ============================================================================
// SESSION SUMMARY
// ============================================================================

/**
 * Returns a snapshot of all session metrics for the post-video summary.
 */
export function getSessionSummary() {
    const avgStability = totalFrames > 0
        ? (stabilityScoreSum / totalFrames).toFixed(1)
        : '0.0';

    return {
        avgStability,
        smoothnessScore: currentSmoothnessScore.toFixed(0),
        peakSmoothnessScore: peakSmoothnessScore.toFixed(0),
        minSmoothnessScore: minSmoothnessScore.toFixed(0),
        avgVelocity: velocityHistory.length > 0
            ? (velocityHistory.reduce((s, v) => s + v, 0) / velocityHistory.length).toFixed(3)
            : '0.000',
        peakVelocity: velocityHistory.length > 0
            ? Math.max(...velocityHistory).toFixed(3)
            : '0.000',
        totalFramesAnalyzed: totalFrames,
    };
}
