// --- ANALYTICS CONFIGURATION ---
const velocityHistory = [];
const ANALYTICS_WINDOW = 30; // Look at the last ~30 frames to calculate smoothness
let currentSmoothnessScore = 100; // Start at 100

export function resetAnalytics() {
    velocityHistory.length = 0;
    currentSmoothnessScore = 100;
}

export function analyzeFlowState(cogHistory, canvasContext, canvasWidth, canvasHeight) {
    if (cogHistory.length > 1) {
        const currentPos = cogHistory[cogHistory.length - 1];
        const previousPos = cogHistory[cogHistory.length - 2];

        // 1. Calculate distance moved this frame (in pixels)
        const dx = (currentPos.x - previousPos.x) * canvasWidth;
        const dy = (currentPos.y - previousPos.y) * canvasHeight;
        const velocity = Math.sqrt(dx * dx + dy * dy);

        velocityHistory.push(velocity);

        // Keep the array limited to our rolling window
        if (velocityHistory.length > ANALYTICS_WINDOW) {
            velocityHistory.shift();
        }

        // 2. Calculate Variance once we have enough data frames
        if (velocityHistory.length === ANALYTICS_WINDOW) {
            // Find the average velocity
            const avgVelocity = velocityHistory.reduce((sum, v) => sum + v, 0) / ANALYTICS_WINDOW;

            // Calculate variance (how much each frame differs from the average)
            let variance = 0;
            for (let i = 0; i < velocityHistory.length; i++) {
                variance += Math.pow(velocityHistory[i] - avgVelocity, 2);
            }
            variance = variance / ANALYTICS_WINDOW;

            // 3. Map Variance to a 0-100 Score
            // A variance of 0 is perfect flow (100). 
            // TWEAK THIS DIVISOR (e.g., 20) during the hackathon based on real test videos!
            let scoreCalc = 100 - (variance / 20);

            // Clamp the score between 0 and 100
            currentSmoothnessScore = Math.max(0, Math.min(100, scoreCalc));
        }
    }

    // --- DRAW THE SCORE ON THE UI ---
    canvasContext.fillStyle = 'rgba(9, 18, 30, 0.7)';
    canvasContext.roundRect(20, 20, 220, 80, 12);
    canvasContext.fill();
    canvasContext.strokeStyle = 'rgba(131, 160, 194, 0.3)';
    canvasContext.lineWidth = 1;
    canvasContext.stroke();

    // Color code: Green if smooth, Orange if jerky
    canvasContext.fillStyle = currentSmoothnessScore > 75 ? '#67f2c4' : '#ffd166';
    canvasContext.font = 'bold 32px Inter, sans-serif';
    canvasContext.fillText(currentSmoothnessScore.toFixed(0), 40, 60);

    canvasContext.fillStyle = '#8fa6c2';
    canvasContext.font = '14px Inter, sans-serif';
    canvasContext.fillText('Flow State Score', 40, 82);
}
