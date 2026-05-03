// --- ANALYTICS CONFIGURATION ---
const velocityHistory = [];
const ANALYTICS_WINDOW = 30; // Look at the last ~30 frames to calculate smoothness
let currentSmoothnessScore = 100; // Start at 100
let currentVelocity = 0;

export function resetAnalytics() {
    velocityHistory.length = 0;
    currentSmoothnessScore = 100;
    currentVelocity = 0;
}

export function getCurrentVelocity() {
    return currentVelocity;
}

export function getCurrentSmoothnessScore() {
    return currentSmoothnessScore;
}

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

    const maxVelocity = Math.max(...velocityHistory, 1);
    const minVelocity = Math.min(...velocityHistory, 0);
    const range = Math.max(maxVelocity - minVelocity, 1);

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

export function analyzeSmoothness(cogHistory, canvasContext, canvasWidth, canvasHeight) {
    if (cogHistory.length > 1) {
        const currentPos = cogHistory[cogHistory.length - 1];
        const previousPos = cogHistory[cogHistory.length - 2];

        // 1. Calculate distance moved this frame (in pixels)
        const dx = (currentPos.x - previousPos.x) * canvasWidth;
        const dy = (currentPos.y - previousPos.y) * canvasHeight;
        const velocity = Math.sqrt(dx * dx + dy * dy);
        currentVelocity = velocity;

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
            let scoreCalc = 100 - (variance / 10);

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
    canvasContext.fillText('Smoothness Score', 40, 82);
}
