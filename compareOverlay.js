import { DrawingUtils, FilesetResolver, PoseLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs';
import { calculateDetailedMetrics } from './dataAnalysis.js';

// ─── CONSTANTS ────────────────────────────────────────────────────────────────
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task';
const MAX_DETECTION_WIDTH = 640;
const MAX_PATH_POINTS = 300;

// EMA / analytics constants (matching main page)
const COG_EMA_ALPHA = 0.35;
const OPTIMAL_EMA_ALPHA = 0.35;
const COG_HISTORY_LEN = 8;
const ANALYTICS_WINDOW = 30;
const JERK_SIGMOID_K = 8.0;
const SCORE_EMA_ALPHA = 0.08;
const STABILITY_SIGMA = 0.04;
const STABILITY_EMA_ALPHA = 0.05;

// ─── SHARED MEDIAPIPE INSTANCE ────────────────────────────────────────────────
let poseLandmarker;
let animationFrameId = null;
let frameCount = 0;

// ─── SEPARATE OFFSCREEN CANVASES (one per video for proper tracking) ──────────
const inputCanvasA = document.createElement('canvas');
const inputCtxA = inputCanvasA.getContext('2d', { willReadFrequently: true });
const inputCanvasB = document.createElement('canvas');
const inputCtxB = inputCanvasB.getContext('2d', { willReadFrequently: true });

// ─── DOM ──────────────────────────────────────────────────────────────────────
const statusText = document.getElementById('status-text');

// ─── PER-PANEL ANALYTICS STATE ────────────────────────────────────────────────
function createAnalyticsState() {
    return {
        velocityHistory: [],
        accelHistory: [],
        smoothnessScore: 100,
        peakSmoothness: 0,
        minSmoothness: 100,
        currentVelocity: 0,
        sessionStabilityEMA: 0,
        totalFrames: 0,
        stabilityScoreSum: 0,
    };
}

function resetAnalyticsState(a) {
    a.velocityHistory.length = 0;
    a.accelHistory.length = 0;
    a.smoothnessScore = 100;
    a.peakSmoothness = 0;
    a.minSmoothness = 100;
    a.currentVelocity = 0;
    a.sessionStabilityEMA = 0;
    a.totalFrames = 0;
    a.stabilityScoreSum = 0;
}

// ─── PER-VIDEO STATE FACTORY ──────────────────────────────────────────────────
function createPanel(id) {
    const video = document.getElementById(`video-${id}`);
    const canvas = document.getElementById(`overlay-${id}`);
    const ctx = canvas.getContext('2d');
    const drawUtils = new DrawingUtils(ctx);
    const fileInput = document.getElementById(`file-${id}`);
    const labelEl = document.getElementById(`label-${id}`);
    const stabilityEl = document.getElementById(`stability-${id}`);
    const accuracyEl = document.getElementById(`accuracy-${id}`);
    const velocityEl = document.getElementById(`velocity-${id}`);

    // Assign dedicated input canvas and context to each panel
    const inputCanvas = id === 'a' ? inputCanvasA : inputCanvasB;
    const inputCtx = id === 'a' ? inputCtxA : inputCtxB;
    
    return {
        id, video, canvas, ctx, drawUtils, fileInput, labelEl,
        stabilityEl, accuracyEl, velocityEl,
        cogHistory: [],
        cogPath: [],
        prevSmoothedCOG: null,
        prevSmoothedOptimal: null,
        lastVideoTime: -1,
        lastResult: null,
        objectUrl: null,
        ended: false,
        analytics: createAnalyticsState(),
        inputCanvas,
        inputCtx,
    };
}

const panels = [createPanel('a'), createPanel('b')];

// ─── MATH HELPERS ─────────────────────────────────────────────────────────────
function midpoint(p1, p2) {
    return { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2, z: (p1.z + p2.z) / 2 };
}

// COG with de Leva segments + EMA smoothing
function calculateCOG(landmarks, panel) {
    const head = landmarks[0];
    const trunk = midpoint(
        midpoint(landmarks[11], landmarks[12]),
        midpoint(landmarks[23], landmarks[24])
    );
    const segments = [
        { pos: head, weight: 0.0700 },
        { pos: trunk, weight: 0.4556 },
        { pos: midpoint(landmarks[23], landmarks[25]), weight: 0.10 },
        { pos: midpoint(landmarks[24], landmarks[26]), weight: 0.10 },
        { pos: midpoint(landmarks[25], landmarks[27]), weight: 0.05 },
        { pos: midpoint(landmarks[26], landmarks[28]), weight: 0.05 },
        { pos: midpoint(landmarks[11], landmarks[13]), weight: 0.03 },
        { pos: midpoint(landmarks[12], landmarks[14]), weight: 0.03 },
        { pos: midpoint(landmarks[13], landmarks[15]), weight: 0.02 },
        { pos: midpoint(landmarks[14], landmarks[16]), weight: 0.02 },
        { pos: midpoint(landmarks[27], landmarks[31] || landmarks[27]), weight: 0.015 },
        { pos: midpoint(landmarks[28], landmarks[32] || landmarks[28]), weight: 0.015 },
    ];
    let raw = { x: 0, y: 0 };
    segments.forEach(s => { raw.x += s.pos.x * s.weight; raw.y += s.pos.y * s.weight; });

    // EMA smoothing
    if (!panel.prevSmoothedCOG) {
        panel.prevSmoothedCOG = { x: raw.x, y: raw.y };
    } else {
        panel.prevSmoothedCOG = {
            x: COG_EMA_ALPHA * raw.x + (1 - COG_EMA_ALPHA) * panel.prevSmoothedCOG.x,
            y: COG_EMA_ALPHA * raw.y + (1 - COG_EMA_ALPHA) * panel.prevSmoothedCOG.y,
        };
    }

    panel.cogHistory.push({ x: panel.prevSmoothedCOG.x, y: panel.prevSmoothedCOG.y });
    if (panel.cogHistory.length > COG_HISTORY_LEN) panel.cogHistory.shift();

    return { x: panel.prevSmoothedCOG.x, y: panel.prevSmoothedCOG.y };
}

// Optimal COG with clamped t + EMA
function calculateOptimalCOG(landmarks, cog, panel) {
    const la = landmarks[27], ra = landmarks[28];
    const lw = landmarks[15], rw = landmarks[16];
    if (!la || !ra || !lw || !rw) return null;

    const baseX = (la.x + ra.x) / 2, baseY = (la.y + ra.y) / 2;
    const pullX = (lw.x + rw.x) / 2, pullY = (lw.y + rw.y) / 2;

    let optX;
    if (Math.abs(pullY - baseY) < 0.001) {
        optX = baseX;
    } else {
        let t = (cog.y - baseY) / (pullY - baseY);
        t = Math.max(0, Math.min(1, t));
        optX = baseX + t * (pullX - baseX);
    }

    const raw = { x: optX, y: cog.y, anchors: { baseX, baseY, pullX, pullY } };

    if (!panel.prevSmoothedOptimal) {
        panel.prevSmoothedOptimal = { x: raw.x, y: raw.y };
    } else {
        panel.prevSmoothedOptimal = {
            x: OPTIMAL_EMA_ALPHA * raw.x + (1 - OPTIMAL_EMA_ALPHA) * panel.prevSmoothedOptimal.x,
            y: OPTIMAL_EMA_ALPHA * raw.y + (1 - OPTIMAL_EMA_ALPHA) * panel.prevSmoothedOptimal.y,
        };
    }

    return { x: panel.prevSmoothedOptimal.x, y: panel.prevSmoothedOptimal.y, anchors: raw.anchors };
}

// ─── PER-PANEL ANALYTICS ──────────────────────────────────────────────────────

// Gaussian stability (matching dataAnalysis.js)
function calculatePanelMetrics(cog, optimal, a) {
    if (!cog || !optimal) return { instant: 0, session: 0 };
    a.totalFrames++;
    const gap = Math.abs(cog.x - optimal.x);
    const instantScore = Math.exp(-(gap * gap) / (2 * STABILITY_SIGMA * STABILITY_SIGMA)) * 100;

    if (a.totalFrames === 1) {
        a.sessionStabilityEMA = instantScore;
    } else {
        a.sessionStabilityEMA = STABILITY_EMA_ALPHA * instantScore + (1 - STABILITY_EMA_ALPHA) * a.sessionStabilityEMA;
    }
    a.stabilityScoreSum += instantScore;

    return { instant: instantScore.toFixed(1), session: a.sessionStabilityEMA.toFixed(1) };
}

// Normalized jerk smoothness (matching dataAnalysis.js)
function analyzePanelSmoothness(cogHistory, a, bodyHeight) {
    if (cogHistory.length < 2) return;
    const curr = cogHistory[cogHistory.length - 1];
    const prev = cogHistory[cogHistory.length - 2];
    const dx = curr.x - prev.x;
    const dy = curr.y - prev.y;
    const rawVel = Math.sqrt(dx * dx + dy * dy);
    const safeH = Math.max(bodyHeight, 0.05);
    const normVel = rawVel / safeH;
    a.currentVelocity = normVel;

    a.velocityHistory.push(normVel);
    if (a.velocityHistory.length > ANALYTICS_WINDOW) a.velocityHistory.shift();

    if (a.velocityHistory.length >= 2) {
        const accel = Math.abs(a.velocityHistory[a.velocityHistory.length - 1] - a.velocityHistory[a.velocityHistory.length - 2]);
        a.accelHistory.push(accel);
        if (a.accelHistory.length > ANALYTICS_WINDOW) a.accelHistory.shift();
    }

    if (a.accelHistory.length >= 2) {
        let jerkSumSq = 0, jerkCount = 0;
        for (let i = 1; i < a.accelHistory.length; i++) {
            const jerk = Math.abs(a.accelHistory[i] - a.accelHistory[i - 1]);
            jerkSumSq += jerk * jerk;
            jerkCount++;
        }
        const rmsJerk = Math.sqrt(jerkSumSq / jerkCount);
        const rawScore = 100 / (1 + JERK_SIGMOID_K * rmsJerk);
        a.smoothnessScore = SCORE_EMA_ALPHA * rawScore + (1 - SCORE_EMA_ALPHA) * a.smoothnessScore;
        a.peakSmoothness = Math.max(a.peakSmoothness, a.smoothnessScore);
        a.minSmoothness = Math.min(a.minSmoothness, a.smoothnessScore);
    }
}

function getPanelSummary(panel) {
    const a = panel.analytics;
    return {
        avgStability: a.totalFrames > 0 ? (a.stabilityScoreSum / a.totalFrames).toFixed(1) : '0.0',
        smoothnessScore: a.smoothnessScore.toFixed(0),
        avgVelocity: a.velocityHistory.length > 0
            ? (a.velocityHistory.reduce((s, v) => s + v, 0) / a.velocityHistory.length).toFixed(3) : '0.000',
        peakVelocity: a.velocityHistory.length > 0
            ? Math.max(...a.velocityHistory).toFixed(3) : '0.000',
        totalFramesAnalyzed: a.totalFrames,
    };
}

// ─── DRAWING ──────────────────────────────────────────────────────────────────
function drawPanel(panel) {
    const { ctx, canvas, video, drawUtils, cogHistory, cogPath,
        stabilityEl, accuracyEl, velocityEl, lastResult, analytics } = panel;
    if (!lastResult) return;


    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Only draw skeleton if we have a valid result with landmarks
    if (!lastResult || !lastResult.landmarks?.[0]?.length) {
        ctx.restore();
        return;
    }

    const landmarks = lastResult.landmarks[0];

    const s = Math.max(canvas.width, canvas.height) / 720;

    // Skeleton
    drawUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS,
        { color: 'rgba(103, 242, 196, 0.6)', lineWidth: 4 * s });
    drawUtils.drawLandmarks(landmarks, { color: '#f7fbff', radius: 2 * s });

    const cog = calculateCOG(landmarks, panel);
    const optimal = calculateOptimalCOG(landmarks, cog, panel);

    // COG path
    const cpx = cog.x * canvas.width, cpy = cog.y * canvas.height;
    cogPath.push({ x: cpx, y: cpy });
    if (cogPath.length > MAX_PATH_POINTS) cogPath.shift();

    // Effort gap
    if (optimal) {
        ctx.beginPath();
        ctx.moveTo(cpx, cpy);
        ctx.lineTo(optimal.x * canvas.width, optimal.y * canvas.height);
        ctx.strokeStyle = '#ff4d4d';
        ctx.lineWidth = 3 * s;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(optimal.x * canvas.width, optimal.y * canvas.height, 6 * s, 0, Math.PI * 2);
        ctx.fillStyle = '#00f2ff';
        ctx.fill();
    }

    // Current COG ring
    ctx.beginPath();
    ctx.arc(cpx, cpy, 12 * s, 0, Math.PI * 2);
    ctx.strokeStyle = '#ffd166';
    ctx.lineWidth = 3 * s;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cpx, cpy, 4 * s, 0, Math.PI * 2);
    ctx.fillStyle = '#ffd166';
    ctx.fill();

    // Labels
    ctx.fillStyle = '#ffd166';
    ctx.font = `bold ${Math.round(13 * s)}px Inter, sans-serif`;
    ctx.fillText('COG', cpx + 14 * s, cpy - 5 * s);
    if (optimal) {
        ctx.fillStyle = '#00f2ff';
        ctx.fillText('Optimal', optimal.x * canvas.width + 14 * s, optimal.y * canvas.height + 14 * s);
    }

    // Body height for normalization
    const ankleMid = midpoint(landmarks[27], landmarks[28]);
    const bodyHeight = Math.sqrt(
        Math.pow(landmarks[0].x - ankleMid.x, 2) + Math.pow(landmarks[0].y - ankleMid.y, 2)
    );

    // Run per-panel analytics
    analyzePanelSmoothness(cogHistory, analytics, bodyHeight);

    if (optimal) {
        const metrics = calculatePanelMetrics(cog, optimal, analytics);
        if (frameCount % 2 === 0) {
            if (stabilityEl) {
                stabilityEl.textContent = `${metrics.session}%`;
                stabilityEl.style.color = parseFloat(metrics.session) > 70 ? '#67f2c4' : '#ffd166';
                stabilityEl.classList.remove('dim');
            }
            if (accuracyEl) {
                accuracyEl.textContent = `${metrics.instant}%`;
                accuracyEl.style.color = parseFloat(metrics.instant) > 80 ? '#4ade80' : '#facc15';
                accuracyEl.classList.remove('dim');
            }
        }
    }

    if (frameCount % 2 === 0 && velocityEl) {
        velocityEl.textContent = `${(analytics.currentVelocity * 100).toFixed(1)}% bh/f`;
        velocityEl.classList.remove('dim');
    }

    // Smoothness overlay box
    const bx = 16 * s, by = 16 * s, bw = 180 * s, bh = 70 * s;
    ctx.fillStyle = 'rgba(9,18,30,0.72)';
    ctx.roundRect(bx, by, bw, bh, 10 * s);
    ctx.fill();
    ctx.strokeStyle = 'rgba(131,160,194,0.3)';
    ctx.lineWidth = s;
    ctx.stroke();

    ctx.fillStyle = analytics.smoothnessScore > 75 ? '#67f2c4' : '#ffd166';
    ctx.font = `bold ${Math.round(24 * s)}px Inter, sans-serif`;
    ctx.fillText(analytics.smoothnessScore.toFixed(0), bx + 14 * s, bx + 30 * s);

    ctx.fillStyle = '#8fa6c2';
    ctx.font = `${Math.round(11 * s)}px Inter, sans-serif`;
    ctx.fillText('Smoothness', bx + 14 * s, by + 50 * s);

    ctx.restore();
}

// ─── CANVAS RESIZE ────────────────────────────────────────────────────────────
function resizePanel(panel) {
    const vw = panel.video.videoWidth || 1280;
    const vh = panel.video.videoHeight || 720;
    panel.canvas.width = vw;
    panel.canvas.height = vh;
    // Keep this panel's input canvas sized appropriately
    const scale = Math.min(1, MAX_DETECTION_WIDTH / Math.max(vw, 1));
    panel.inputCanvas.width = Math.round(vw * scale);
    panel.inputCanvas.height = Math.round(vh * scale);
}

// ─── CORE LOOP ────────────────────────────────────────────────────────────────
function trackFrame() {
    if (!poseLandmarker) {
        animationFrameId = requestAnimationFrame(trackFrame);
        return;
    }

    frameCount++;

    // Stagger detection: only detect ONE panel per frame for better performance
    // Panel A on even frames, Panel B on odd frames
    const detectPanelIndex = frameCount % 2;

    panels.forEach((panel, index) => {
        if (panel.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
            panel.video.currentTime !== panel.lastVideoTime) {

            panel.lastVideoTime = panel.video.currentTime;
            
            // Only run detection for the designated panel this frame
            if (index === detectPanelIndex) {
                resizePanel(panel);
                
                // Use this panel's dedicated input canvas
                panel.inputCtx.clearRect(0, 0, panel.inputCanvas.width, panel.inputCanvas.height);
                panel.inputCtx.drawImage(panel.video, 0, 0, panel.inputCanvas.width, panel.inputCanvas.height);
                
                // Run pose detection
                panel.lastResult = poseLandmarker.detectForVideo(panel.inputCanvas, performance.now());
            }
        }
    });

    // Draw both panels every frame (each uses its most recent detection)
    panels.forEach(drawPanel);
    animationFrameId = requestAnimationFrame(trackFrame);
}

// ─── MODEL LOADING ────────────────────────────────────────────────────────────
async function loadLandmarker() {
    setStatus('Loading MediaPipe model...');
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numPoses: 1,
        minPoseDetectionConfidence: 0.3,  // Lower threshold for better detection
        minPosePresenceConfidence: 0.3,   // Lower threshold for better detection
        minTrackingConfidence: 0.3,       // Lower threshold for better tracking
    });

    try {
        const w = document.createElement('canvas');
        w.width = MAX_DETECTION_WIDTH;
        w.height = Math.round(MAX_DETECTION_WIDTH * 9 / 16);
        poseLandmarker.detectForVideo(w, performance.now());
    } catch (_) { }

    setStatus('Ready — upload two videos to compare');
}

function setStatus(msg) {
    if (statusText) statusText.textContent = msg;
}

// ─── COMPARISON SUMMARY ───────────────────────────────────────────────────────

function showComparisonSummary() {
    const section = document.getElementById('compare-summary-section');
    if (!section) return;

    const summA = getPanelSummary(panels[0]);
    const summB = getPanelSummary(panels[1]);

    const set = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    };

    // Populate Run A
    set('cmp-stability-a', `${summA.avgStability}%`);
    set('cmp-smoothness-a', summA.smoothnessScore);
    set('cmp-velocity-a', `${(parseFloat(summA.avgVelocity) * 100).toFixed(1)}%`);
    set('cmp-frames-a', summA.totalFramesAnalyzed.toLocaleString());

    // Populate Run B
    set('cmp-stability-b', `${summB.avgStability}%`);
    set('cmp-smoothness-b', summB.smoothnessScore);
    set('cmp-velocity-b', `${(parseFloat(summB.avgVelocity) * 100).toFixed(1)}%`);
    set('cmp-frames-b', summB.totalFramesAnalyzed.toLocaleString());

    // Color-code winners
    const colorWinner = (idA, idB, valA, valB, higherIsBetter = true) => {
        const elA = document.getElementById(idA);
        const elB = document.getElementById(idB);
        if (!elA || !elB) return;
        const a = parseFloat(valA), b = parseFloat(valB);
        const aWins = higherIsBetter ? a >= b : a <= b;
        elA.style.color = aWins ? '#67f2c4' : '#ffd166';
        elB.style.color = !aWins ? '#67f2c4' : '#ffd166';
    };

    colorWinner('cmp-stability-a', 'cmp-stability-b', summA.avgStability, summB.avgStability);
    colorWinner('cmp-smoothness-a', 'cmp-smoothness-b', summA.smoothnessScore, summB.smoothnessScore);
    colorWinner('cmp-velocity-a', 'cmp-velocity-b', summA.avgVelocity, summB.avgVelocity, false);

    // Verdict
    let aScore = 0, bScore = 0;
    if (parseFloat(summA.avgStability) > parseFloat(summB.avgStability)) aScore++; else bScore++;
    if (parseFloat(summA.smoothnessScore) > parseFloat(summB.smoothnessScore)) aScore++; else bScore++;
    if (parseFloat(summA.avgVelocity) < parseFloat(summB.avgVelocity)) aScore++; else bScore++;

    const verdictEl = document.getElementById('cmp-verdict');
    if (verdictEl) {
        if (aScore > bScore) {
            verdictEl.textContent = 'Run A had better overall metrics';
            verdictEl.style.color = '#67f2c4';
        } else if (bScore > aScore) {
            verdictEl.textContent = 'Run B had better overall metrics';
            verdictEl.style.color = 'var(--accent)';
        } else {
            verdictEl.textContent = 'Both runs performed similarly';
            verdictEl.style.color = 'var(--muted)';
        }
    }

    // Show the section and scroll it into view
    section.classList.add('visible');
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function hideComparisonSummary() {
    const section = document.getElementById('compare-summary-section');
    if (section) section.classList.remove('visible');
}

// ─── FILE INPUT HANDLERS ─────────────────────────────────────────────────────
panels.forEach(panel => {
    panel.fileInput.addEventListener('change', () => {
        const file = panel.fileInput.files?.[0];
        if (!file) return;
        if (panel.objectUrl) URL.revokeObjectURL(panel.objectUrl);
        panel.objectUrl = URL.createObjectURL(file);
        panel.video.src = panel.objectUrl;
        panel.cogHistory.length = 0;
        panel.cogPath.length = 0;
        panel.prevSmoothedCOG = null;
        panel.prevSmoothedOptimal = null;
        panel.lastVideoTime = -1;
        panel.lastResult = null;
        panel.ended = false;
        resetAnalyticsState(panel.analytics);

        // Hide summary when loading new videos
        hideComparisonSummary();

        panel.video.onloadedmetadata = () => {
            resizePanel(panel);
            panel.video.play();
            if (panel.labelEl) panel.labelEl.textContent = file.name;
            setStatus(`${panel.id.toUpperCase()}: ${file.name}`);
        };

        if (animationFrameId === null) trackFrame();
    });

    // Show summary when a video ends
    panel.video.addEventListener('ended', () => {
        panel.ended = true;
        showComparisonSummary();
    });
});

// Expose both show and hide for the inline script (module might load async)
window._hideComparisonSummary = hideComparisonSummary;
window._showComparisonSummary = showComparisonSummary;
window._comparePanels = panels;

// ─── BOOT ─────────────────────────────────────────────────────────────────────
await loadLandmarker();
trackFrame();
