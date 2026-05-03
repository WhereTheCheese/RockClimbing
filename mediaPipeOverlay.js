import { DrawingUtils, FilesetResolver, PoseLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs';
import {
    resetAnalytics,
    analyzeSmoothness,
    drawVelocityChart,
    getCurrentVelocity,
    getCurrentSmoothnessScore,
    calculateDetailedMetrics,
    getSessionSummary
} from './dataAnalysis.js';
import { showSessionSummary } from './sessionSummary.js';

const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const canvasContext = canvas.getContext('2d');

const inputCanvas = document.createElement('canvas');
const inputCtx = inputCanvas.getContext('2d', { willReadFrequently: true });

const drawingUtils = new DrawingUtils(canvasContext);
const webcamButton = document.getElementById('webcam-button');
const videoFileInput = document.getElementById('video-file');
const statusText = document.getElementById('status-text');

// Velocity graph elements
const velocityChart = document.getElementById('velocity-chart');
const velocityCurrent = document.getElementById('data-velocity');
const velocityChartCtx = velocityChart ? velocityChart.getContext('2d') : null;

// Video controls
const playPauseBtn = document.getElementById('play-pause-btn');
const seekBar = document.getElementById('seek-bar');
const timeDisplay = document.getElementById('time-display');
const speedControl = document.getElementById('speed-control');
const prevFrameBtn = document.getElementById('prev-frame-btn');
const nextFrameBtn = document.getElementById('next-frame-btn');

// Analytics UI elements (cached so we don't query the DOM every frame)
const stabilityCurrent = document.getElementById('data-stability');
const accuracyElement = document.getElementById('data-cog-accuracy');

// Model settings controls
const modelSelect = document.getElementById('model-select');
const delegateSelect = document.getElementById('delegate-select');

let poseLandmarker;
let animationFrameId = null;
let lastVideoTime = -1;
let currentObjectUrl = null;

// --- PERFORMANCE TUNING ---
let frameCount = 0;
let lastDetectionResult = null;          // Cache last result so we can draw on skipped frames
const DETECT_EVERY_N_FRAMES = 2;         // Run MediaPipe every 2nd frame (~30 detections/sec at 60fps)
const UI_UPDATE_EVERY_N_FRAMES = 2;      // Throttle DOM text writes to ~30Hz (chart still draws every frame)

// --- CONFIGURATION & TRACKING HISTORY ---
const cogPath = [];
const COG_EMA_ALPHA = 0.35;           // EMA smoothing for COG (0=max smooth, 1=no smooth)
const OPTIMAL_EMA_ALPHA = 0.35;       // EMA smoothing for optimal COG
const MAX_PATH_POINTS = 300;
const MAX_DETECTION_WIDTH = 640;       // Scale down for MediaPipe — it doesn't need full-res frames

// EMA state (persists across frames, reset on new video/seek)
let prevSmoothedCOG = null;
let prevSmoothedOptimal = null;

// Keep a short history of raw COG for the analytics module (smoothness needs recent positions)
const cogHistory = [];
const COG_HISTORY_LEN = 8; // enough for jerk calculation

function getMidpoint(p1, p2) {
    return {
        x: (p1.x + p2.x) / 2,
        y: (p1.y + p2.y) / 2,
        z: (p1.z + p2.z) / 2
    };
}

/**
 * Calculates the anthropometric COG based on de Leva (1996) segment weights.
 * Smoothed via Exponential Moving Average for minimal phase lag.
 *
 * Segment weights (adjusted from de Leva male averages to fit MediaPipe landmarks):
 *   Head 0.07, Trunk 0.4556, Thighs 0.10×2, Shanks 0.05×2, 
 *   Upper arms 0.03×2, Forearms 0.02×2, Feet 0.015×2
 *   Total = 1.0
 */
function calculateCOG(landmarks) {
    const head = landmarks[0];
    const shoulderMid = getMidpoint(landmarks[11], landmarks[12]);
    const hipMid = getMidpoint(landmarks[23], landmarks[24]);
    const trunk = getMidpoint(shoulderMid, hipMid);

    const segments = [
        { pos: head, weight: 0.0700 },                                   // Head
        { pos: trunk, weight: 0.4556 },                                  // Trunk (torso)
        { pos: getMidpoint(landmarks[23], landmarks[25]), weight: 0.10 }, // R-Thigh
        { pos: getMidpoint(landmarks[24], landmarks[26]), weight: 0.10 }, // L-Thigh
        { pos: getMidpoint(landmarks[25], landmarks[27]), weight: 0.05 }, // R-Shank
        { pos: getMidpoint(landmarks[26], landmarks[28]), weight: 0.05 }, // L-Shank
        { pos: getMidpoint(landmarks[11], landmarks[13]), weight: 0.03 }, // R-UpperArm
        { pos: getMidpoint(landmarks[12], landmarks[14]), weight: 0.03 }, // L-UpperArm
        { pos: getMidpoint(landmarks[13], landmarks[15]), weight: 0.02 }, // R-Forearm
        { pos: getMidpoint(landmarks[14], landmarks[16]), weight: 0.02 }, // L-Forearm
        { pos: getMidpoint(landmarks[27], landmarks[31]), weight: 0.015 }, // R-Foot (ankle↔toe)
        { pos: getMidpoint(landmarks[28], landmarks[32]), weight: 0.015 }, // L-Foot (ankle↔toe)
    ];

    let rawCOG = { x: 0, y: 0 };
    segments.forEach(s => {
        rawCOG.x += s.pos.x * s.weight;
        rawCOG.y += s.pos.y * s.weight;
    });

    // EMA smoothing: eliminates the ~2-3 frame lag of SMA
    if (!prevSmoothedCOG) {
        prevSmoothedCOG = { x: rawCOG.x, y: rawCOG.y };
    } else {
        prevSmoothedCOG = {
            x: COG_EMA_ALPHA * rawCOG.x + (1 - COG_EMA_ALPHA) * prevSmoothedCOG.x,
            y: COG_EMA_ALPHA * rawCOG.y + (1 - COG_EMA_ALPHA) * prevSmoothedCOG.y
        };
    }

    // Keep a short raw history for the analytics module (smoothness jerk calc)
    cogHistory.push({ x: prevSmoothedCOG.x, y: prevSmoothedCOG.y });
    if (cogHistory.length > COG_HISTORY_LEN) cogHistory.shift();

    return { x: prevSmoothedCOG.x, y: prevSmoothedCOG.y };
}

/**
 * Calculates the "Achievable Optimal X" by interpolating between the 
 * base of support (feet) and the upper anchor (hands) at the current COG height.
 * 
 * Now clamps the interpolation parameter t to [0, 1] so overhangs and
 * inverted positions don't produce nonsensical optimal points.
 * Smoothed via EMA instead of SMA for less phase lag.
 */
function calculateOptimalCOG(landmarks, currentCog) {
    const leftAnkle = landmarks[27];
    const rightAnkle = landmarks[28];
    const leftWrist = landmarks[15];
    const rightWrist = landmarks[16];

    if (!leftAnkle || !rightAnkle || !leftWrist || !rightWrist) return null;

    // 1. Define Lower Anchor (Base of Support)
    const baseX = (leftAnkle.x + rightAnkle.x) / 2;
    const baseY = (leftAnkle.y + rightAnkle.y) / 2;

    // 2. Define Upper Anchor (Pull Center)
    const pullX = (leftWrist.x + rightWrist.x) / 2;
    const pullY = (leftWrist.y + rightWrist.y) / 2;

    let optimalX;

    // 3. Find where the current Y intersects the Tension Line
    // Prevent divide by zero if hands and feet are exactly horizontal (ex: heel hook)
    if (Math.abs(pullY - baseY) < 0.001) {
        optimalX = baseX;
    } else {
        // Calculate the percentage of height (t) the COG is at between feet and hands
        let t = (currentCog.y - baseY) / (pullY - baseY);
        // Clamp t to [0, 1] — prevents nonsensical extrapolation on overhangs
        t = Math.max(0, Math.min(1, t));
        // Map that percentage to the X axis
        optimalX = baseX + t * (pullX - baseX);
    }

    const rawOptimal = {
        x: optimalX,
        y: currentCog.y,
        // We return the anchor points to draw the Tension Line in drawResults
        anchors: { baseX, baseY, pullX, pullY }
    };

    // EMA smoothing
    if (!prevSmoothedOptimal) {
        prevSmoothedOptimal = { x: rawOptimal.x, y: rawOptimal.y };
    } else {
        prevSmoothedOptimal = {
            x: OPTIMAL_EMA_ALPHA * rawOptimal.x + (1 - OPTIMAL_EMA_ALPHA) * prevSmoothedOptimal.x,
            y: OPTIMAL_EMA_ALPHA * rawOptimal.y + (1 - OPTIMAL_EMA_ALPHA) * prevSmoothedOptimal.y
        };
    }

    return {
        x: prevSmoothedOptimal.x,
        y: prevSmoothedOptimal.y,
        anchors: rawOptimal.anchors
    };
}

// --- UTILITIES ---

function setStatus(message) {
    statusText.textContent = message;
}

function resizeCanvas() {
    const vw = video.videoWidth || 1280;
    const vh = video.videoHeight || 720;

    // Overlay stays at full resolution for crisp visuals
    canvas.width = vw;
    canvas.height = vh;

    // Detection canvas scaled down — MediaPipe doesn't need full-res
    const scale = Math.min(1, MAX_DETECTION_WIDTH / vw);
    inputCanvas.width = Math.round(vw * scale);
    inputCanvas.height = Math.round(vh * scale);
}

function resizeVelocityChart() {
    if (!velocityChart || !velocityChartCtx) return;
    const dpr = window.devicePixelRatio || 1;
    const cssWidth = velocityChart.clientWidth || 320;
    const cssHeight = velocityChart.clientHeight || 200;
    velocityChart.width = Math.floor(cssWidth * dpr);
    velocityChart.height = Math.floor(cssHeight * dpr);
    velocityChartCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function stopActiveStream() {
    if (video.srcObject) {
        for (const track of video.srcObject.getTracks()) track.stop();
        video.srcObject = null;
    }
}

function resetLoop() {
    lastVideoTime = -1;
    cogHistory.length = 0;
    cogPath.length = 0;
    prevSmoothedCOG = null;
    prevSmoothedOptimal = null;
    resetAnalytics();
    if (animationFrameId !== null) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
}

// --- DRAWING FUNCTIONS ---

/* 
function drawCogPath() {
    if (cogPath.length < 2) return;
    canvasContext.save();
    canvasContext.beginPath();
    canvasContext.moveTo(cogPath[0].x, cogPath[0].y);
    for (let i = 1; i < cogPath.length; i++) {
        canvasContext.lineTo(cogPath[i].x, cogPath[i].y);
    }
    canvasContext.strokeStyle = 'rgba(255, 209, 102, 0.4)';
    canvasContext.lineWidth = 2;
    canvasContext.stroke();
    canvasContext.restore();
} */

function drawResults(result) {
    canvasContext.save();
    canvasContext.clearRect(0, 0, canvas.width, canvas.height);
    // Draw from the original video (full-res), not the scaled-down detection canvas
    canvasContext.drawImage(video, 0, 0, canvas.width, canvas.height);

    const landmarks = result.landmarks?.[0];
    if (landmarks?.length) {
        // Scale factor: sizes are authored for 720p; scale proportionally to actual resolution
        const s = Math.max(canvas.width, canvas.height) / 720;

        // 1. Draw Skeleton
        drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
            color: 'rgba(103, 242, 196, 0.6)',
            lineWidth: 4 * s
        });
        drawingUtils.drawLandmarks(landmarks, { color: '#f7fbff', radius: 2 * s });

        // 2. Calculations
        const currentCog = calculateCOG(landmarks);
        const optimalData = calculateOptimalCOG(landmarks, currentCog);

        // Update Path
        cogPath.push({ x: currentCog.x * canvas.width, y: currentCog.y * canvas.height });
        if (cogPath.length > MAX_PATH_POINTS) cogPath.shift();
        // drawCogPath();

        // 3. Alignment Integration
        if (optimalData) {
            // Get both instant accuracy and session stability metrics
            const metrics = calculateDetailedMetrics(currentCog, optimalData);

            // Update the 'CoM Stability' card (Session Average)
            if (stabilityCurrent) {
                stabilityCurrent.textContent = `${metrics.session}%`;
                stabilityCurrent.style.color = parseFloat(metrics.session) > 70 ? '#67f2c4' : '#ffd166';
                stabilityCurrent.classList.remove('placeholder');
            }

            // Update the 'COG Accuracy' card (Real-time Snapshot)
            if (accuracyElement) {
                accuracyElement.textContent = `${metrics.instant}%`;
                accuracyElement.style.color = parseFloat(metrics.instant) > 80 ? '#4ade80' : '#facc15';
                accuracyElement.classList.remove('placeholder');
            }

            // The Axis of Tension (Line connecting Hands to Feet)
            /* canvasContext.setLineDash([5 * s, 5 * s]);
             canvasContext.beginPath();
             canvasContext.moveTo(optimalData.anchors.baseX * canvas.width, optimalData.anchors.baseY * canvas.height);
             canvasContext.lineTo(optimalData.anchors.pullX * canvas.width, optimalData.anchors.pullY * canvas.height);
             canvasContext.strokeStyle = 'rgba(0, 242, 255, 0.4)';
             canvasContext.lineWidth = 2 * s;
             canvasContext.stroke();
             canvasContext.setLineDash([]); LINE dash didn't look very good */

            // Effort Gap (Horizontal line between Current COG and the Tension Line)
            canvasContext.beginPath();
            canvasContext.moveTo(currentCog.x * canvas.width, currentCog.y * canvas.height);
            canvasContext.lineTo(optimalData.x * canvas.width, optimalData.y * canvas.height);
            canvasContext.strokeStyle = '#ff4d4d';
            canvasContext.lineWidth = 3 * s;
            canvasContext.stroke();

            // Achievable Optimal Point on the Tension Line
            canvasContext.beginPath();
            canvasContext.arc(optimalData.x * canvas.width, optimalData.y * canvas.height, 6 * s, 0, Math.PI * 2);
            canvasContext.fillStyle = '#00f2ff';
            canvasContext.fill();
        }

        // 4. Draw Current COG
        const cx = currentCog.x * canvas.width;
        const cy = currentCog.y * canvas.height;

        canvasContext.beginPath();
        canvasContext.arc(cx, cy, 12 * s, 0, Math.PI * 2);
        canvasContext.strokeStyle = '#ffd166';
        canvasContext.lineWidth = 3 * s;
        canvasContext.stroke();

        canvasContext.beginPath();
        canvasContext.arc(cx, cy, 4 * s, 0, Math.PI * 2);
        canvasContext.fillStyle = '#ffd166';
        canvasContext.fill();

        // Labels
        canvasContext.fillStyle = '#ffd166';
        canvasContext.font = `bold ${Math.round(14 * s)}px Inter, sans-serif`;
        canvasContext.fillText('CURRENT', cx + 16 * s, cy - 6 * s);

        if (optimalData) {
            canvasContext.fillStyle = '#00f2ff';
            canvasContext.fillText('Optimal COM', (optimalData.x * canvas.width) + 16 * s, (optimalData.y * canvas.height) + 16 * s);
        }

        // --- CALCULATE DATA ANALYTICS ---
        // Compute body height for normalization (head → ankle midpoint, in normalized coords)
        const ankleMid = getMidpoint(landmarks[27], landmarks[28]);
        const bodyHeight = Math.sqrt(
            Math.pow(landmarks[0].x - ankleMid.x, 2) +
            Math.pow(landmarks[0].y - ankleMid.y, 2)
        );
        analyzeSmoothness(cogHistory, canvasContext, canvas.width, canvas.height, s, bodyHeight);

        // Velocity chart redraws every frame (it's just canvas lines — cheap)
        if (velocityChartCtx) {
            drawVelocityChart(
                velocityChartCtx,
                velocityChart.width / (window.devicePixelRatio || 1),
                velocityChart.height / (window.devicePixelRatio || 1)
            );
        }

        // DOM text writes throttled — browsers can struggle with layout at 60Hz
        if (frameCount % UI_UPDATE_EVERY_N_FRAMES === 0) {
            if (velocityCurrent) {
                // %bh/f: percent body heights per frame (give description in tooltip)
                velocityCurrent.textContent = `${(getCurrentVelocity() * 100).toFixed(1)}% bh/f`;
            }
        }
    }

    canvasContext.restore();
}

// --- CORE ENGINE ---

const MODEL_URLS = {
    lite: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
    full: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
    heavy: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
};

let _visionInstance = null; // cache the WASM fileset so reloads don't re-download it

async function loadLandmarker(modelKey, delegate) {
    const modelUrl = MODEL_URLS[modelKey] || MODEL_URLS.full;
    setStatus(`Loading ${modelKey} model (${delegate})...`);

    // Close previous instance if reloading
    if (poseLandmarker) {
        poseLandmarker.close();
        poseLandmarker = null;
    }

    if (!_visionInstance) {
        _visionInstance = await FilesetResolver.forVisionTasks(WASM_URL);
    }

    poseLandmarker = await PoseLandmarker.createFromOptions(_visionInstance, {
        baseOptions: {
            modelAssetPath: modelUrl,
            delegate: delegate
        },
        runningMode: 'VIDEO',
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
    });

    setStatus(`Ready — ${modelKey} (${delegate})`);

    // Pre-warm: compile GPU shaders during loading, not on first frame
    try {
        const warmupCanvas = document.createElement('canvas');
        warmupCanvas.width = MAX_DETECTION_WIDTH;
        warmupCanvas.height = Math.round(MAX_DETECTION_WIDTH * 9 / 16);
        poseLandmarker.detectForVideo(warmupCanvas, performance.now());
    } catch (_) { /* warmup failure is harmless */ }

    return poseLandmarker;
}

function trackFrame() {
    if (!poseLandmarker || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        animationFrameId = requestAnimationFrame(trackFrame);
        return;
    }

    frameCount++;

    if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;

        // Only run heavy ML inference every N frames; reuse cached result otherwise
        if (frameCount % DETECT_EVERY_N_FRAMES === 0) {
            inputCtx.clearRect(0, 0, inputCanvas.width, inputCanvas.height);
            inputCtx.drawImage(video, 0, 0, inputCanvas.width, inputCanvas.height);
            lastDetectionResult = poseLandmarker.detectForVideo(inputCanvas, performance.now());
        }

        if (lastDetectionResult) {
            drawResults(lastDetectionResult);
        }
    }
    animationFrameId = requestAnimationFrame(trackFrame);
}

// --- EVENT LISTENERS ---

webcamButton.addEventListener('click', async () => {
    try {
        stopActiveStream();
        resetLoop();
        video.srcObject = await navigator.mediaDevices.getUserMedia({ video: true });
        await video.play();
        playPauseBtn.disabled = true;
        seekBar.disabled = true;
        resizeCanvas();
        trackFrame();
        setStatus('Webcam active.');
    } catch (e) { setStatus('Webcam error.'); }
});

videoFileInput.addEventListener('change', async () => {
    const file = videoFileInput.files?.[0];
    if (!file) return;
    stopActiveStream();
    resetLoop();
    if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = URL.createObjectURL(file);
    video.src = currentObjectUrl;
    video.onloadedmetadata = () => {
        playPauseBtn.disabled = false;
        seekBar.disabled = false;
        speedControl.disabled = false;
        prevFrameBtn.disabled = false;
        nextFrameBtn.disabled = false;
        playPauseBtn.textContent = 'Pause';
        resizeCanvas();
        video.play();
        trackFrame();
        setStatus(`Analyzing: ${file.name}`);
    };
});

// --- VIDEO CONTROLS ---

function formatTime(seconds) {
    if (isNaN(seconds)) return "0:00";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

video.addEventListener('timeupdate', () => {
    if (video.duration) {
        seekBar.value = (video.currentTime / video.duration) * 100;
        timeDisplay.textContent = `${formatTime(video.currentTime)} / ${formatTime(video.duration)}`;
    }
});

playPauseBtn.addEventListener('click', () => {
    if (video.paused) {
        video.play();
        playPauseBtn.textContent = 'Pause';
    } else {
        video.pause();
        playPauseBtn.textContent = 'Play';
    }
});

seekBar.addEventListener('input', () => {
    const time = (seekBar.value / 100) * video.duration;
    video.currentTime = time;
    // Clear histories and EMA state so drawing doesn't jump
    cogHistory.length = 0;
    cogPath.length = 0;
    prevSmoothedCOG = null;
    prevSmoothedOptimal = null;
});

speedControl.addEventListener('change', () => {
    video.playbackRate = parseFloat(speedControl.value);
});

// Frame stepping functions
async function stepFrame(direction) {
    if (!video.duration || !poseLandmarker) return;

    // Pause the video if playing
    if (!video.paused) {
        video.pause();
        playPauseBtn.textContent = 'Play';
    }

    // Estimate frame duration (assuming 30fps, adjust if needed)
    const fps = 30;
    const frameDuration = 1 / fps;

    // Step forward or backward by one frame
    video.currentTime = Math.max(0, Math.min(video.duration, video.currentTime + (direction * frameDuration)));

    // Clear histories and EMA state so drawing doesn't jump
    cogHistory.length = 0;
    cogPath.length = 0;
    prevSmoothedCOG = null;
    prevSmoothedOptimal = null;

    // Wait for the video to seek to the new time, then manually process the frame
    await new Promise(resolve => {
        const onSeeked = () => {
            video.removeEventListener('seeked', onSeeked);
            resolve();
        };
        video.addEventListener('seeked', onSeeked);
    });

    // Manually trigger frame processing
    if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        inputCtx.clearRect(0, 0, inputCanvas.width, inputCanvas.height);
        inputCtx.drawImage(video, 0, 0, inputCanvas.width, inputCanvas.height);
        lastDetectionResult = poseLandmarker.detectForVideo(inputCanvas, performance.now());
        if (lastDetectionResult) {
            drawResults(lastDetectionResult);
        }
    }
}

prevFrameBtn.addEventListener('click', () => stepFrame(-1));
nextFrameBtn.addEventListener('click', () => stepFrame(1));

// --- MODEL / DELEGATE SWITCHING ---
async function reloadModel() {
    const modelKey = modelSelect.value;
    const delegate = delegateSelect.value;
    // Disable selects during load to prevent double-clicks
    modelSelect.disabled = true;
    delegateSelect.disabled = true;
    try {
        await loadLandmarker(modelKey, delegate);
    } catch (e) {
        setStatus(`Error loading ${modelKey} (${delegate}) — ${e.message}`);
    }
    modelSelect.disabled = false;
    delegateSelect.disabled = false;
}

modelSelect.addEventListener('change', reloadModel);
delegateSelect.addEventListener('change', reloadModel);

// Initialization
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm';
await loadLandmarker(modelSelect.value, delegateSelect.value);
video.addEventListener('loadedmetadata', resizeCanvas);
window.addEventListener('resize', resizeVelocityChart);
resizeVelocityChart();

// --- POST-VIDEO SUMMARY ---
video.addEventListener('ended', () => {
    const summary = getSessionSummary();
    showSessionSummary(summary, {
        onReplay: () => {
            resetLoop();
            video.currentTime = 0;
            video.play();
            playPauseBtn.textContent = 'Pause';
            trackFrame();
        },
        onDismiss: () => {
            playPauseBtn.textContent = 'Play';
        }
    });
    playPauseBtn.textContent = 'Play';
});