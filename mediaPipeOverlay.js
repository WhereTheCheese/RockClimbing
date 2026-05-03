import { DrawingUtils, FilesetResolver, PoseLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs';
import {
    resetAnalytics,
    analyzeSmoothness,
    drawVelocityChart,
    getCurrentVelocity,
    getCurrentSmoothnessScore
} from './dataAnalysis.js';

const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const canvasContext = canvas.getContext('2d');

video.width = 640;
video.height = 480;


const inputCanvas = document.createElement('canvas');
const inputCtx = inputCanvas.getContext('2d', { willReadFrequently: true });

const drawingUtils = new DrawingUtils(canvasContext);
const webcamButton = document.getElementById('webcam-button');
const videoFileInput = document.getElementById('video-file');
const statusText = document.getElementById('status-text');

// Velocity graph elements
const velocityChart = document.getElementById('velocity-chart');
const velocityCurrent = document.getElementById('data-velocity');
const smoothnessCurrent = document.getElementById('data-stability');
const velocityChartCtx = velocityChart ? velocityChart.getContext('2d') : null;

let poseLandmarker;
let animationFrameId = null;
let lastVideoTime = -1;
let currentObjectUrl = null;

// --- CONFIGURATION & TRACKING HISTORY ---
const cogHistory = [];
const optimalHistory = [];
const cogPath = [];
const SMOOTHING_WINDOW = 5;
const MAX_PATH_POINTS = 300;
const MAX_DETECTION_WIDTH = 640; // Scale down for MediaPipe — it doesn't need full-res frames

function getMidpoint(p1, p2) {
    return {
        x: (p1.x + p2.x) / 2,
        y: (p1.y + p2.y) / 2,
        z: (p1.z + p2.z) / 2
    };
}

/**
 * Calculates the anthropometric COG based on segment weights
 */
function calculateCOG(landmarks) {
    const head = landmarks[0];
    const shoulderMid = getMidpoint(landmarks[11], landmarks[12]);
    const hipMid = getMidpoint(landmarks[23], landmarks[24]);
    const trunk = getMidpoint(shoulderMid, hipMid);

    const segments = [
        { pos: head, weight: 0.08 },
        { pos: trunk, weight: 0.50 },
        { pos: getMidpoint(landmarks[23], landmarks[25]), weight: 0.10 }, // R-Thigh
        { pos: getMidpoint(landmarks[24], landmarks[26]), weight: 0.10 }, // L-Thigh
        { pos: getMidpoint(landmarks[25], landmarks[27]), weight: 0.06 }, // R-Leg
        { pos: getMidpoint(landmarks[26], landmarks[28]), weight: 0.06 }, // L-Leg
        { pos: getMidpoint(landmarks[11], landmarks[13]), weight: 0.03 }, // R-UpperArm
        { pos: getMidpoint(landmarks[12], landmarks[14]), weight: 0.03 }, // L-UpperArm
        { pos: getMidpoint(landmarks[13], landmarks[15]), weight: 0.02 }, // R-Forearm
        { pos: getMidpoint(landmarks[14], landmarks[16]), weight: 0.02 }, // L-Forearm
    ];

    let rawCOG = { x: 0, y: 0 };
    segments.forEach(s => {
        rawCOG.x += s.pos.x * s.weight;
        rawCOG.y += s.pos.y * s.weight;
    });

    cogHistory.push(rawCOG);
    if (cogHistory.length > SMOOTHING_WINDOW) cogHistory.shift();

    return cogHistory.reduce((acc, curr) => ({
        x: acc.x + curr.x / cogHistory.length,
        y: acc.y + curr.y / cogHistory.length
    }), { x: 0, y: 0 });
}

/**
 * Calculates the "Achievable Optimal X" by interpolating between the 
 * base of support (feet) and the upper anchor (hands) at the current COG height.
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
    // Prevent divide by zero if hands and feet are exactly horizontal (e.g., severe heel hook)
    if (Math.abs(pullY - baseY) < 0.001) {
        optimalX = baseX;
    } else {
        // Calculate the percentage of height (t) the COG is at between feet and hands
        const t = (currentCog.y - baseY) / (pullY - baseY);
        // Map that percentage to the X axis
        optimalX = baseX + t * (pullX - baseX);
    }

    const rawOptimal = {
        x: optimalX,
        y: currentCog.y,
        // We return the anchor points to draw the Tension Line in drawResults
        anchors: { baseX, baseY, pullX, pullY }
    };

    optimalHistory.push(rawOptimal);
    if (optimalHistory.length > SMOOTHING_WINDOW) optimalHistory.shift();

    const smoothed = optimalHistory.reduce((acc, curr) => ({
        x: acc.x + curr.x / optimalHistory.length,
        y: acc.y + curr.y / optimalHistory.length,
        anchors: rawOptimal.anchors // Keep current anchors for drawing
    }), { x: 0, y: 0 });

    return smoothed;
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
    optimalHistory.length = 0;
    cogPath.length = 0;
    resetAnalytics();
    if (animationFrameId !== null) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
}

// --- DRAWING FUNCTIONS ---

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
}

function drawResults(result) {
    canvasContext.save();
    canvasContext.clearRect(0, 0, canvas.width, canvas.height);
    // Draw from the original video (full-res), not the scaled-down detection canvas
    canvasContext.drawImage(video, 0, 0, canvas.width, canvas.height);

    const landmarks = result.landmarks?.[0];
    if (landmarks?.length) {
        // 1. Draw Skeleton
        drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
            color: 'rgba(103, 242, 196, 0.6)',
            lineWidth: 2
        });
        drawingUtils.drawLandmarks(landmarks, { color: '#f7fbff', radius: 1 });

        // 2. Calculations
        const currentCog = calculateCOG(landmarks);
        const optimalData = calculateOptimalCOG(landmarks, currentCog);

        // Update Path
        cogPath.push({ x: currentCog.x * canvas.width, y: currentCog.y * canvas.height });
        if (cogPath.length > MAX_PATH_POINTS) cogPath.shift();
        drawCogPath();

        // 3. Draw Optimal Elements
        if (optimalData) {
            // The Axis of Tension (Line connecting Hands to Feet)
            canvasContext.setLineDash([5, 5]);
            canvasContext.beginPath();
            canvasContext.moveTo(optimalData.anchors.baseX * canvas.width, optimalData.anchors.baseY * canvas.height);
            canvasContext.lineTo(optimalData.anchors.pullX * canvas.width, optimalData.anchors.pullY * canvas.height);
            canvasContext.strokeStyle = 'rgba(0, 242, 255, 0.4)';
            canvasContext.lineWidth = 2;
            canvasContext.stroke();
            canvasContext.setLineDash([]);

            // Effort Gap (Horizontal line between Current COG and the Tension Line)
            canvasContext.beginPath();
            canvasContext.moveTo(currentCog.x * canvas.width, currentCog.y * canvas.height);
            canvasContext.lineTo(optimalData.x * canvas.width, optimalData.y * canvas.height);
            canvasContext.strokeStyle = '#ff4d4d';
            canvasContext.lineWidth = 3;
            canvasContext.stroke();

            // Achievable Optimal Point on the Tension Line
            canvasContext.beginPath();
            canvasContext.arc(optimalData.x * canvas.width, optimalData.y * canvas.height, 6, 0, Math.PI * 2);
            canvasContext.fillStyle = '#00f2ff';
            canvasContext.fill();
        }

        // 4. Draw Current COG
        const cx = currentCog.x * canvas.width;
        const cy = currentCog.y * canvas.height;

        canvasContext.beginPath();
        canvasContext.arc(cx, cy, 12, 0, Math.PI * 2);
        canvasContext.strokeStyle = '#ffd166';
        canvasContext.lineWidth = 3;
        canvasContext.stroke();

        canvasContext.beginPath();
        canvasContext.arc(cx, cy, 4, 0, Math.PI * 2);
        canvasContext.fillStyle = '#ffd166';
        canvasContext.fill();

        // Labels
        canvasContext.fillStyle = '#ffd166';
        canvasContext.font = 'bold 12px Inter, sans-serif';
        canvasContext.fillText('CURRENT', cx + 15, cy - 5);

        if (optimalData) {
            canvasContext.fillStyle = '#00f2ff';
            canvasContext.fillText('Optimal COM', (optimalData.x * canvas.width) + 15, (optimalData.y * canvas.height) + 15);
        }

        // --- CALCULATE DATA ANALYTICS ---
        analyzeSmoothness(cogHistory, canvasContext, canvas.width, canvas.height);

        // Velocity graph of center of mass movement
        if (velocityChartCtx) {
            drawVelocityChart(
                velocityChartCtx,
                velocityChart.width / (window.devicePixelRatio || 1),
                velocityChart.height / (window.devicePixelRatio || 1)
            );
        }
        if (velocityCurrent) {
            velocityCurrent.textContent = `${getCurrentVelocity().toFixed(1)} px/frame`;
        }
        if (smoothnessCurrent) {
            smoothnessCurrent.textContent = getCurrentSmoothnessScore().toFixed(0);
        }
    }

    canvasContext.restore();
}

// --- CORE ENGINE ---

async function loadLandmarker() {
    if (poseLandmarker) return poseLandmarker;
    setStatus('Loading MediaPipe model...');
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task' },
        runningMode: 'VIDEO',
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
    });
    setStatus('Ready.');
    return poseLandmarker;
}

function trackFrame() {
    if (!poseLandmarker || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        animationFrameId = requestAnimationFrame(trackFrame);
        return;
    }

    if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        inputCtx.clearRect(0, 0, inputCanvas.width, inputCanvas.height);
        inputCtx.drawImage(video, 0, 0, inputCanvas.width, inputCanvas.height);
        const result = poseLandmarker.detectForVideo(inputCanvas, performance.now());
        drawResults(result);
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
        resizeCanvas();
        video.play();
        trackFrame();
        setStatus(`Analyzing: ${file.name}`);
    };
});

// Initialization
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm';
await loadLandmarker();
video.addEventListener('loadedmetadata', resizeCanvas);
window.addEventListener('resize', resizeVelocityChart);
resizeVelocityChart();
