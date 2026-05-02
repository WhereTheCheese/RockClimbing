import { DrawingUtils, FilesetResolver, PoseLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs';

const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const canvasContext = canvas.getContext('2d');
const inputCanvas = document.createElement('canvas');
const inputCtx = inputCanvas.getContext('2d');
const drawingUtils = new DrawingUtils(canvasContext);
const webcamButton = document.getElementById('webcam-button');
const videoFileInput = document.getElementById('video-file');
const statusText = document.getElementById('status-text');

let poseLandmarker;
let animationFrameId = null;
let lastVideoTime = -1;

// --- COG TRACKING CONFIGURATION ---
const cogHistory = []; 
const cogPath = [];
const SMOOTHING_WINDOW = 5; // Average over 5 frames to reduce jitter
const MAX_PATH_POINTS = 300;

function getMidpoint(p1, p2) {
    return {
        x: (p1.x + p2.x) / 2,
        y: (p1.y + p2.y) / 2,
        z: (p1.z + p2.z) / 2
    };
}

function calculateCOG(landmarks) {
    // 1. Define Key Segments
    const head = landmarks[0]; // Nose
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

    // 2. Apply Smoothing
    cogHistory.push(rawCOG);
    if (cogHistory.length > SMOOTHING_WINDOW) cogHistory.shift();

    const smoothedCOG = cogHistory.reduce((acc, curr) => ({
        x: acc.x + curr.x / cogHistory.length,
        y: acc.y + curr.y / cogHistory.length
    }), { x: 0, y: 0 });

    return smoothedCOG;
}
// ----------------------------------

const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm';

function setStatus(message) {
    statusText.textContent = message;
}

function resizeCanvas() {
    const vw = video.videoWidth || 1280;
    const vh = video.videoHeight || 720;
    if (vw && vh && vw < vh) {
        inputCanvas.width = vh;
        inputCanvas.height = vw;
        canvas.width = vh;
        canvas.height = vw;
    } else {
        inputCanvas.width = vw;
        inputCanvas.height = vh;
        canvas.width = vw;
        canvas.height = vh;
    }
}

function stopActiveStream() {
    if (video.srcObject) {
        for (const track of video.srcObject.getTracks()) {
            track.stop();
        }
        video.srcObject = null;
    }
}

function resetLoop() {
    lastVideoTime = -1;
    cogHistory.length = 0; // Clear history on new video/webcam
    cogPath.length = 0;
    if (animationFrameId !== null) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
}

function drawCogPath() {
    if (cogPath.length < 2) {
        return;
    }

    canvasContext.save();
    canvasContext.beginPath();
    canvasContext.moveTo(cogPath[0].x, cogPath[0].y);
    for (let i = 1; i < cogPath.length; i += 1) {
        canvasContext.lineTo(cogPath[i].x, cogPath[i].y);
    }
    canvasContext.strokeStyle = '#ffd166';
    canvasContext.lineWidth = 3;
    canvasContext.lineJoin = 'round';
    canvasContext.lineCap = 'round';
    canvasContext.stroke();
    canvasContext.restore();
}

function drawResults(result) {
    canvasContext.save();
    canvasContext.clearRect(0, 0, canvas.width, canvas.height);
    canvasContext.drawImage(inputCanvas, 0, 0, canvas.width, canvas.height);

    const landmarks = result.landmarks?.[0];
    if (landmarks?.length) {
        // KEEP SKELETON OUTLINE
        drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
            color: '#67f2c4',
            lineWidth: 2
        });
        drawingUtils.drawLandmarks(landmarks, {
            color: '#f7fbff',
            radius: 2
        });

        // CALCULATE AND DRAW COG
        const cog = calculateCOG(landmarks);
        cogPath.push({
            x: cog.x * canvas.width,
            y: cog.y * canvas.height
        });
        if (cogPath.length > MAX_PATH_POINTS) {
            cogPath.shift();
        }

        drawCogPath();

        // Draw COG Outer Ring
        canvasContext.beginPath();
        canvasContext.arc(cog.x * canvas.width, cog.y * canvas.height, 12, 0, Math.PI * 2);
        canvasContext.strokeStyle = '#ffd166';
        canvasContext.lineWidth = 3;
        canvasContext.stroke();

        // Draw COG Center Point
        canvasContext.beginPath();
        canvasContext.arc(cog.x * canvas.width, cog.y * canvas.height, 4, 0, Math.PI * 2);
        canvasContext.fillStyle = '#ffd166';
        canvasContext.fill();
        
        // Label the COG
        canvasContext.fillStyle = '#ffd166';
        canvasContext.font = 'bold 12px Inter, sans-serif';
        canvasContext.fillText('COG', (cog.x * canvas.width) + 15, (cog.y * canvas.height) + 5);
    }

    canvasContext.restore();
}

async function loadLandmarker() {
    if (poseLandmarker) return poseLandmarker;
    setStatus('Loading MediaPipe model...');
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL },
        runningMode: 'VIDEO',
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
    });
    setStatus('Model ready.');
    return poseLandmarker;
}

function trackFrame() {
    if (!poseLandmarker || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        animationFrameId = requestAnimationFrame(trackFrame);
        return;
    }

    const vw = video.videoWidth || 0;
    const vh = video.videoHeight || 0;
    if (vw && vh && vw < vh) {
        inputCtx.save();
        inputCtx.clearRect(0, 0, inputCanvas.width, inputCanvas.height);
        inputCtx.translate(inputCanvas.width / 2, inputCanvas.height / 2);
        inputCtx.rotate(-Math.PI / 2);
        inputCtx.drawImage(video, -vw / 2, -vh / 2, vw, vh);
        inputCtx.restore();
    } else {
        inputCtx.clearRect(0, 0, inputCanvas.width, inputCanvas.height);
        inputCtx.drawImage(video, 0, 0, inputCanvas.width, inputCanvas.height);
    }

    if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        const result = poseLandmarker.detectForVideo(inputCanvas, performance.now());
        drawResults(result);
    }

    animationFrameId = requestAnimationFrame(trackFrame);
}

async function startTracking() {
    resizeCanvas();
    await loadLandmarker();
    resetLoop();
    trackFrame();
}

webcamButton.addEventListener('click', async () => {
    try {
        stopActiveStream();
        resetLoop();
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        await video.play();
        await startTracking();
        setStatus('Webcam active.');
    } catch (error) {
        setStatus('Webcam error. Check permissions.');
    }
});

videoFileInput.addEventListener('change', async () => {
    const file = videoFileInput.files?.[0];
    if (!file) return;
    stopActiveStream();
    resetLoop();
    video.src = URL.createObjectURL(file);
    video.onloadedmetadata = async () => {
        resizeCanvas();
        await video.play();
        await startTracking();
    };
});

await loadLandmarker();
video.addEventListener('loadedmetadata', resizeCanvas);
