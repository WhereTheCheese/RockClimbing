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
const SMOOTHING_WINDOW = 5; // Average over 5 frames to reduce jitter

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
    if (animationFrameId !== null) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
}

// --- ADDED: CONFIG FOR OPTIMAL TRACKING ---
const optimalHistory = [];

function calculateOptimalCOG(landmarks, currentCog) {
    // We define the base of support using the ankles (27, 28)
    const leftAnkle = landmarks[27];
    const rightAnkle = landmarks[28];
    
    if (!leftAnkle || !rightAnkle) return null;

    // Optimal X is the midpoint between the feet
    const baseMidpointX = (leftAnkle.x + rightAnkle.x) / 2;

    // We keep the Current Y (vertical height) but move the X to the "Perfect Balance" line
    const rawOptimal = {
        x: baseMidpointX,
        y: currentCog.y
    };

    // Smooth the optimal point too
    optimalHistory.push(rawOptimal);
    if (optimalHistory.length > 5) optimalHistory.shift();

    return optimalHistory.reduce((acc, curr) => ({
        x: acc.x + curr.x / optimalHistory.length,
        y: acc.y + curr.y / optimalHistory.length
    }), { x: 0, y: 0 });
}

// --- UPDATED DRAWING LOGIC ---
function drawResults(result) {
    canvasContext.save();
    canvasContext.clearRect(0, 0, canvas.width, canvas.height);
    canvasContext.drawImage(inputCanvas, 0, 0, canvas.width, canvas.height);

    const landmarks = result.landmarks?.[0];
    if (landmarks?.length) {
        // 1. Draw Skeleton
        drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
            color: 'rgba(103, 242, 196, 0.5)', // Faded green for skeleton
            lineWidth: 2
        });

        // 2. Calculate COGs
        const currentCog = calculateCOG(landmarks);
        const optimalCog = calculateOptimalCOG(landmarks, currentCog);

        if (optimalCog) {
            // 3. Draw Balance Line (Vertical dashed line from the base)
            canvasContext.setLineDash([5, 5]);
            canvasContext.beginPath();
            canvasContext.moveTo(optimalCog.x * canvas.width, 0);
            canvasContext.lineTo(optimalCog.x * canvas.width, canvas.height);
            canvasContext.strokeStyle = 'rgba(0, 255, 255, 0.3)';
            canvasContext.stroke();
            canvasContext.setLineDash([]);

            // 4. Draw Offset Line (The gap between current and optimal)
            canvasContext.beginPath();
            canvasContext.moveTo(currentCog.x * canvas.width, currentCog.y * canvas.height);
            canvasContext.lineTo(optimalCog.x * canvas.width, optimalCog.y * canvas.height);
            canvasContext.strokeStyle = '#ff4d4d'; // Red for the "struggle" gap
            canvasContext.lineWidth = 2;
            canvasContext.stroke();

            // 5. Draw Optimal COG (The "Goal")
            canvasContext.beginPath();
            canvasContext.arc(optimalCog.x * canvas.width, optimalCog.y * canvas.height, 8, 0, Math.PI * 2);
            canvasContext.fillStyle = '#00f2ff'; // Cyan for optimal
            canvasContext.fill();
        }

        // 6. Draw Current COG (The "Reality")
        canvasContext.beginPath();
        canvasContext.arc(currentCog.x * canvas.width, currentCog.y * canvas.height, 10, 0, Math.PI * 2);
        canvasContext.fillStyle = '#ffd166'; // Yellow for current
        canvasContext.fill();
        canvasContext.strokeStyle = '#000';
        canvasContext.stroke();

        // 7. Legend / Text
        canvasContext.font = '12px Inter';
        canvasContext.fillStyle = '#ffd166';
        canvasContext.fillText('Current COG', (currentCog.x * canvas.width) + 15, (currentCog.y * canvas.height) - 5);
        if (optimalCog) {
            canvasContext.fillStyle = '#00f2ff';
            canvasContext.fillText('Optimal Balance', (optimalCog.x * canvas.width) + 15, (optimalCog.y * canvas.height) + 15);
        }
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
