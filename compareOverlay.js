import { DrawingUtils, FilesetResolver, PoseLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs';

// ─── CONSTANTS ────────────────────────────────────────────────────────────────
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task';
const MAX_DETECTION_WIDTH = 640;
const SMOOTHING_WINDOW = 5;
const MAX_PATH_POINTS = 300;

// ─── SHARED MEDIAPIPE INSTANCE ────────────────────────────────────────────────
let poseLandmarker;
let animationFrameId = null;
let frameCount = 0;

// ─── SHARED OFFSCREEN CANVAS (single instance, reused for both videos) ────────
const inputCanvas = document.createElement('canvas');
const inputCtx = inputCanvas.getContext('2d', { willReadFrequently: true });

// ─── DOM ──────────────────────────────────────────────────────────────────────
const statusText = document.getElementById('status-text');

// ─── TIME HELPER ──────────────────────────────────────────────────────────────
function formatTime(seconds) {
    if (isNaN(seconds)) return "0:00";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// ─── PER-VIDEO STATE FACTORY ──────────────────────────────────────────────────
/**
 * Creates an isolated state + DOM bundle for one video panel.
 * @param {'a'|'b'} id
 */
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
    const seekBar = document.getElementById(`seek-bar-${id}`);
    const timeDisplay = document.getElementById(`time-display-${id}`);

    return {
        id, video, canvas, ctx, drawUtils, fileInput, labelEl,
        stabilityEl, accuracyEl, velocityEl, seekBar, timeDisplay,
        cogHistory: [],
        optimalHistory: [],
        cogPath: [],
        lastVideoTime: -1,
        lastResult: null,
        objectUrl: null,
    };
}

const panels = [createPanel('a'), createPanel('b')];

// ─── MATH HELPERS ─────────────────────────────────────────────────────────────
function midpoint(p1, p2) {
    return { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2, z: (p1.z + p2.z) / 2 };
}

function calculateCOG(landmarks, panel) {
    const head = landmarks[0];
    const trunk = midpoint(
        midpoint(landmarks[11], landmarks[12]),
        midpoint(landmarks[23], landmarks[24])
    );
    const segments = [
        { pos: head, weight: 0.08 },
        { pos: trunk, weight: 0.50 },
        { pos: midpoint(landmarks[23], landmarks[25]), weight: 0.10 },
        { pos: midpoint(landmarks[24], landmarks[26]), weight: 0.10 },
        { pos: midpoint(landmarks[25], landmarks[27]), weight: 0.06 },
        { pos: midpoint(landmarks[26], landmarks[28]), weight: 0.06 },
        { pos: midpoint(landmarks[11], landmarks[13]), weight: 0.03 },
        { pos: midpoint(landmarks[12], landmarks[14]), weight: 0.03 },
        { pos: midpoint(landmarks[13], landmarks[15]), weight: 0.02 },
        { pos: midpoint(landmarks[14], landmarks[16]), weight: 0.02 },
    ];
    let raw = { x: 0, y: 0 };
    segments.forEach(s => { raw.x += s.pos.x * s.weight; raw.y += s.pos.y * s.weight; });

    const h = panel.cogHistory;
    h.push(raw);
    if (h.length > SMOOTHING_WINDOW) h.shift();
    return h.reduce((acc, c) => ({ x: acc.x + c.x / h.length, y: acc.y + c.y / h.length }), { x: 0, y: 0 });
}

function calculateOptimalCOG(landmarks, cog, panel) {
    const la = landmarks[27], ra = landmarks[28];
    const lw = landmarks[15], rw = landmarks[16];
    if (!la || !ra || !lw || !rw) return null;

    const baseX = (la.x + ra.x) / 2, baseY = (la.y + ra.y) / 2;
    const pullX = (lw.x + rw.x) / 2, pullY = (lw.y + rw.y) / 2;
    let optX = Math.abs(pullY - baseY) < 0.001
        ? baseX
        : baseX + ((cog.y - baseY) / (pullY - baseY)) * (pullX - baseX);

    const raw = { x: optX, y: cog.y, anchors: { baseX, baseY, pullX, pullY } };
    const oh = panel.optimalHistory;
    oh.push(raw);
    if (oh.length > SMOOTHING_WINDOW) oh.shift();

    return oh.reduce((acc, c) => ({
        x: acc.x + c.x / oh.length,
        y: acc.y + c.y / oh.length,
        anchors: raw.anchors
    }), { x: 0, y: 0 });
}

// ─── DRAWING ──────────────────────────────────────────────────────────────────
function drawPanel(panel) {
    const { ctx, canvas, video, drawUtils, cogHistory, cogPath, optimalHistory,
        stabilityEl, accuracyEl, velocityEl, lastResult } = panel;
    if (!lastResult) return;

    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const landmarks = lastResult.landmarks?.[0];
    if (!landmarks?.length) { ctx.restore(); return; }

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

        // Optimal point
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

    // Smoothness box (reuse same math, draw on panel canvas)
    if (cogHistory.length > 1) {
        const prev = cogHistory[cogHistory.length - 2];
        const curr = cogHistory[cogHistory.length - 1];
        const vel = Math.sqrt(
            Math.pow((curr.x - prev.x) * canvas.width, 2) +
            Math.pow((curr.y - prev.y) * canvas.height, 2)
        );

        // Update DOM readouts
        if (frameCount % 2 === 0) {
            if (velocityEl) velocityEl.textContent = `${vel.toFixed(1)} px/f`;
        }
    }

    // COG accuracy readout
    if (optimal && stabilityEl && accuracyEl) {
        const gap = Math.abs(cog.x - optimal.x);
        const instant = Math.max(0, Math.min(100, (1 - gap / 0.1) * 100));
        if (frameCount % 2 === 0) {
            accuracyEl.textContent = `${instant.toFixed(1)}%`;
            accuracyEl.style.color = instant > 80 ? '#4ade80' : '#facc15';
        }
    }

    // Smoothness overlay box
    const bx = 16 * s, by = 16 * s, bw = 180 * s, bh = 70 * s;
    ctx.fillStyle = 'rgba(9,18,30,0.72)';
    ctx.roundRect(bx, by, bw, bh, 10 * s);
    ctx.fill();
    ctx.strokeStyle = 'rgba(131,160,194,0.3)';
    ctx.lineWidth = s;
    ctx.stroke();
    ctx.fillStyle = '#67f2c4';
    ctx.font = `bold ${Math.round(28 * s)}px Inter, sans-serif`;
    ctx.fillText(panel.id.toUpperCase() === 'A' ? 'Run A' : 'Run B', bx + 14 * s, by + 44 * s);

    ctx.restore();
}

// ─── CANVAS RESIZE ────────────────────────────────────────────────────────────
function resizePanel(panel) {
    const vw = panel.video.videoWidth || 1280;
    const vh = panel.video.videoHeight || 720;
    panel.canvas.width = vw;
    panel.canvas.height = vh;
    // Keep inputCanvas sized to the largest we'll process
    const scale = Math.min(1, MAX_DETECTION_WIDTH / Math.max(vw, 1));
    inputCanvas.width = Math.round(vw * scale);
    inputCanvas.height = Math.round(vh * scale);
}

// ─── CORE LOOP ────────────────────────────────────────────────────────────────
function trackFrame() {
    if (!poseLandmarker) {
        animationFrameId = requestAnimationFrame(trackFrame);
        return;
    }

    frameCount++;

    // Stagger: even frames → panel A, odd frames → panel B
    // Net inference cost = same as a single video
    const targets = frameCount % 2 === 0
        ? [panels[0], panels[1]]   // A then (draw) B
        : [panels[1], panels[0]];  // B then (draw) A

    const detect = targets[0];
    if (detect.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
        detect.video.currentTime !== detect.lastVideoTime) {

        detect.lastVideoTime = detect.video.currentTime;
        resizePanel(detect); // keep inputCanvas sized to current video
        inputCtx.clearRect(0, 0, inputCanvas.width, inputCanvas.height);
        inputCtx.drawImage(detect.video, 0, 0, inputCanvas.width, inputCanvas.height);
        detect.lastResult = poseLandmarker.detectForVideo(inputCanvas, performance.now());
    }

    // Draw both panels every rAF tick
    panels.forEach(drawPanel);

    animationFrameId = requestAnimationFrame(trackFrame);
}

// ─── MODEL LOADING ────────────────────────────────────────────────────────────
async function loadLandmarker() {
    setStatus('Loading MediaPipe model...');
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: MODEL_URL,
            delegate: 'GPU'
        },
        runningMode: 'VIDEO',
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
    });

    // GPU pre-warm
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

// ─── FILE & SEEK HANDLERS ─────────────────────────────────────────────────────
panels.forEach(panel => {
    
    // Update seek bar UI as video plays
    panel.video.addEventListener('timeupdate', () => {
        if (panel.video.duration && panel.seekBar) {
            panel.seekBar.value = (panel.video.currentTime / panel.video.duration) * 100;
            if (panel.timeDisplay) {
                panel.timeDisplay.textContent = `${formatTime(panel.video.currentTime)} / ${formatTime(panel.video.duration)}`;
            }
        }
    });

    // Handle manual seek adjustment
    if (panel.seekBar) {
        panel.seekBar.addEventListener('input', () => {
            const time = (panel.seekBar.value / 100) * panel.video.duration;
            panel.video.currentTime = time;
            
            // Clear tracking histories so the drawn lines don't glitch/jump across the screen
            panel.cogHistory.length = 0;
            panel.optimalHistory.length = 0;
            panel.cogPath.length = 0;
        });
    }

    panel.fileInput.addEventListener('change', () => {
        const file = panel.fileInput.files?.[0];
        if (!file) return;
        if (panel.objectUrl) URL.revokeObjectURL(panel.objectUrl);
        panel.objectUrl = URL.createObjectURL(file);
        panel.video.src = panel.objectUrl;
        panel.cogHistory.length = 0;
        panel.optimalHistory.length = 0;
        panel.cogPath.length = 0;
        panel.lastVideoTime = -1;
        panel.lastResult = null;
        
        panel.video.onloadedmetadata = () => {
            resizePanel(panel);
            panel.video.play();
            if (panel.labelEl) panel.labelEl.textContent = file.name;
            if (panel.seekBar) panel.seekBar.disabled = false;
            setStatus(`${panel.id.toUpperCase()}: ${file.name}`);
        };

        // Start loop if not already running
        if (animationFrameId === null) trackFrame();
    });
});

// ─── BOOT ─────────────────────────────────────────────────────────────────────
await loadLandmarker();
trackFrame(); // start loop immediately so we're ready when files are uploaded
