import { DrawingUtils, FilesetResolver, PoseLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs';

        const video = document.getElementById('video');
        const canvas = document.getElementById('overlay');
        const canvasContext = canvas.getContext('2d');
        // Offscreen/input canvas used to normalize orientation for detection
        const inputCanvas = document.createElement('canvas');
        const inputCtx = inputCanvas.getContext('2d');
        const drawingUtils = new DrawingUtils(canvasContext);
        const webcamButton = document.getElementById('webcam-button');
        const videoFileInput = document.getElementById('video-file');
        const statusText = document.getElementById('status-text');

        let poseLandmarker;
        let animationFrameId = null;
        let lastVideoTime = -1;

        const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
        const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm';

        function setStatus(message) {
            statusText.textContent = message;
        }

        function resizeCanvas() {
            // If the source video is portrait (height > width) we rotate the frame
            // so the model always receives a landscape-oriented image. The overlay
            // canvas must match the rotated input dimensions.
            const vw = video.videoWidth || 1280;
            const vh = video.videoHeight || 720;
            if (vw && vh && vw < vh) {
                // portrait -> rotate 90deg: input canvas becomes (height x width)
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
            if (animationFrameId !== null) {
                cancelAnimationFrame(animationFrameId);
                animationFrameId = null;
            }
        }

function drawResults(result) {
    canvasContext.save();
    canvasContext.clearRect(0, 0, canvas.width, canvas.height);
    canvasContext.drawImage(video, 0, 0, canvas.width, canvas.height);

    const landmarks = result.landmarks?.[0];
    if (landmarks?.length) {
        // 1. Draw the standard skeleton
        drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
            color: '#67f2c4',
            lineWidth: 2
        });
        drawingUtils.drawLandmarks(landmarks, {
            color: '#f7fbff',
            radius: 2
        });

        // 2. Helper to get midpoint of two landmarks
        const mid = (idx1, idx2) => ({
            x: (landmarks[idx1].x + landmarks[idx2].x) / 2,
            y: (landmarks[idx1].y + landmarks[idx2].y) / 2
        });

        // 3. Define Body Segments and Weights (Scientific Anthropometric Data)
        // We use the midpoint of joints to represent the "center" of that limb's mass
        const segments = [
            { pos: landmarks[0], weight: 0.08 },                       // Head (Nose)
            { pos: mid(11, 24), weight: 0.50 },                        // Torso (Shoulder to Hip center)
            { pos: mid(23, 25), weight: 0.10 }, { pos: mid(24, 26), weight: 0.10 }, // Thighs
            { pos: mid(25, 27), weight: 0.06 }, { pos: mid(26, 28), weight: 0.06 }, // Lower Legs
            { pos: mid(11, 13), weight: 0.03 }, { pos: mid(12, 14), weight: 0.03 }, // Upper Arms
            { pos: mid(13, 15), weight: 0.02 }, { pos: mid(14, 16), weight: 0.02 }  // Forearms
        ];

        // 4. Calculate Weighted Center of Gravity
        let cogX = 0;
        let cogY = 0;
        
        segments.forEach(s => {
            cogX += s.pos.x * s.weight;
            cogY += s.pos.y * s.weight;
        });

        // 5. Draw the COG Indicator
        canvasContext.beginPath();
        // Drawing a "crosshair" or target style for better visibility
        canvasContext.arc(cogX * canvas.width, cogY * canvas.height, 12, 0, Math.PI * 2);
        canvasContext.strokeStyle = '#ffd166';
        canvasContext.lineWidth = 3;
        canvasContext.stroke();
        
        // Inner dot
        canvasContext.beginPath();
        canvasContext.arc(cogX * canvas.width, cogY * canvas.height, 4, 0, Math.PI * 2);
        canvasContext.fillStyle = '#ffd166';
        canvasContext.fill();
    }

    canvasContext.restore();
}

        function trackFrame() {
            if (!poseLandmarker || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
                animationFrameId = requestAnimationFrame(trackFrame);
                return;
            }

            // Draw the video frame into the input canvas with rotation if needed.
            const vw = video.videoWidth || 0;
            const vh = video.videoHeight || 0;
            if (vw && vh && vw < vh) {
                // portrait: rotate -90deg so the resulting image is upright for the model
                inputCtx.save();
                inputCtx.clearRect(0, 0, inputCanvas.width, inputCanvas.height);
                inputCtx.translate(inputCanvas.width / 2, inputCanvas.height / 2);
                inputCtx.rotate(-Math.PI / 2);
                // draw the video centered (video width/height are swapped visually)
                inputCtx.drawImage(video, -vw / 2, -vh / 2, vw, vh);
                inputCtx.restore();
            } else {
                inputCtx.clearRect(0, 0, inputCanvas.width, inputCanvas.height);
                inputCtx.drawImage(video, 0, 0, inputCanvas.width, inputCanvas.height);
            }

            if (video.currentTime !== lastVideoTime) {
                lastVideoTime = video.currentTime;
                // Run detection on the normalized input canvas so landmarks match what we draw
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
                setStatus('Webcam tracking is active.');
            } catch (error) {
                console.error(error);
                setStatus('Could not start the webcam. Use a video file or run on localhost/https.');
            }
        });

        videoFileInput.addEventListener('change', async () => {
            const file = videoFileInput.files?.[0];
            if (!file) {
                return;
            }

            stopActiveStream();
            resetLoop();

            const objectUrl = URL.createObjectURL(file);
            video.srcObject = null;
            video.src = objectUrl;
            video.onloadedmetadata = async () => {
                resizeCanvas();
                await video.play();
                await startTracking();
                setStatus(`Tracking ${file.name}.`);
            };
        });

        await loadLandmarker();
        video.addEventListener('loadedmetadata', resizeCanvas);
