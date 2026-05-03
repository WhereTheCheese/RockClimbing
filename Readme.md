# ClimberAid

**AI-Powered Rock Climbing Performance Analysis**

ClimberAid is a browser-based biomechanics analysis tool that uses computer vision to help climbers improve their technique. By tracking your center of mass and body positioning in real-time, it provides instant feedback on balance, stability, and movement efficiency.

[![Live Demo](https://img.shields.io/badge/Demo-Live-success)](https://wherethecheese.github.io/RockClimbing/)

---

## Climbing Videos

Access our collection of demo videos and sample climbing footage:
**[Google Drive - Demo Videos](https://drive.google.com/drive/folders/1cU3PPqO5yyzSqmKspAggIqzKcUFcy37z?usp=sharing_eil_se_dm&ts=69f6d5a0)**

---

## Features

### Real-Time Analysis
- **Pose Detection**: 33-point skeletal tracking using Google MediaPipe
- **Center of Mass (CoM) Tracking**: Anthropometric calculations for accurate body positioning
- **Optimal Balance Line**: Visual guide showing the ideal CoM position based on hand and foot placement
- **Smoothness Score**: Real-time feedback on movement efficiency and flow

### Performance Metrics
- **Average Stability**: Session-wide balance consistency percentage
- **Frame Stability**: Instant accuracy of current body position
- **Movement Velocity**: Speed analysis with variance detection
- **Velocity Graph**: Visual timeline of movement patterns

### Comparison Mode
- Side-by-side video analysis
- Synchronized playback controls
- Comparative metrics dashboard
- Frame-by-frame stepping

### User-Friendly Controls
- Upload video files or use live webcam
- Adjustable playback speed (0.25x - 2x)
- Frame-by-frame navigation
- Multiple pose detection models (Lite/Full/Heavy)
- GPU/CPU processing options

---

## Getting Started

### Quick Start
1. Open `index.html` in a modern web browser
2. Upload a climbing video or start your webcam
3. Watch as the AI analyzes your technique in real-time

### No Installation Required
ClimberAid runs entirely in your browser - no backend, no downloads, no setup!

### Supported Browsers
- Chrome/Edge (Recommended)
- Firefox
- Safari

---

## Technology Stack

### Frontend
- **Pure HTML5/CSS3/JavaScript** - No framework dependencies
- **Vanilla JS Modules** - Modern ES6+ architecture

### AI/Computer Vision
- **MediaPipe Pose Landmarker** - Google's state-of-the-art pose detection
- **WebAssembly (WASM)** - High-performance ML inference in the browser

### Graphics & Visualization
- **HTML5 Canvas** - Real-time overlay rendering
- **Custom Chart Engine** - Lightweight velocity graphing

### Performance Optimizations
- Frame skipping for 60fps rendering
- Downscaled detection canvas (640px max)
- GPU acceleration support
- Cached detection results

---

## How It Works

### 1. Pose Detection
MediaPipe extracts 33 body landmarks (joints) from each video frame with X, Y, Z coordinates.

### 2. Center of Mass Calculation
We calculate CoM using anthropometric segment weights:
- Head: 8%
- Trunk: 50%
- Thighs: 20% (10% each)
- Lower legs: 12% (6% each)
- Upper arms: 6% (3% each)
- Forearms: 4% (2% each)

### 3. Optimal Balance Line
The "Axis of Tension" connects your hands (pull center) to your feet (base of support). The optimal CoM position lies along this line at your current height.

### 4. Stability Scoring
- **Horizontal Gap**: Distance between actual and optimal CoM
- **Threshold**: 5% of frame width for "aligned" status
- **Session Average**: Percentage of aligned frames

### 5. Smoothness Analysis
- Tracks velocity variance over a 30-frame rolling window
- Lower variance = smoother, more efficient movement
- Scores from 0-100 (higher is better)

---

## Project Structure

```
RockClimbing/
├── index.html              # Main single-view page
├── compare.html            # Side-by-side comparison page
├── mediaPipeOverlay.js     # Core pose detection & rendering
├── compareOverlay.js       # Dual-video comparison logic
├── dataAnalysis.js         # Metrics calculation & charting
├── alignmentTracker.js     # Balance analysis algorithms
├── ClimberAid_Logo.png     # Application logo
└── Readme.md               # This file
```

---

## Use Cases

- **Training**: Identify inefficient movement patterns
- **Coaching**: Provide objective feedback to students
- **Self-Analysis**: Review your beta and technique
- **Competition Prep**: Compare different approaches to the same route
- **Progress Tracking**: Measure improvement over time

---

## Technical Details

### MediaPipe Models
- **Lite**: Fast, lower accuracy (~30ms/frame)
- **Full**: Balanced performance (default, ~50ms/frame)
- **Heavy**: Maximum accuracy (~80ms/frame)

### Performance Targets
- 30+ FPS rendering on modern hardware
- Detection every 2nd frame (staggered in comparison mode)
- UI updates throttled to 30Hz for smooth performance

### Browser Requirements
- WebGL support for GPU acceleration
- ES6 module support
- Canvas 2D rendering context
- MediaDevices API (for webcam)

---

## Team

- **Nathaniel Wu** - [LinkedIn](https://www.linkedin.com/in/nathaniel-wu-3755a22b2/)
- **Andrew Washburn** - [LinkedIn](https://www.linkedin.com/in/andrew-washburn-4855a32b2/)
- **Jacob Brayko** - [LinkedIn](https://www.linkedin.com/in/jacob-brayko-520ab53aa/)
- **Zach Yarvis** - [LinkedIn](https://www.linkedin.com/in/zach-yarvis-81327b2b2/)

---

## License

This project was created as part of a hackathon/educational project. Feel free to use and modify for learning purposes.

---

## Acknowledgments

- **Google MediaPipe** - For the incredible pose detection technology
- **Rock climbing community** - For inspiration and feedback
- **Open source contributors** - For making browser-based ML possible

---

## Future Improvements

- 3D pose analysis with depth estimation
- Mobile app with offline processing
- Historical progress tracking database
- Route difficulty prediction ML model
- Social features and community leaderboards
- Integration with climbing gym management systems
- AR overlay mode for live coaching
- Wearable device integration

---

**Built with love for climbers, by climbers**
