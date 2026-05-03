https://drive.google.com/drive/folders/1cU3PPqO5yyzSqmKspAggIqzKcUFcy37z?usp=sharing

Extract the Skeleton: MediaPipe gives you 33 joint coordinates (X, Y) per video frame.
Calculate Center of Mass (COM): You don't need perfect biomechanics. You can approximate the climber's COM by taking the average position of the two hip nodes (MediaPipe landmarks 23 and 24) and the two shoulder nodes (landmarks 11 and 12).
Track the Path: Store the COM coordinate for every frame. Draw a line connecting them on an HTML5 Canvas layered over the video.
Calculate "Smoothness" (The Wow Factor): Calculate the distance the COM moves between each frame. If the distance fluctuates wildly (stop-and-go movement), the score drops. If the velocity is relatively constant, the score is high (indicating "Flow").

The Tech Stack (100% Browser-Based)
Frontend Framework: React or Next.js (Tailwind for quick styling).
AI/Vision: @mediapipe/pose (Google's official NPM package) or TensorFlow.js.
Graphics: Standard HTML5 <canvas> positioned exactly on top of an HTML5 <video> element.
Charts: Recharts or Chart.js to show the velocity/smoothness graph next to the video.

The 24-Hour Execution Timeline
Hours 1-4: The Skeleton (UI & Setup)
Initialize the React app.
Create a clean, dark-mode UI with an upload button for video files.
Get the video playing in the browser with an invisible <canvas> perfectly overlaid on top of it.
Hours 4-10: The Brains (MediaPipe Integration)
Import MediaPipe. This is the hardest technical hurdle.
Feed the video frames into MediaPipe's onResults callback.
Draw simple circles on the canvas at the exact X/Y coordinates MediaPipe gives you for the hands and feet. Ensure they scale correctly with the video dimensions.
Hours 10-16: The Logic (COM & The Trail)
Write the function to calculate the Center of Mass from the hip and shoulder coordinates.
Save the COM into an array for every frame.
Write the canvas drawing logic to stroke a line connecting all the points in your COM array, creating the "glowing trail."
Hours 16-20: The Polish (Metrics & Graphs)
Write the math to calculate the "Smoothness Score" (velocity variance).
Add a real-time line chart next to the video that spikes when the climber makes a jerky, inefficient move.
Color-code the COM trail: Green for smooth movement, Red for jerky stops.
Hours 20-24: The Pitch & Demo Prep
Crucial: Do not rely on live demos failing! Pre-record a video of the app perfectly analyzing a climbing clip to show the judges, just in case the Wi-Fi drops.
Practice the pitch: "We built an accessible, AI-driven biomechanics coach that runs entirely in your browser..."
