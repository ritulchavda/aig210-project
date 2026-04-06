# Vision-Controlled Tetris

A browser-based Tetris game controlled by live hand gestures using MediaPipe Hand Landmarker.

## Features

- Real-time hand landmark tracking from webcam
- Black background hand-vector visualization panel
- 6-gesture recognition pipeline:
  - Palm
  - Index
  - TwoFinger
  - Left flick (motion)
  - Right flick (motion)
  - Down flick (motion)
- Fully playable Tetris engine with:
  - piece movement and rotation
  - line clearing
  - score, lines, level, and increasing speed
- Dark glossy UI inspired by modern glass interfaces

## Gesture Mapping & Hand Requirements

- **Palm 🤚**: Neutral State
- **Two Fingers (Index + Middle)**: Rotate piece
- **Index ☝**: Hard drop
- **Left flick 🫲**: Move piece left
- **Right flick 🫱**: Move piece right
- **Down flick 🫳**: Soft drop piece

**Note:** Each gesture will only trigger on the specified hand. Performing a right-hand gesture with your left hand will be ignored, and vice versa.

## Run

Because the app uses ES modules and webcam APIs, run it through a local web server (not directly with file://).

### Option 1: VS Code Live Server

1. Open `index.html`.
2. Use the Live Server extension to start a local server.
3. Allow camera permission in the browser.

### Option 2: Python HTTP server

From the project folder:

```powershell
python -m http.server 8000
```

Then open:

http://localhost:8000

## Train Static Gesture Model (Python)

The website can use a learned static gesture model (`Palm`, `Index`, `TwoFinger`) from `gesture_profiles.json`.

1. Install trainer dependencies:

```powershell
pip install -r requirements-gestures.txt
```

2. Run the trainer:

```powershell
python train_gestures.py --samples-per-gesture 160
```

3. In the trainer window:

- Press `1` and hold Palm until complete.
- Press `2` and hold Index until complete.
- Press `3` and hold Two Fingers (Index + Middle) until complete.
- Press `S` to save.

4. Restart the webpage. The app will detect and use the trained profile file automatically.

If `gesture_profiles.json` is not trained yet, the app falls back to built-in static rules.

## Debug & Logging

The app displays real-time gesture recognition and motion direction on the camera feed.

### Enable/Disable Console Logging

To toggle logging output in the browser console, open DevTools (F12) and run:

```javascript
ENABLE_LOGGING = true;   // Enable logs
ENABLE_LOGGING = false;  // Disable logs
```

When enabled, you'll see logs like:
- `[System]` - Initialization steps
- `[Profiles]` - Gesture profile loading
- `[Camera]` - Camera setup and errors
- `[Gesture]` - Recognized gestures
- `[Motion]` - Motion direction detection (Left/Right/Down)

### On-Screen Display

The camera feed shows:
1. **Top center**: Current gesture name (Palm, TwoFinger, Index, Tracking, etc.)
2. **Center with arrow**: Motion direction indicator (Left/Right/Down) with animated arrow for 500ms after motion is detected

## Notes

- Internet access is required initially to fetch MediaPipe model and runtime files from CDN.
- Good lighting and keeping one hand in frame improves recognition quality.
