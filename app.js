import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

let ENABLE_LOGGING = true;

function log(...args) {
  if (ENABLE_LOGGING) {
    console.log(...args);
  }
}

const cameraCanvas = document.getElementById("cameraCanvas");
const cameraCtx = cameraCanvas.getContext("2d");
const cameraVideo = document.getElementById("cameraVideo");
const gestureLabel = document.getElementById("gestureLabel");
const handStatus = document.getElementById("handStatus");
const systemStatus = document.getElementById("systemStatus");

const gameCanvas = document.getElementById("gameCanvas");
const gameCtx = gameCanvas.getContext("2d");
const scoreValue = document.getElementById("scoreValue");
const linesValue = document.getElementById("linesValue");
const levelValue = document.getElementById("levelValue");
const gameState = document.getElementById("gameState");
const startButton = document.getElementById("startButton");
const pauseButton = document.getElementById("pauseButton");

let handLandmarker;
let cameraReady = false;
let appRunning = false;
let cameraStream;
let rafCameraId;
let rafGameId;
let lastVideoTime = -1;
let learnedGestureProfiles = null;
let lastRecognizedGesture = "None";
let lastMotionDirection = null;
let motionDisplayTime = 0;
let lastScreenHandX = null;
let horizontalMotionCarry = 0;

const gestureHistory = [];
const gestureCooldown = {
  Index: 0,
  TwoFinger: 0,
  Left: 0,
  Right: 0,
  Down: 0,
};

const HORIZONTAL_MOVE_SCALE = 12;
const HORIZONTAL_MOVE_DEADZONE = 0.01;

const CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [0, 17],
];

const COLS = 10;
const ROWS = 20;
const BLOCK_SIZE = 30;
const DROP_BASE = 780;

const COLORS = {
  0: "#00000000",
  1: "#69e0ff",
  2: "#6ca3ff",
  3: "#7fffca",
  4: "#ffbb6d",
  5: "#ff77af",
  6: "#ffe16d",
  7: "#a88cff",
};

const PIECES = [
  [[1, 1, 1, 1]],
  [
    [2, 0, 0],
    [2, 2, 2],
  ],
  [
    [0, 0, 3],
    [3, 3, 3],
  ],
  [
    [4, 4],
    [4, 4],
  ],
  [
    [0, 5, 5],
    [5, 5, 0],
  ],
  [
    [0, 6, 0],
    [6, 6, 6],
  ],
  [
    [7, 7, 0],
    [0, 7, 7],
  ],
];

const game = {
  board: createBoard(),
  current: null,
  score: 0,
  lines: 0,
  level: 1,
  dropCounter: 0,
  dropInterval: DROP_BASE,
  lastTime: 0,
  active: false,
  paused: false,
};

drawCameraIdle();
drawGame();

startButton.addEventListener("click", async () => {
  if (!appRunning) {
    await startSystem();
    return;

  }

  if (game.active) {
    resetGame();
  } else {
    startGameLoop();
  }
});

pauseButton.addEventListener("click", () => {
  if (!game.active) {
    return;
  }

  togglePause();
});

async function startSystem() {
  systemStatus.textContent = "Initializing models...";
  log("[System] Starting initialization...");

  try {
    log("[System] Loading gesture profiles...");
    await loadGestureProfiles();
    log("[System] Profiles loaded");

    log("[System] Initializing hand landmark detector...");
    await initHandLandmarker();
    log("[System] Hand detector ready");

    log("[System] Initializing camera...");
    await initCamera();
    log("[System] Camera ready");

    startGameLoop();

    appRunning = true;
    startButton.textContent = "Restart Game";
    systemStatus.textContent = learnedGestureProfiles
      ? "Live (learned static gestures)"
      : "Live (fallback static gestures)";
    gameState.textContent = "Running";

    log("[System] Starting camera frame processing...");
    rafCameraId = requestAnimationFrame(processCameraFrame);
  } catch (error) {
    systemStatus.textContent = "Initialization failed";
    log("[System] Error:", error);
    alert(
      "Could not start camera or hand tracking. Check camera permission and internet access for MediaPipe files. See console for details.",
    );
  }
}

async function loadGestureProfiles() {
  try {
    log("[Profiles] Fetching gesture_profiles.json...");
    const response = await fetch("gesture_profiles.json", { cache: "no-store" });
    if (!response.ok) {
      log("[Profiles] File not found, using fallback rules");
      learnedGestureProfiles = null;
      return;
    }

    const payload = await response.json();
    if (!payload?.trained || !payload?.profiles || !payload?.featureLength) {
      log("[Profiles] File exists but not trained, using fallback rules");
      learnedGestureProfiles = null;
      return;
    }

    log(`[Profiles] Loaded trained profiles for: ${Object.keys(payload.profiles).join(", ")}`);
    learnedGestureProfiles = payload;
  } catch (error) {
    log("[Profiles] Error loading:", error.message);
    learnedGestureProfiles = null;
  }
}

async function initHandLandmarker() {
  if (handLandmarker) {
    return;
  }

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm",
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.55,
    minTrackingConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
  });
}

async function initCamera() {
  if (cameraReady) {
    return;
  }

  log("[Camera] Requesting user media...");
  cameraStream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false,
  });

  log("[Camera] Stream obtained, attaching to video element...");
  cameraVideo.srcObject = cameraStream;

  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("Video loadeddata timeout")), 5000);
    cameraVideo.onloadeddata = () => {
      clearTimeout(timeout);
      resolve();
    };
    cameraVideo.play().catch((e) => {
      clearTimeout(timeout);
      reject(e);
    });
  });

  log("[Camera] Video playing and ready");
  cameraReady = true;
}

function processCameraFrame() {
  if (!cameraReady || !handLandmarker || !appRunning) {
    return;
  }

  const now = performance.now();

  if (cameraVideo.currentTime !== lastVideoTime) {
    lastVideoTime = cameraVideo.currentTime;
    try {
      const result = handLandmarker.detectForVideo(cameraVideo, now);
      if (result.landmarks.length) {
        updateHorizontalMovement(result.landmarks[0], now);
      } else {
        resetHorizontalMovement();
      }
      const recognized = handleGestures(result, now);
      renderCamera(result, recognized);
    } catch (error) {
      log("[Camera] Detection error:", error);
    }
  }

  rafCameraId = requestAnimationFrame(processCameraFrame);
}

function renderCamera(result, recognized = "None") {
  cameraCtx.fillStyle = "#000";
  cameraCtx.fillRect(0, 0, cameraCanvas.width, cameraCanvas.height);

  if (!result.landmarks.length) {
    handStatus.textContent = "No hand detected";
    drawCanvasOverlay();
    return;
  }

  handStatus.textContent = "Hand detected";
  drawPalmWireframe(result.landmarks[0]);
  drawCanvasOverlay();
}

function drawPalmWireframe(landmarks) {
  cameraCtx.strokeStyle = "rgba(76, 220, 255, 0.95)";
  cameraCtx.lineWidth = 2.5;
  cameraCtx.lineCap = "round";
  cameraCtx.lineJoin = "round";

  for (const [a, b] of CONNECTIONS) {
    const p1 = mapPoint(landmarks[a], true);
    const p2 = mapPoint(landmarks[b], true);
    cameraCtx.beginPath();
    cameraCtx.moveTo(p1.x, p1.y);
    cameraCtx.lineTo(p2.x, p2.y);
    cameraCtx.stroke();
  }

  cameraCtx.fillStyle = "rgba(180, 244, 255, 0.9)";
  for (const point of landmarks) {
    const p = mapPoint(point, true);
    cameraCtx.beginPath();
    cameraCtx.arc(p.x, p.y, 3, 0, Math.PI * 2);
    cameraCtx.fill();
  }
}

function drawCanvasOverlay() {
  const padding = 20;
  const topLine = padding;
  const bottomLine = cameraCanvas.height - padding - 40;

  cameraCtx.font = "bold 24px Outfit";
  cameraCtx.textAlign = "center";
  cameraCtx.fillStyle = "#ffffff";
  cameraCtx.strokeStyle = "rgba(0, 0, 0, 0.6)";
  cameraCtx.lineWidth = 3;

  const gestureText = lastRecognizedGesture;
  cameraCtx.strokeText(gestureText, cameraCanvas.width / 2, topLine + 30);
  cameraCtx.fillText(gestureText, cameraCanvas.width / 2, topLine + 30);

  if (lastMotionDirection && performance.now() - motionDisplayTime < 500) {
    drawMotionIndicator(lastMotionDirection);
  }
}

function drawMotionIndicator(direction) {
  const centerX = cameraCanvas.width / 2;
  const centerY = cameraCanvas.height / 2;
  const arrowLength = 80;
  const arrowHeadSize = 15;

  cameraCtx.strokeStyle = "rgba(68, 200, 255, 0.9)";
  cameraCtx.fillStyle = "rgba(68, 200, 255, 0.7)";
  cameraCtx.lineWidth = 4;
  cameraCtx.lineCap = "round";
  cameraCtx.lineJoin = "round";

  let dx = 0,
    dy = 0;
  if (direction === "Left") {
    dx = -arrowLength;
  } else if (direction === "Right") {
    dx = arrowLength;
  } else if (direction === "Down") {
    dy = arrowLength;
  }

  cameraCtx.beginPath();
  cameraCtx.moveTo(centerX - dx / 2, centerY - dy / 2);
  cameraCtx.lineTo(centerX + dx / 2, centerY + dy / 2);
  cameraCtx.stroke();

  const angle = Math.atan2(dy, dx);
  cameraCtx.save();
  cameraCtx.translate(centerX + dx / 2, centerY + dy / 2);
  cameraCtx.rotate(angle);
  cameraCtx.beginPath();
  cameraCtx.moveTo(0, 0);
  cameraCtx.lineTo(-arrowHeadSize, -arrowHeadSize / 2);
  cameraCtx.lineTo(-arrowHeadSize, arrowHeadSize / 2);
  cameraCtx.closePath();
  cameraCtx.fill();
  cameraCtx.restore();

  cameraCtx.font = "bold 18px Outfit";
  cameraCtx.fillStyle = "#ffffff";
  cameraCtx.strokeStyle = "rgba(0, 0, 0, 0.6)";
  cameraCtx.lineWidth = 2;
  cameraCtx.textAlign = "center";
  const labelOffset = arrowLength * 0.7;
  const labelX = centerX + (dx / arrowLength) * labelOffset;
  const labelY = centerY + (dy / arrowLength) * labelOffset;
  cameraCtx.strokeText(direction, labelX, labelY + 40);
  cameraCtx.fillText(direction, labelX, labelY + 40);
}

function mapPoint(point, mirrored = false) {
  return {
    x: (mirrored ? 1 - point.x : point.x) * cameraCanvas.width,
    y: point.y * cameraCanvas.height,
  };
}

function handleGestures(result, now) {
  if (!result.landmarks.length) {
    gestureHistory.length = 0;
    gestureLabel.textContent = "None";
    lastRecognizedGesture = "None";
    resetHorizontalMovement();
    return "None";
  }

  const landmarks = result.landmarks[0];
  const handedness = result.handednesses?.[0]?.[0]?.categoryName ?? "Right";

  const motionGesture = detectMotionGesture(landmarks, now);
  let recognized = motionGesture || detectStaticGesture(landmarks, handedness);

  if (!recognized) {
    recognized = "Tracking";
  }

  log(`[Gesture] Motion: ${motionGesture || "none"}, Recognized: ${recognized}, Hand: ${handedness}`);
  lastRecognizedGesture = recognized;
  if (motionGesture) {
    lastMotionDirection = motionGesture;
    motionDisplayTime = performance.now();
  }
  gestureLabel.textContent = recognized;
  triggerGestureAction(recognized, now, handedness);
  return recognized;
}

function detectMotionGesture(landmarks, now) {
  const wrist = landmarks[0];
  gestureHistory.push({ t: now, x: wrist.x, y: wrist.y });

  while (gestureHistory.length && now - gestureHistory[0].t > 360) {
    gestureHistory.shift();
  }

  if (gestureHistory.length < 4) {
    return null;
  }

  const first = gestureHistory[0];
  const last = gestureHistory[gestureHistory.length - 1];
  const dx = last.x - first.x;
  const dy = last.y - first.y;

  if (Math.abs(dx) > 0.1 && Math.abs(dx) > Math.abs(dy) * 1.4) {
    if (dx < -0.1) {
      log(`[Motion] Left detected: dx=${dx.toFixed(3)}, historyLen=${gestureHistory.length}`);
      return "Right";
    }

    if (dx > 0.1) {
      log(`[Motion] Right detected: dx=${dx.toFixed(3)}, historyLen=${gestureHistory.length}`);
      return "Left";
    }
  }

  if (dy > 0.12 && Math.abs(dy) > Math.abs(dx) * 1.35) {
    log(`[Motion] Down detected: dy=${dy.toFixed(3)}, historyLen=${gestureHistory.length}`);
    return "Down";
  }

  return null;
}

function detectStaticGesture(landmarks, handedness) {
  const learned = classifyWithLearnedProfiles(landmarks);
  if (learned) {
    return learned;
  }

  const index = isFingerExtended(landmarks, 8, 6);
  const middle = isFingerExtended(landmarks, 12, 10);
  const ring = isFingerExtended(landmarks, 16, 14);
  const pinky = isFingerExtended(landmarks, 20, 18);
  const thumb = isThumbExtended(landmarks, handedness);

  const extendedCount = [thumb, index, middle, ring, pinky].filter(Boolean).length;

  if (index && !middle && !ring && !pinky) {
    return "Index";
  }

  if (index && middle && !ring && !pinky) {
    return "TwoFinger";
  }

  if (extendedCount <= 1 && !index && !middle && !ring && !pinky) {
    return "Fist";
  }

  if (index && middle && ring && pinky && thumb) {
    return "Palm";
  }

  return null;
}

function classifyWithLearnedProfiles(landmarks) {
  if (!learnedGestureProfiles?.profiles) {
    return null;
  }

  const feature = extractNormalizedFeature(landmarks);
  if (!feature || feature.length !== learnedGestureProfiles.featureLength) {
    return null;
  }

  const mirrored = mirrorFeature(feature);
  let bestLabel = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  let bestThreshold = Number.POSITIVE_INFINITY;

  for (const [label, profile] of Object.entries(learnedGestureProfiles.profiles)) {
    if (!Array.isArray(profile.centroid) || typeof profile.threshold !== "number") {
      continue;
    }

    if (profile.centroid.length !== feature.length) {
      continue;
    }

    const d1 = euclideanDistance(feature, profile.centroid);
    const d2 = euclideanDistance(mirrored, profile.centroid);
    const distance = Math.min(d1, d2);

    if (distance < bestDistance) {
      bestDistance = distance;
      bestLabel = label;
      bestThreshold = profile.threshold;
    }
  }

  if (!bestLabel) {
    return null;
  }

  const adaptiveThreshold = bestThreshold * 1.5;
  const isMatch = bestDistance <= adaptiveThreshold;
  log(`${bestLabel}: dist=${bestDistance.toFixed(3)}, threshold=${bestThreshold.toFixed(3)}, adaptive=${adaptiveThreshold.toFixed(3)}, match=${isMatch}`);

  return isMatch ? bestLabel : null;
}

function extractNormalizedFeature(landmarks) {
  if (!landmarks || landmarks.length !== 21) {
    return null;
  }

  const wrist = landmarks[0];
  const points = landmarks.map((point) => [point.x - wrist.x, point.y - wrist.y]);
  let maxNorm = 0;

  for (const [dx, dy] of points) {
    const norm = Math.hypot(dx, dy);
    if (norm > maxNorm) {
      maxNorm = norm;
    }
  }

  if (maxNorm < 1e-6) {
    return null;
  }

  const feature = [];
  for (const [dx, dy] of points) {
    feature.push(dx / maxNorm, dy / maxNorm);
  }

  return feature;
}

function mirrorFeature(feature) {
  const mirrored = feature.slice();
  for (let i = 0; i < mirrored.length; i += 2) {
    mirrored[i] *= -1;
  }

  return mirrored;
}

function euclideanDistance(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    total += d * d;
  }

  return Math.sqrt(total);
}

function isFingerExtended(landmarks, tipId, pipId) {
  return landmarks[tipId].y < landmarks[pipId].y - 0.025;
}

function isThumbExtended(landmarks, handedness) {
  const tip = landmarks[4];
  const mcp = landmarks[2];
  if (handedness === "Left") {
    return tip.x < mcp.x - 0.02;
  }

  return tip.x > mcp.x + 0.02;
}

function triggerGestureAction(gesture, now, handedness = "Right") {
  if (!game.active) {
    return;
  }

  if (gesture === "Tracking") {
    return;
  }

  const cooldowns = {
    Palm: 850,
    Index: 450,
    TwoFinger: 300,
    Left: 180,
    Right: 180,
    Down: 120,
  };

  if (now - gestureCooldown[gesture] < cooldowns[gesture]) {
    return;
  }

  // All other gestures require right hand
  if (game.paused) {
    log(`[Control] ${gesture} blocked - game is paused`);
    return;
  }

  if (handedness !== "Right") {
    log(`[Control] ${gesture} ignored - detected on ${handedness} hand, requires Right hand`);
    return;
  }

  gestureCooldown[gesture] = now;

  switch (gesture) {
    case "Index":
      log(`[Control] Index (Right hand) - Hard drop`);
      hardDrop();
      break;
    case "TwoFinger":
      log(`[Control] TwoFinger (Right hand) - Rotate`);
      rotatePiece(1);
      break;
    case "Left":
      log(`[Control] Left motion detected - horizontal tracking handles movement`);
      break;
    case "Right":
      log(`[Control] Right motion detected - horizontal tracking handles movement`);
      break;
    case "Down":
      log(`[Control] Down flick (Right hand) - Soft drop`);
      softDrop();
      break;
    default:
      break;
  }
}

function createBoard() {
  return Array.from({ length: ROWS }, () => Array(COLS).fill(0));
}

function updateHorizontalMovement(landmarks, now) {
  if (!game.active || game.paused) {
    resetHorizontalMovement();
    return;
  }

  const wrist = landmarks[0];
  const screenX = 1 - wrist.x;

  if (lastScreenHandX === null) {
    lastScreenHandX = screenX;
    horizontalMotionCarry = 0;
    return;
  }

  const dx = screenX - lastScreenHandX;
  lastScreenHandX = screenX;

  if (Math.abs(dx) < HORIZONTAL_MOVE_DEADZONE) {
    return;
  }

  horizontalMotionCarry += dx * HORIZONTAL_MOVE_SCALE;
  const steps = Math.trunc(horizontalMotionCarry);

  if (steps === 0) {
    return;
  }

  movePieceSteps(steps);
  horizontalMotionCarry -= steps;
}

function resetHorizontalMovement() {
  lastScreenHandX = null;
  horizontalMotionCarry = 0;
}

function resetGame() {
  game.board = createBoard();
  game.score = 0;
  game.lines = 0;
  game.level = 1;
  game.dropCounter = 0;
  game.dropInterval = DROP_BASE;
  game.paused = false;
  game.current = null;
  resetHorizontalMovement();
  pauseButton.textContent = "Pause Game";
  pauseButton.disabled = false;
  updateHud();
  startGameLoop();
}

function startGameLoop() {
  game.active = true;
  game.paused = false;
  game.lastTime = 0;
  resetHorizontalMovement();
  pauseButton.textContent = "Pause Game";
  pauseButton.disabled = false;
  spawnPiece();
  gameState.textContent = "Running";

  if (rafGameId) {
    cancelAnimationFrame(rafGameId);
  }

  rafGameId = requestAnimationFrame(updateGame);
}

function spawnPiece() {
  const shape = PIECES[(Math.random() * PIECES.length) | 0].map((row) => [...row]);
  game.current = {
    matrix: shape,
    pos: {
      x: ((COLS / 2) | 0) - ((shape[0].length / 2) | 0),
      y: 0,
    },
  };

  if (collides(game.board, game.current)) {
    endGame();
  }
}

function updateGame(time = 0) {
  const delta = time - game.lastTime;
  game.lastTime = time;

  if (game.active && !game.paused) {
    game.dropCounter += delta;
    if (game.dropCounter > game.dropInterval) {
      softDrop();
    }
  }

  drawGame();
  rafGameId = requestAnimationFrame(updateGame);
}

function movePiece(dir) {
  game.current.pos.x += dir;
  if (collides(game.board, game.current)) {
    game.current.pos.x -= dir;
    return false;
  }

  return true;
}

function movePieceSteps(steps) {
  const direction = Math.sign(steps);
  let remaining = Math.abs(steps);

  while (remaining > 0) {
    if (!movePiece(direction)) {
      break;
    }

    remaining -= 1;
  }
}

function rotatePiece(dir) {
  const pos = game.current.pos.x;
  let offset = 1;

  rotateMatrix(game.current.matrix, dir);
  while (collides(game.board, game.current)) {
    game.current.pos.x += offset;
    offset = -(offset + (offset > 0 ? 1 : -1));

    if (Math.abs(offset) > game.current.matrix[0].length) {
      rotateMatrix(game.current.matrix, -dir);
      game.current.pos.x = pos;
      return;
    }
  }
}

function softDrop() {
  game.current.pos.y += 1;

  if (collides(game.board, game.current)) {
    game.current.pos.y -= 1;
    merge(game.board, game.current);
    sweepLines();
    spawnPiece();
  }

  game.dropCounter = 0;
}

function hardDrop() {
  while (!collides(game.board, game.current)) {
    game.current.pos.y += 1;
  }

  game.current.pos.y -= 1;
  merge(game.board, game.current);
  sweepLines();
  spawnPiece();
  game.dropCounter = 0;
}

function togglePause() {
  game.paused = !game.paused;
  gameState.textContent = game.paused ? "Paused" : "Running";
  pauseButton.textContent = game.paused ? "Resume Game" : "Pause Game";
  resetHorizontalMovement();
}

function endGame() {
  game.active = false;
  gameState.textContent = "Game Over";
  pauseButton.textContent = "Pause Game";
  pauseButton.disabled = true;
  resetHorizontalMovement();
}

function sweepLines() {
  let rowCount = 1;

  outer: for (let y = game.board.length - 1; y >= 0; y -= 1) {
    for (let x = 0; x < COLS; x += 1) {
      if (game.board[y][x] === 0) {
        continue outer;
      }
    }

    const row = game.board.splice(y, 1)[0].fill(0);
    game.board.unshift(row);
    y += 1;

    game.score += rowCount * 100;
    game.lines += 1;
    rowCount *= 2;
  }

  game.level = Math.floor(game.lines / 10) + 1;
  game.dropInterval = Math.max(140, DROP_BASE - (game.level - 1) * 70);
  updateHud();
}

function updateHud() {
  scoreValue.textContent = String(game.score);
  linesValue.textContent = String(game.lines);
  levelValue.textContent = String(game.level);
}

function collides(board, player) {
  const matrix = player.matrix;
  const pos = player.pos;

  for (let y = 0; y < matrix.length; y += 1) {
    for (let x = 0; x < matrix[y].length; x += 1) {
      if (
        matrix[y][x] !== 0 &&
        (board[y + pos.y] && board[y + pos.y][x + pos.x]) !== 0
      ) {
        return true;
      }
    }
  }

  return false;
}

function merge(board, player) {
  player.matrix.forEach((row, y) => {
    row.forEach((value, x) => {
      if (value !== 0) {
        board[y + player.pos.y][x + player.pos.x] = value;
      }
    });
  });
}

function rotateMatrix(matrix, dir) {
  const height = matrix.length;
  const width = matrix[0].length;
  let rotated;

  if (dir > 0) {
    // Clockwise: new[y][x] = old[height - 1 - x][y]
    rotated = Array.from({ length: width }, (_, y) =>
      Array.from({ length: height }, (_, x) => matrix[height - 1 - x][y]),
    );
  } else {
    // Counter-clockwise: new[y][x] = old[x][width - 1 - y]
    rotated = Array.from({ length: width }, (_, y) =>
      Array.from({ length: height }, (_, x) => matrix[x][width - 1 - y]),
    );
  }

  matrix.length = 0;
  rotated.forEach((row) => matrix.push(row));
}

function drawGame() {
  gameCtx.clearRect(0, 0, gameCanvas.width, gameCanvas.height);

  const gradient = gameCtx.createLinearGradient(0, 0, 0, gameCanvas.height);
  gradient.addColorStop(0, "#1b2f46");
  gradient.addColorStop(1, "#050a12");
  gameCtx.fillStyle = gradient;
  gameCtx.fillRect(0, 0, gameCanvas.width, gameCanvas.height);

  drawMatrix(game.board, { x: 0, y: 0 }, 0.55);

  if (game.current && game.active) {
    drawMatrix(game.current.matrix, game.current.pos, 1);
  }

  drawGrid();

  if (!game.active && appRunning) {
    drawOverlayText("GAME OVER", "Use Start button to restart");
  } else if (game.paused) {
    drawOverlayText("PAUSED", "Use Pause button to resume");
  }
}

function drawMatrix(matrix, offset, alpha = 1) {
  matrix.forEach((row, y) => {
    row.forEach((value, x) => {
      if (value !== 0) {
        drawBlock((x + offset.x) * BLOCK_SIZE, (y + offset.y) * BLOCK_SIZE, value, alpha);
      }
    });
  });
}

function drawBlock(px, py, value, alpha) {
  const size = BLOCK_SIZE;
  const color = COLORS[value];

  gameCtx.globalAlpha = alpha;
  gameCtx.fillStyle = color;
  roundRect(gameCtx, px + 1, py + 1, size - 2, size - 2, 7, true, false);

  const gloss = gameCtx.createLinearGradient(px, py, px, py + size);
  gloss.addColorStop(0, "rgba(255,255,255,0.44)");
  gloss.addColorStop(0.38, "rgba(255,255,255,0.14)");
  gloss.addColorStop(1, "rgba(0,0,0,0.18)");
  gameCtx.fillStyle = gloss;
  roundRect(gameCtx, px + 2, py + 2, size - 4, size - 4, 6, true, false);

  gameCtx.strokeStyle = "rgba(208, 239, 255, 0.42)";
  gameCtx.lineWidth = 1;
  roundRect(gameCtx, px + 2, py + 2, size - 4, size - 4, 6, false, true);
  gameCtx.globalAlpha = 1;
}

function drawGrid() {
  gameCtx.strokeStyle = "rgba(130, 178, 214, 0.16)";
  gameCtx.lineWidth = 1;

  for (let x = 0; x <= COLS; x += 1) {
    gameCtx.beginPath();
    gameCtx.moveTo(x * BLOCK_SIZE, 0);
    gameCtx.lineTo(x * BLOCK_SIZE, ROWS * BLOCK_SIZE);
    gameCtx.stroke();
  }

  for (let y = 0; y <= ROWS; y += 1) {
    gameCtx.beginPath();
    gameCtx.moveTo(0, y * BLOCK_SIZE);
    gameCtx.lineTo(COLS * BLOCK_SIZE, y * BLOCK_SIZE);
    gameCtx.stroke();
  }
}

function drawOverlayText(title, subtitle) {
  gameCtx.fillStyle = "rgba(0,0,0,0.46)";
  gameCtx.fillRect(0, 0, gameCanvas.width, gameCanvas.height);

  gameCtx.textAlign = "center";
  gameCtx.fillStyle = "#f4fbff";
  gameCtx.font = "700 34px Outfit";
  gameCtx.fillText(title, gameCanvas.width / 2, gameCanvas.height / 2 - 8);

  gameCtx.font = "500 15px Outfit";
  gameCtx.fillStyle = "#c4dcee";
  gameCtx.fillText(subtitle, gameCanvas.width / 2, gameCanvas.height / 2 + 24);
}

function roundRect(ctx, x, y, width, height, radius, fill, stroke) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + width - r, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + r);
  ctx.lineTo(x + width, y + height - r);
  ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  ctx.lineTo(x + r, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();

  if (fill) {
    ctx.fill();
  }

  if (stroke) {
    ctx.stroke();
  }
}

function drawCameraIdle() {
  cameraCtx.fillStyle = "rgba(0, 0, 0, 0.5)";
  cameraCtx.fillRect(0, 0, cameraCanvas.width, cameraCanvas.height);
  cameraCtx.fillStyle = "#ffffff";
  cameraCtx.font = "500 20px Outfit";
  cameraCtx.textAlign = "center";
  cameraCtx.fillText("Press Start System to enable camera", cameraCanvas.width / 2, cameraCanvas.height / 2 - 12);
  cameraCtx.font = "400 14px Outfit";
  cameraCtx.fillStyle = "rgba(255, 255, 255, 0.7)";
  cameraCtx.fillText("Camera feed will appear here", cameraCanvas.width / 2, cameraCanvas.height / 2 + 20);
}
