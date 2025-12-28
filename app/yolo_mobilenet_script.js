const CONFIG = {
    PERCLOS_WINDOW_SIZE: 300,
    PERCLOS_THRESHOLD: 0.30,
    MICROSLEEP_THRESHOLD: 15,
    SOUND_ENABLED: true,
    MAX_HISTORY: 30,
    YOLO_INPUT_SIZE: 224, 
    MOBILENET_INPUT_SIZE: 224,
    YOLO_GRID_SIZE: 7,
    YOLO_NUM_ANCHORS: 3,
    EYE_CONFIDENCE_THRESHOLD: 0.5,
};

let state = {
    isMonitoring: false,
    startTime: 0,
    eyeHistory: [],
    perclosScore: 0,
    closedStreakFrames: 0,
    alertCount: 0,
    lastAlertTime: 0,
    noFaceDetected: true,
    currentStatus: "Unknown",
    inferenceTime: 0
};

const elements = {
    startBtn: document.getElementById("startBtn"),
    stopBtn: document.getElementById("stopBtn"),
    resetBtn: document.getElementById("resetBtn"),
    videoFeed: document.getElementById("videoFeed"),
    videoPlaceholder: document.getElementById("videoPlaceholder"),
    canvas: document.getElementById("canvas"),
    statusIndicator: document.getElementById("statusIndicator"),
    statusText: document.getElementById("statusText"),
    timer: document.getElementById("timer"),
    timeline: document.getElementById("timeline"),
    alertLog: document.getElementById("alertLog"),
    
    perclosValue: document.getElementById("perclosValue"),
    perclosBar: document.getElementById("perclosBar"),
    closedStreak: document.getElementById("closedStreak"),
    latencyValue: document.getElementById("latencyValue"),
    totalAlerts: document.getElementById("totalAlerts"),
    eyeStateText: document.getElementById("eyeStateText")
};

let yoloSession = null;
let mobilenetSession = null;
let camera = null;
let timerInterval = null;
const ctx = elements.canvas.getContext("2d");

for (let i = 0; i < CONFIG.MAX_HISTORY; i++) {
    const bar = document.createElement("div");
    bar.className = "timeline-bar";
    elements.timeline.appendChild(bar);
}

elements.startBtn.disabled = true;

async function initModels() {
    try {
        updateStatus("warning", "LOADING AI MODELS...");
        
        console.log("Loading YOLO Eye Detector...");
        yoloSession = await ort.InferenceSession.create('./models/yolo_eye_detector.onnx');
        console.log("YOLO Eye Detector Loaded.");

        console.log("Loading MobileNet Classifier...");
        mobilenetSession = await ort.InferenceSession.create('./models/mobilenet_eye_classifier.onnx');
        console.log("MobileNet Classifier Loaded.");
        
        updateStatus("ready", "SYSTEM READY");
        elements.startBtn.disabled = false;
        
    } catch (e) {
        console.error("Model Error:", e);
        alert("System Error: AI Models failed to load. Check console for details.");
    }
}
initModels();

elements.startBtn.addEventListener("click", startSystem);
elements.stopBtn.addEventListener("click", stopSystem);
elements.resetBtn.addEventListener("click", resetStats);

async function startSystem() {
    if (state.isMonitoring) return;
    
    try {
        camera = new Camera(elements.videoFeed, {
            onFrame: async () => {
                if(!state.isMonitoring) return;
                await processFrame();
            },
            width: 1280,
            height: 720
        });

        await camera.start();
        
        elements.videoPlaceholder.style.display = "none";
        elements.canvas.style.display = "block";
        
        state.isMonitoring = true;
        state.startTime = Date.now();
        timerInterval = setInterval(updateTimer, 1000);
        
        updateStatus("active", "MONITORING ACTIVE");
        elements.startBtn.disabled = true;
        elements.stopBtn.disabled = false;
        
    } catch (e) {
        console.error(e);
        alert("Camera Access Denied or Camera Error");
    }
}

function stopSystem() {
    state.isMonitoring = false;
    if(camera) camera.stop();
    clearInterval(timerInterval);
    
    elements.canvas.style.display = "none";
    elements.videoFeed.style.opacity = "1"; 
    elements.videoPlaceholder.style.display = "flex";
    
    updateStatus("ready", "SYSTEM STOPPED");
    elements.startBtn.disabled = false;
    elements.stopBtn.disabled = true;
}

async function processFrame() {
    const tStart = performance.now();
    
    if (elements.videoFeed.style.opacity !== "0") {
        elements.videoFeed.style.opacity = "0";
    }

    elements.canvas.width = elements.videoFeed.videoWidth;
    elements.canvas.height = elements.videoFeed.videoHeight;
    ctx.drawImage(elements.videoFeed, 0, 0, elements.canvas.width, elements.canvas.height);

    const eyeDetections = await detectEyes();

    if (eyeDetections.length > 0) {
        state.noFaceDetected = false;

        let leftEyeStatus = null;
        let rightEyeStatus = null;

        for (const eye of eyeDetections) {
            const eyeState = await classifyEyeState(eye);
            
            const isLeftEye = eye.cx < elements.canvas.width / 2;
            
            if (isLeftEye) {
                leftEyeStatus = eyeState;
            } else {
                rightEyeStatus = eyeState;
            }

            drawEyeBox(eye, eyeState);
        }

        const isClosed = (leftEyeStatus === 1 && rightEyeStatus === 1);
        state.currentStatus = isClosed ? "CLOSED" : "OPEN";
        
        processDetection(isClosed);

    } else {
        processNoFaceDetected();
    }
    
    const tEnd = performance.now();
    state.inferenceTime = (tEnd - tStart).toFixed(1);
    updateUI();
}

async function detectEyes() {
    const inputSize = CONFIG.YOLO_INPUT_SIZE;
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = inputSize;
    tempCanvas.height = inputSize;
    const tCtx = tempCanvas.getContext('2d');
    
    tCtx.drawImage(elements.canvas, 0, 0, inputSize, inputSize);
    
    const imageData = tCtx.getImageData(0, 0, inputSize, inputSize).data;
    const float32Data = new Float32Array(inputSize * inputSize);
    
    for (let i = 0; i < inputSize * inputSize; i++) {
        const r = imageData[i * 4];
        const g = imageData[i * 4 + 1];
        const b = imageData[i * 4 + 2];
        const gray = 0.299 * r + 0.587 * g + 0.114 * b;
        float32Data[i] = gray / 255.0;
    }
    
    const tensor = new ort.Tensor('float32', float32Data, [1, 1, inputSize, inputSize]);
    const feeds = { input: tensor };
    const results = await yoloSession.run(feeds);
    
    const output = results.output.data;
    const gridSize = CONFIG.YOLO_GRID_SIZE;
    const numAnchors = CONFIG.YOLO_NUM_ANCHORS;
    
    const detections = [];
    const scaleX = elements.canvas.width / inputSize;
    const scaleY = elements.canvas.height / inputSize;
    
    for (let row = 0; row < gridSize; row++) {
        for (let col = 0; col < gridSize; col++) {
            for (let anchor = 0; anchor < numAnchors; anchor++) {
                const baseIdx = (row * gridSize + col) * (numAnchors * 6) + anchor * 6;
                
                const x = output[baseIdx];
                const y = output[baseIdx + 1];
                const w = output[baseIdx + 2];
                const h = output[baseIdx + 3];
                const objectness = output[baseIdx + 4];
                const classProb = output[baseIdx + 5];
                
                const confidence = objectness * classProb;
                
                if (confidence > CONFIG.EYE_CONFIDENCE_THRESHOLD) {
                    const cx = ((col + x) / gridSize) * elements.canvas.width;
                    const cy = ((row + y) / gridSize) * elements.canvas.height;
                    const width = w * elements.canvas.width;
                    const height = h * elements.canvas.height;
                    
                    detections.push({
                        cx: cx,
                        cy: cy,
                        width: width,
                        height: height,
                        confidence: confidence,
                        x1: cx - width / 2,
                        y1: cy - height / 2,
                        x2: cx + width / 2,
                        y2: cy + height / 2
                    });
                }
            }
        }
    }
    
    const filteredDetections = applyNMS(detections, 0.5);
    
    return filteredDetections;
}

function applyNMS(detections, iouThreshold) {
    if (detections.length === 0) return [];
    
    detections.sort((a, b) => b.confidence - a.confidence);
    
    const keep = [];
    const suppressed = new Set();
    
    for (let i = 0; i < detections.length; i++) {
        if (suppressed.has(i)) continue;
        
        keep.push(detections[i]);
        
        for (let j = i + 1; j < detections.length; j++) {
            if (suppressed.has(j)) continue;
            
            const iou = calculateIoU(detections[i], detections[j]);
            if (iou > iouThreshold) {
                suppressed.add(j);
            }
        }
    }
    
    return keep;
}

function calculateIoU(box1, box2) {
    const x1 = Math.max(box1.x1, box2.x1);
    const y1 = Math.max(box1.y1, box2.y1);
    const x2 = Math.min(box1.x2, box2.x2);
    const y2 = Math.min(box1.y2, box2.y2);
    
    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    const area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    const union = area1 + area2 - intersection;
    
    return intersection / union;
}

async function classifyEyeState(eyeDetection) {
    const inputSize = CONFIG.MOBILENET_INPUT_SIZE;
    
    const padding = 1.2; 
    const width = eyeDetection.width * padding;
    const height = eyeDetection.height * padding;
    const size = Math.max(width, height); 
    
    const x1 = Math.max(0, eyeDetection.cx - size / 2);
    const y1 = Math.max(0, eyeDetection.cy - size / 2);
    const x2 = Math.min(elements.canvas.width, x1 + size);
    const y2 = Math.min(elements.canvas.height, y1 + size);
    
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = inputSize;
    tempCanvas.height = inputSize;
    const tCtx = tempCanvas.getContext('2d');
    
    tCtx.drawImage(
        elements.canvas,
        x1, y1, x2 - x1, y2 - y1,
        0, 0, inputSize, inputSize
    );
    
    const isFirstEye = eyeDetection.cx < elements.canvas.width / 2;
    if (isFirstEye) {
        const debugX = elements.canvas.width - 100;
        const debugY = 80;
        ctx.fillStyle = "rgba(0,0,0,0.5)";
        ctx.fillRect(debugX - 5, debugY - 5, 74, 84);
        ctx.drawImage(tempCanvas, debugX, debugY, 64, 64);
        ctx.fillStyle = "#fff";
        ctx.font = "12px Arial";
        ctx.fillText("AI Input", debugX, debugY + 75);
    }
    
    const imageData = tCtx.getImageData(0, 0, inputSize, inputSize).data;
    
    const float32Data = new Float32Array(3 * inputSize * inputSize);
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    
    for (let i = 0; i < inputSize * inputSize; i++) {
        const r = imageData[i * 4] / 255.0;
        const g = imageData[i * 4 + 1] / 255.0;
        const b = imageData[i * 4 + 2] / 255.0;
        
        float32Data[i] = (r - mean[0]) / std[0]; 
        float32Data[inputSize * inputSize + i] = (g - mean[1]) / std[1]; 
        float32Data[2 * inputSize * inputSize + i] = (b - mean[2]) / std[2]; 
    }
    
    const tensor = new ort.Tensor('float32', float32Data, [1, 3, inputSize, inputSize]);
    const feeds = { input: tensor };
    const results = await mobilenetSession.run(feeds);
    
    const output = results.output.data;
    
    return output[0] > output[1] ? 1 : 0;
}

function processDetection(isClosed) {
    state.eyeHistory.push(isClosed ? 1 : 0);
    if (state.eyeHistory.length > CONFIG.PERCLOS_WINDOW_SIZE) state.eyeHistory.shift();
    
    const sum = state.eyeHistory.reduce((a, b) => a + b, 0);
    state.perclosScore = sum / state.eyeHistory.length;

    if (isClosed) {
        state.closedStreakFrames++;
    } else {
        state.closedStreakFrames = 0;
    }
    
    updateTimeline(isClosed);
    checkSafetyThresholds();
}

function processNoFaceDetected() {
    state.noFaceDetected = true;
    state.currentStatus = "NO FACE";
    updateStatus("warning", "NO EYES DETECTED");
    
    state.eyeHistory.push(0); 
    if(state.eyeHistory.length > CONFIG.PERCLOS_WINDOW_SIZE) state.eyeHistory.shift();
    
    updateTimeline(false, true); 
}

function checkSafetyThresholds() {
    const now = Date.now();
    if (now - state.lastAlertTime < 3000) return; 

    if (state.closedStreakFrames >= CONFIG.MICROSLEEP_THRESHOLD) {
        triggerAlert("CRITICAL", "MICROSLEEP DETECTED");
        state.lastAlertTime = now;
    } 
    else if (state.eyeHistory.length > 50 && state.perclosScore > CONFIG.PERCLOS_THRESHOLD) {
        const pct = (state.perclosScore * 100).toFixed(0);
        triggerAlert("WARNING", `FATIGUE LEVELS HIGH (${pct}%)`);
        state.lastAlertTime = now;
    }
}

function updateUI() {
    const perclosPct = (state.perclosScore * 100).toFixed(0);
    elements.perclosValue.innerText = `${perclosPct}%`;
    elements.perclosBar.style.width = `${perclosPct}%`;
    
    if(state.perclosScore > CONFIG.PERCLOS_THRESHOLD) {
        elements.perclosBar.style.background = "#ef4444"; 
    } else if (state.perclosScore > 0.15) {
        elements.perclosBar.style.background = "#f59e0b"; 
    } else {
        elements.perclosBar.style.background = "#10b981"; 
    }

    const approxSec = (state.closedStreakFrames / 30).toFixed(1);
    elements.closedStreak.innerText = `${approxSec}s`;

    elements.latencyValue.innerText = `${state.inferenceTime}ms`;
    elements.totalAlerts.innerText = state.alertCount;
    elements.eyeStateText.innerText = state.currentStatus;
    
    elements.eyeStateText.style.color = state.currentStatus === "CLOSED" ? "#ef4444" : "#10b981";
}

function drawEyeBox(eye, eyeState) {
    const color = eyeState === 1 ? "#ef4444" : "#10b981"; 
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(eye.x1, eye.y1, eye.x2 - eye.x1, eye.y2 - eye.y1);
    
    ctx.fillStyle = color;
    ctx.font = "12px Arial";
    const label = `${(eye.confidence * 100).toFixed(0)}%`;
    ctx.fillText(label, eye.x1, eye.y1 - 5);
}

function updateTimeline(isClosed, isInactive=false) {
    const bars = Array.from(elements.timeline.querySelectorAll(".timeline-bar"));
    elements.timeline.removeChild(bars[0]);
    const newBar = document.createElement("div");
    if (isInactive) newBar.className = "timeline-bar inactive";
    else newBar.className = `timeline-bar ${isClosed ? "closed" : "open"}`;
    elements.timeline.appendChild(newBar);
}

function updateStatus(type, text) {
    elements.statusIndicator.className = `status-indicator ${type}`;
    elements.statusText.textContent = text;
}

function triggerAlert(level, message) {
    state.alertCount++;
    
    const item = document.createElement("div");
    item.className = `alert-item ${level.toLowerCase()}`;
    
    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    item.innerHTML = `
        <div class="alert-time">${time}</div>
        <div class="alert-content">
            <span class="alert-badge ${level.toLowerCase()}">${level}</span>
            <span class="alert-msg">${message}</span>
        </div>
    `;
    
    const empty = elements.alertLog.querySelector(".alert-empty");
    if (empty) empty.remove();
    
    elements.alertLog.prepend(item);
    
    if (elements.alertLog.children.length > 50) {
        elements.alertLog.removeChild(elements.alertLog.lastChild);
    }
}

function updateTimer() {
    const elapsed = Math.floor((Date.now() - state.startTime) / 1000);
    const h = Math.floor(elapsed / 3600).toString().padStart(2, "0");
    const m = Math.floor((elapsed % 3600) / 60).toString().padStart(2, "0");
    const s = (elapsed % 60).toString().padStart(2, "0");
    elements.timer.textContent = `${h}:${m}:${s}`;
}

function resetStats() {
    state = { ...state, eyeHistory: [], perclosScore: 0, closedStreakFrames: 0, alertCount: 0 };
    elements.alertLog.innerHTML = '<div class="alert-empty">No critical events recorded</div>';
    const bars = elements.timeline.querySelectorAll(".timeline-bar");
    bars.forEach(b => b.className = "timeline-bar");
    updateUI();
}