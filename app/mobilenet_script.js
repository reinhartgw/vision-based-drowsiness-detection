const CONFIG = {
    PERCLOS_WINDOW_SIZE: 300,
    PERCLOS_THRESHOLD: 0.30,
    MICROSLEEP_THRESHOLD: 15,
    SOUND_ENABLED: true,
    MAX_HISTORY: 30,
    RIGHT_EYE: [33, 133, 160, 159, 158, 144, 145, 153],
    LEFT_EYE:  [362, 263, 387, 386, 385, 373, 374, 380],
    DEBUG: true 
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

let ortSession = null;
let faceMesh = null;
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
        
        ortSession = await ort.InferenceSession.create('./models/mobilenet_eye_classifier.onnx');
        console.log("MobileNet Classifier Loaded (224x224 RGB)");

        faceMesh = new FaceMesh({locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
        }});
        
        faceMesh.setOptions({
            maxNumFaces: 1,
            refineLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        faceMesh.onResults(onFaceResults);
        
        updateStatus("ready", "SYSTEM READY");
        elements.startBtn.disabled = false;
        
    } catch (e) {
        console.error("Model Error:", e);
        alert("System Error: AI Models failed to load.");
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
                if (faceMesh) {
                    await faceMesh.send({image: elements.videoFeed});
                }
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

async function onFaceResults(results) {
    const tStart = performance.now();
    
    if (elements.videoFeed.style.opacity !== "0") {
        elements.videoFeed.style.opacity = "0";
    }

    elements.canvas.width = results.image.width;
    elements.canvas.height = results.image.height;
    ctx.drawImage(results.image, 0, 0, elements.canvas.width, elements.canvas.height);

    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        state.noFaceDetected = false;
        const landmarks = results.multiFaceLandmarks[0];

        drawFaceBox(landmarks);

        const leftEyeStatus = await predictEye(landmarks, CONFIG.LEFT_EYE, true);
        const rightEyeStatus = await predictEye(landmarks, CONFIG.RIGHT_EYE, false);

        const isClosed = (leftEyeStatus === 1 && rightEyeStatus === 1);
        state.currentStatus = isClosed ? "CLOSED" : "OPEN";
        
        if (CONFIG.DEBUG) {
            console.log(`[FINAL] Left=${leftEyeStatus === 0 ? 'OPEN' : 'CLOSED'}, Right=${rightEyeStatus === 0 ? 'OPEN' : 'CLOSED'} => Eyes: ${state.currentStatus}`);
            console.log('---');
        }
        
        if (state.isMonitoring) {
            updateStatus("active", "MONITORING ACTIVE");
        }
        
        processDetection(isClosed);

    } else {
        processNoFaceDetected();
    }
    
    const tEnd = performance.now();
    state.inferenceTime = (tEnd - tStart).toFixed(1);
    updateUI();
}


async function predictEye(landmarks, indices, drawDebug) {
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    const w = elements.canvas.width;
    const h = elements.canvas.height;

    indices.forEach(idx => {
        const x = landmarks[idx].x * w;
        const y = landmarks[idx].y * h;
        if(x < xMin) xMin = x;
        if(x > xMax) xMax = x;
        if(y < yMin) yMin = y;
        if(y > yMax) yMax = y;
    });

    const eyeWidth = xMax - xMin;
    const eyeHeight = yMax - yMin;
    const cx = xMin + (eyeWidth / 2);
    const cy = yMin + (eyeHeight / 2);
    const size = Math.max(eyeWidth, eyeHeight) * 1.5; 
    
    const startX = Math.max(0, cx - (size / 2));
    const startY = Math.max(0, cy - (size / 2));

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 224;
    tempCanvas.height = 224;
    const tCtx = tempCanvas.getContext('2d');
    
    tCtx.drawImage(elements.canvas, startX, startY, size, size, 0, 0, 224, 224);

    if(drawDebug) {
        const debugX = elements.canvas.width - 100;
        const debugY = 80;
        ctx.fillStyle = "rgba(0,0,0,0.5)";
        ctx.fillRect(debugX - 5, debugY - 5, 74, 84);
        ctx.drawImage(tempCanvas, debugX, debugY, 64, 64);
        ctx.fillStyle = "#fff";
        ctx.font = "12px Arial";
        ctx.fillText("MobileNet", debugX, debugY + 75);
    }

    const imageData = tCtx.getImageData(0, 0, 224, 224).data;
    const float32Data = new Float32Array(3 * 224 * 224);
    
    let minVal = 255;
    let maxVal = 0;
    const grayBuffer = new Float32Array(224 * 224);

    for (let i = 0; i < 224 * 224; i++) {
        const r = imageData[i * 4];
        const g = imageData[i * 4 + 1];
        const b = imageData[i * 4 + 2];
        
        const gray = 0.299 * r + 0.587 * g + 0.114 * b;
        grayBuffer[i] = gray;

        if (gray < minVal) minVal = gray;
        if (gray > maxVal) maxVal = gray;
    }

    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    const range = maxVal - minVal;

    for (let i = 0; i < 224 * 224; i++) {
        let val = grayBuffer[i];

        if (range > 0) {
            val = (val - minVal) / range; 
        } else {
            val = 0.5; 
        }

        
        const normVal = (val - mean[0]) / std[0]; 
        float32Data[i] = normVal;             
        float32Data[i + 224 * 224] = normVal;  
        float32Data[i + 2 * 224 * 224] = normVal; 
    }

    const tensor = new ort.Tensor('float32', float32Data, [1, 3, 224, 224]);
    const feeds = { input: tensor };
    const results = await ortSession.run(feeds);
    const output = results.output.data; 

    const scoreOpen = output[1];
    const scoreClosed = output[0];

    const prediction = scoreClosed > scoreOpen ? 1 : 0;
    
    if (CONFIG.DEBUG) {
        const probOpen = Math.exp(scoreOpen) / (Math.exp(scoreOpen) + Math.exp(scoreClosed));
        const probClosed = Math.exp(scoreClosed) / (Math.exp(scoreOpen) + Math.exp(scoreClosed));
        const eyeName = drawDebug ? "LEFT" : "RIGHT";
        console.log(`[${eyeName}] Open: ${(probOpen*100).toFixed(1)}% | Closed: ${(probClosed*100).toFixed(1)}% | Pred: ${prediction===1?'CLOSED':'OPEN'}`);
    }

    return prediction;
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
    updateStatus("warning", "NO FACE DETECTED");
    
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

function drawFaceBox(landmarks) {
    let xMin = 1, xMax = 0, yMin = 1, yMax = 0;
    landmarks.forEach(pt => {
        if(pt.x < xMin) xMin = pt.x;
        if(pt.x > xMax) xMax = pt.x;
        if(pt.y < yMin) yMin = pt.y;
        if(pt.y > yMax) yMax = pt.y;
    });
    
    const w = elements.canvas.width;
    const h = elements.canvas.height;
    
    const color = state.currentStatus === "CLOSED" ? "#ef4444" : "#10b981";
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(xMin * w, yMin * h, (xMax-xMin)*w, (yMax-yMin)*h);
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