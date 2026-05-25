/* global FaceDetector, FaceLandmarker, FilesetResolver */
// ═══════════════════════════════════════════════════════════════
//      Core Logic
// ═══════════════════════════════════════════════════════════════

// ─── STATE ───────────────────────────────────────────────────────────────────
const S = {
    mode: 'upload',
    stream: null,
    rafId: null,

    // ── MediaPipe task instances ──
    faceDetector: null,    // BlazeFace short-range (fast live guard)
    faceLandmarker: null,  // 478 landmarks (ornaments + distance)

    // ── Running mode tracking (IMAGE | VIDEO) ──
    detectorMode: null,
    landmarkerMode: null,

    // ── Analysis state ──
    faceFound: false,
    imgData: null,        // ImageData from last analyzed frame
    canvasImg: null,      // HTMLImageElement for static upload
    lastLandmarks: null,  // Last FaceLandmarker landmarks (used at deep-analysis time)

    // ── Mesh style: 'full' | 'contour' | 'minimal' | 'off' ──
    meshStyle: 'full',
};

// ─── DOM refs ────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const canvas = $('output-canvas');
const video = $('video');
const ctx = canvas.getContext('2d');

// ─── STATUS ──────────────────────────────────────────────────────────────────
function setStatus(state, msg) {
    $('status-dot').className = 'dot ' + state;
    $('status-text').textContent = msg;
}

// ─── SIDEBAR TOGGLE ──────────────────────────────────────────────────────────
function toggleSidebar() {
    $('sidebar').classList.toggle('collapsed');
}

// ─── MODE SWITCH ─────────────────────────────────────────────────────────────
function setDistUIVisible(v) {
    $('card-dist').style.display = v ? '' : 'none';
    $('settings-panel') && ($('settings-panel').style.display = v ? '' : 'none');
}

async function switchMode(m) {
    if (m === S.mode) return;
    S.mode = m;
    $('tab-upload').className = 'mode-tab' + (m === 'upload' ? ' active' : '');
    $('tab-webcam').className = 'mode-tab' + (m === 'webcam' ? ' active' : '');
    if (m === 'webcam') {
        $('upload-controls').classList.remove('visible');
        $('drop-zone').classList.add('hidden');
        setDistUIVisible(true);
        startCam();
    } else {
        stopCam();
        await Promise.all([ensureDetectorMode('IMAGE'), ensureLandmarkerMode('IMAGE')]);
        $('drop-zone').classList.remove('hidden');
        $('upload-controls').classList.remove('visible');
        video.style.display = 'none';
        canvas.style.display = 'none';
        $('analyze-btn').disabled = true;
        setDistUIVisible(false);
    }
}

// ─── MEDIAPIPE INIT ───────────────────────────────────────────────────────────
const WASM_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';
const MODEL_BASE = 'https://storage.googleapis.com/mediapipe-models';

const MODELS = {
    detector: `${MODEL_BASE}/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite`,
    landmarker: `${MODEL_BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
};

const BACKEND_URL = 'https://medha-capstone-project.onrender.com';

async function waitForMPTasks() {
    if (window.FaceLandmarker) return;
    await new Promise(r => window.addEventListener('mp-tasks-ready', r, { once: true }));
}

let _visionPromise = null;
function getVision() {
    if (!_visionPromise) _visionPromise = FilesetResolver.forVisionTasks(WASM_CDN);
    return _visionPromise;
}

async function makeDetector(mode) {
    const v = await getVision();
    return FaceDetector.createFromOptions(v, {
        baseOptions: { modelAssetPath: MODELS.detector, delegate: 'GPU' },
        runningMode: mode,
        minDetectionConfidence: 0.5,
        minSuppressionThreshold: 0.3,
    });
}

async function makeLandmarker(mode) {
    const v = await getVision();
    return FaceLandmarker.createFromOptions(v, {
        baseOptions: { modelAssetPath: MODELS.landmarker, delegate: 'GPU' },
        runningMode: mode,
        numFaces: 1,
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: false,
    });
}

async function initAllTasks() {
    try {
        setStatus('processing', 'Downloading MediaPipe WASM + models (first load only)…');
        await waitForMPTasks();
        [S.faceDetector, S.faceLandmarker] = await Promise.all([
            makeDetector('IMAGE'),
            makeLandmarker('IMAGE'),
        ]);
        S.detectorMode = 'IMAGE';
        S.landmarkerMode = 'IMAGE';
        setStatus('ready', 'Ready — upload a portrait or start webcam');
    } catch (e) {
        setStatus('error', 'Init failed: ' + e.message);
        console.error(e);
    }
}

async function ensureDetectorMode(mode) {
    if (!S.faceDetector || S.detectorMode === mode) return;
    await S.faceDetector.setOptions({ runningMode: mode });
    S.detectorMode = mode;
}
async function ensureLandmarkerMode(mode) {
    if (!S.faceLandmarker || S.landmarkerMode === mode) return;
    await S.faceLandmarker.setOptions({ runningMode: mode });
    S.landmarkerMode = mode;
}

// ─── IMAGE UPLOAD ─────────────────────────────────────────────────────────────
function handleFile(e) {
    const file = e.target.files[0];
    if (!file) return;
    // Reset value so selecting the same file again triggers onchange
    e.target.value = '';
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = async () => {
        S.canvasImg = img;
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        ctx.drawImage(img, 0, 0);
        $('drop-zone').classList.add('hidden');
        canvas.style.display = 'block';
        $('upload-controls').classList.add('visible');
        setDistUIVisible(false);
        setStatus('processing', 'Analyzing…');
        await runFaceLandmarker(img);
    };
    img.src = url;
}

function resetUpload() {
    // Clear state
    S.canvasImg = null;
    S.imgData = null;
    S.lastLandmarks = null;
    S.faceFound = false;

    // Reset canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.style.display = 'none';

    // Reset UI
    $('drop-zone').classList.remove('hidden');
    $('upload-controls').classList.remove('visible');
    $('analyze-btn').disabled = true;

    // Reset detection sidebar cards
    $('val-detect').textContent = '—';
    $('det-arc-pct').textContent = '—';
    $('det-arc').setAttribute('stroke-dasharray', '0 113');
    $('det-count').textContent = '—';
    $('card-detect').classList.remove('lit');
    $('val-dist').textContent = '—';
    $('card-dist').classList.remove('lit');

    setStatus('ready', 'Ready — upload a portrait or start webcam');
}

// Drag & drop
const wrap = $('canvas-wrap');
wrap.addEventListener('dragover', e => e.preventDefault());
wrap.addEventListener('drop', e => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('image/')) handleFile({ target: { files: [f] } });
});

// ─── WEBCAM ──────────────────────────────────────────────────────────────────
async function startCam() {
    try {
        S.stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
        });
        video.srcObject = S.stream;
        video.style.display = 'block';
        $('cam-controls').classList.add('visible');
        video.onloadeddata = async () => {
            await Promise.all([ensureDetectorMode('VIDEO'), ensureLandmarkerMode('VIDEO')]);
            setStatus('ready', 'Webcam active — analyzing live');
            loopCam();
        };
    } catch (err) {
        setStatus('error', 'Camera: ' + err.message);
        switchMode('upload');
    }
}

function stopCam() {
    if (S.stream) { S.stream.getTracks().forEach(t => t.stop()); S.stream = null; }
    if (S.rafId) { cancelAnimationFrame(S.rafId); S.rafId = null; }
    $('cam-controls').classList.remove('visible');
    video.style.display = 'none';
}

function loopCam() {
    if (!S.stream || !S.faceDetector || !S.faceLandmarker) {
        S.rafId = requestAnimationFrame(loopCam);
        return;
    }
    if (video.readyState >= 2) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.save();
        ctx.scale(-1, 1);
        ctx.drawImage(video, -canvas.width, 0);
        ctx.restore();
        S.imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        canvas.style.display = 'block';

        const ts = performance.now();

        // Stage 1: FaceDetector guard (BlazeFace, ~2 ms)
        const det = S.faceDetector.detectForVideo(video, ts);
        updateDetectionUI(det, canvas.width, canvas.height);

        if (det.detections && det.detections.length > 0) {
            // Stage 2: FaceLandmarker (only when face confirmed)
            const lmResult = S.faceLandmarker.detectForVideo(video, ts);
            onFaceResults(lmResult, det);
        } else {
            S.faceFound = false;
            setStatus('processing', 'No face detected — please look at the camera');
        }
    }
    S.rafId = requestAnimationFrame(loopCam);
}

function captureFrame() {
    $('analyze-btn').disabled = false;
    setDistUIVisible(false);
    setStatus('ready', 'Frame captured — ready for deep analysis');
}

// ─── FACE PIPELINE (static image) ────────────────────────────────────────────
async function runFaceLandmarker(source) {
    if (!S.faceDetector || !S.faceLandmarker) return;
    await Promise.all([ensureDetectorMode('IMAGE'), ensureLandmarkerMode('IMAGE')]);
    S.imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    const det = S.faceDetector.detect(source);
    updateDetectionUI(det, canvas.width, canvas.height);

    if (!det.detections || det.detections.length === 0) {
        setStatus('error', 'No face detected — try another image');
        return;
    }
    const lmResult = S.faceLandmarker.detect(source);
    onFaceResults(lmResult, det);
}

// ─── DISTANCE RANGE SETTINGS ─────────────────────────────────────────────────
const DIST = { min: 10, max: 28 };

function toggleSettings() {
    const body = $('settings-body');
    const chev = $('settings-chevron');
    const open = body.classList.toggle('visible');
    chev.classList.toggle('open', open);
}

function onDistSlider() {
    let mn = parseInt($('dist-min').value);
    let mx = parseInt($('dist-max').value);
    if (mn >= mx - 3) { mn = mx - 3; $('dist-min').value = mn; }
    DIST.min = mn; DIST.max = mx;
    $('dist-min-val').textContent = mn + '%';
    $('dist-max-val').textContent = mx + '%';
    updateDistZone();
}

function ipdToTrackPct(ipd) {
    return Math.min(100, Math.max(0, ((ipd - 5) / (40 - 5)) * 100));
}

function updateDistZone() {
    const zl = ipdToTrackPct(DIST.min);
    const zr = ipdToTrackPct(DIST.max);
    $('dist-zone').style.left = zl + '%';
    $('dist-zone').style.width = (zr - zl) + '%';
}

updateDistZone();

// ─── DISTANCE CALCULATION ────────────────────────────────────────────────────
function calcFaceIPD(lm, W, H) {
    const lEye = lm[468] || lm[33];
    const rEye = lm[473] || lm[263];
    const dx = (rEye.x - lEye.x) * W;
    const dy = (rEye.y - lEye.y) * H;
    return (Math.sqrt(dx * dx + dy * dy) / W) * 100;
}

// ─── DISTANCE UI ─────────────────────────────────────────────────────────────
function updateDistanceUI(ipd) {
    const pin = $('dist-pin');
    const stat = $('dist-status');
    const valEl = $('val-dist');
    const subEl = $('sub-dist');

    pin.style.left = ipdToTrackPct(ipd) + '%';

    let label, zone, statusText, statusClass;
    if (ipd < DIST.min) {
        label = 'Too Far'; zone = 'too-far'; statusText = 'Move closer to camera'; statusClass = 'far';
    } else if (ipd > DIST.max) {
        label = 'Too Close'; zone = 'too-close'; statusText = 'Step back a little'; statusClass = 'near';
    } else {
        label = 'Good Distance'; zone = 'in-range'; statusText = 'Distance is within range'; statusClass = 'ok';
    }

    pin.className = 'dist-pin ' + zone;
    valEl.textContent = label;
    valEl.style.color = zone === 'in-range' ? 'var(--sage)' : zone === 'too-close' ? 'var(--rose)' : 'var(--gold)';
    subEl.textContent = `IPD ${ipd.toFixed(1)}% of frame · range ${DIST.min}–${DIST.max}%`;
    stat.textContent = statusText;
    stat.className = 'dist-status ' + statusClass;
    $('card-dist').classList.add('lit');
}

// ─── DETECTION UI ────────────────────────────────────────────────────────────
function updateDetectionUI(det, W, H) {
    if (!det.detections || det.detections.length === 0) return;
    const d = det.detections[0];
    const score = Math.round((d.categories?.[0]?.score ?? 0) * 100);
    const bb = d.boundingBox;
    const circ = 2 * Math.PI * 18;

    $('val-detect').textContent = `${score}% confidence`;
    $('det-arc-pct').textContent = score + '%';
    $('det-arc').setAttribute('stroke-dasharray', `${(score / 100) * circ} ${circ}`);
    $('det-count').textContent = det.detections.length;
    $('card-detect').classList.add('lit');

    if (bb && S.meshStyle !== 'off') {
        ctx.strokeStyle = 'rgba(184,126,168,0.45)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.strokeRect(bb.originX, bb.originY, bb.width, bb.height);
        ctx.setLineDash([]);
    }
}

// ─── RESULTS HANDLER ─────────────────────────────────────────────────────────
function onFaceResults(result, det) {
    if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
        if (S.mode === 'upload') setStatus('error', 'No face landmarks — try another image');
        S.faceFound = false;
        return;
    }

    const lm = result.faceLandmarks[0];
    const W = canvas.width, H = canvas.height;

    S.lastLandmarks = lm;

    const ipd = calcFaceIPD(lm, W, H);
    updateDistanceUI(ipd);
    const inRange = (ipd >= DIST.min && ipd <= DIST.max);

    if (S.mode === 'webcam' && !inRange) {
        drawOverlay(lm, W, H);
        setStatus('processing', `Out of range · ${ipd < DIST.min ? 'Move closer' : 'Step back'} · IPD ${ipd.toFixed(1)}%`);
        return;
    }

    if (S.mode === 'upload' && S.canvasImg) ctx.drawImage(S.canvasImg, 0, 0, W, H);

    S.imgData = ctx.getImageData(0, 0, W, H);
    S.faceFound = true;

    if (det) updateDetectionUI(det, W, H);
    drawOverlay(lm, W, H);

    $('analyze-btn').disabled = false;
    setStatus('ready', `Face in range · ${lm.length} landmarks · IPD ${ipd.toFixed(1)}%`);
}

// ─── COLOR SPACE MATH ─────────────────────────────────────────────────────────
function sRGBtoLinear(c) {
    c /= 255;
    return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

function rgbToLab(r, g, b) {
    const rl = sRGBtoLinear(r), gl = sRGBtoLinear(g), bl = sRGBtoLinear(b);
    let X = rl * 41.239 + gl * 35.758 + bl * 18.048;
    let Y = rl * 21.267 + gl * 71.515 + bl * 7.218;
    let Z = rl * 1.933 + gl * 11.919 + bl * 95.053;
    const f = (t, n) => { const v = t / n; return v > 0.008856 ? Math.cbrt(v) : 7.787 * v + 16 / 116; };
    const fx = f(X, 95.047), fy = f(Y, 100.0), fz = f(Z, 108.883);
    return { L: 116 * fy - 16, a: 500 * (fx - fy), b: 200 * (fy - fz) };
}

// ─── PIXEL SAMPLING ──────────────────────────────────────────────────────────
const SKIN_LM = [1, 4, 5, 6, 10, 36, 50, 100, 101, 116, 117, 123, 205, 234,
    266, 280, 329, 330, 345, 346, 425, 338, 297, 332, 284, 251];

function sampleArea(imgData, lm, W, H, r = 4) {
    const cx = Math.round(lm.x * W);
    const cy = Math.round(lm.y * H);
    let rv = 0, gv = 0, bv = 0, n = 0;
    for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
            const px = cx + dx, py = cy + dy;
            if (px >= 0 && px < W && py >= 0 && py < H) {
                const i = (py * W + px) * 4;
                rv += imgData.data[i]; gv += imgData.data[i + 1]; bv += imgData.data[i + 2];
                n++;
            }
        }
    }
    return n > 0 ? { r: rv / n, g: gv / n, b: bv / n } : null;
}


// ─── SKIN TONE ANALYSIS ─────────────────────────────
function analyzeSkinTone(imgData, lm, W, H) {
    let rS = 0, gS = 0, bS = 0, n = 0;
    for (const idx of SKIN_LM) {
        if (!lm[idx]) continue;
        const px = sampleArea(imgData, lm[idx], W, H);
        if (px) { rS += px.r; gS += px.g; bS += px.b; n++; }
    }
    if (n === 0) return null;

    const r = rS / n, g = gS / n, b = bS / n;
    const lab = rgbToLab(r, g, b);
    const ita = Math.atan2(lab.L - 50, lab.b) * (180 / Math.PI);

    let fitzpatrick, label;
    if (ita > 55) { fitzpatrick = 'Type I'; label = 'Very Light'; }
    else if (ita > 41) { fitzpatrick = 'Type II'; label = 'Light'; }
    else if (ita > 28) { fitzpatrick = 'Type III'; label = 'Light Medium'; }
    else if (ita > 10) { fitzpatrick = 'Type IV'; label = 'Medium Dark'; }
    else if (ita > -30) { fitzpatrick = 'Type V'; label = 'Dark'; }
    else { fitzpatrick = 'Type VI'; label = 'Deep Dark'; }

    // Use Lab a* / b* directly — more reliable than raw RGB deltas.
    //   a* > 0 → red/pink cast    b* > 0 → yellow/golden cast
    //
    // Cool:   reddish with little yellow  (a* high, b* low)
    // Warm:   yellow/golden cast           (b* clearly dominant)
    // Olive:  greenish-yellow, MEDIUM-DARK skin only (low a*, moderate b*, L < 72)
    //         — very fair skin with low a* is NOT olive, just pale warm/neutral
    // Neutral: balanced or below detection threshold
    const chroma = Math.sqrt(lab.a * lab.a + lab.b * lab.b);
    const hueAB  = Math.atan2(lab.b, lab.a) * (180 / Math.PI);
    let undertone;
    if (chroma < 8) {
        undertone = 'Neutral';                               // too desaturated to call
    } else if (lab.b < 6 && lab.a > 8) {
        undertone = 'Cool · Pink-Rose';                      // pink/red, little yellow
    } else if (lab.a < 6 && lab.b > 12 && lab.L < 72) {
        undertone = 'Neutral-Olive';                         // olive: only medium-dark skin
    } else if (lab.b > 14 || (hueAB > 55 && chroma > 12)) {
        undertone = 'Warm · Golden';                         // clear yellow/golden cast
    } else {
        undertone = 'Neutral';
    }

    return { fitzpatrick, label, undertone };
}

// ─── UNDERTONE LOOKUP TABLE ──────────────────────────────────────────────────
// Recommendations come from this table
const UNDERTONE_RECS = {
    Cool: { metals: ['Silver', 'White Gold', 'Platinum'], colors: ['Blue', 'Pink', 'Purple', 'Lavender', 'Emerald'] },
    Warm: { metals: ['Gold', 'Rose Gold', 'Copper'], colors: ['Orange', 'Brown', 'Terracotta', 'Olive', 'Warm Red'] },
    Neutral: { metals: ['Gold', 'Silver', 'Rose Gold'], colors: ['Teal', 'Dusty Rose', 'Mauve', 'Soft White', 'Burgundy'] },
};

// Derive canonical undertone from skin pixel analysis
function deriveUndertone(skin) {
    if (!skin) return 'Neutral';
    if (skin.undertone.includes('Warm')) return 'Warm';
    if (skin.undertone.includes('Cool')) return 'Cool';
    return 'Neutral';
}

// ─── MESH STYLE ───────────────────────────────────────────────────────────────
function setMesh(style) {
    S.meshStyle = style;
}

// ─── OVERLAY DRAWING ─────────────────────────────────────────────────────────
function drawConnections(lm, connections, W, H) {
    connections.forEach(({ start, end }) => {
        const s = lm[start], e = lm[end];
        if (!s || !e) return;
        ctx.beginPath();
        ctx.moveTo(s.x * W, s.y * H);
        ctx.lineTo(e.x * W, e.y * H);
        ctx.stroke();
    });
}

function drawOverlay(lm, W, H) {
    if (S.meshStyle === 'off') return;
    ctx.save();

    if (S.meshStyle === 'full') {
        ctx.strokeStyle = 'rgba(184,126,168,0.10)';
        ctx.lineWidth = 0.5;
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_TESSELATION, W, H);
    }

    if (S.meshStyle === 'full' || S.meshStyle === 'contour') {
        ctx.strokeStyle = 'rgba(184,126,168,0.45)'; ctx.lineWidth = 1;
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, W, H);

        ctx.strokeStyle = 'rgba(184,126,168,0.70)'; ctx.lineWidth = 0.9;
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, W, H);
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, W, H);

        ctx.strokeStyle = 'rgba(184,126,168,0.55)'; ctx.lineWidth = 0.9;
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, W, H);
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, W, H);

        ctx.strokeStyle = 'rgba(207,168,196,0.65)'; ctx.lineWidth = 0.9;
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_LIPS, W, H);

        ctx.strokeStyle = 'rgba(122,191,160,0.60)'; ctx.lineWidth = 1;
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, W, H);
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, W, H);

        [468, 473].forEach(idx => {
            if (!lm[idx]) return;
            ctx.beginPath();
            ctx.arc(lm[idx].x * W, lm[idx].y * H, 2.5, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(122,191,160,0.7)';
            ctx.fill();
        });
    }

    if (S.meshStyle === 'minimal') {
        ctx.strokeStyle = 'rgba(184,126,168,0.40)'; ctx.lineWidth = 0.8;
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, W, H);
        ctx.strokeStyle = 'rgba(184,126,168,0.60)';
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, W, H);
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, W, H);
        ctx.strokeStyle = 'rgba(207,168,196,0.55)';
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_LIPS, W, H);
        ctx.strokeStyle = 'rgba(122,191,160,0.50)';
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, W, H);
        drawConnections(lm, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, W, H);
    }

    ctx.restore();
}

// ─── FACE CROP ───────────────────────────────────────────────────────────────
function getFaceCrop(lm, W, H) {
    let minX = 1, maxX = 0, minY = 1, maxY = 0;
    for (const pt of lm) {
        if (pt.x < minX) minX = pt.x;
        if (pt.x > maxX) maxX = pt.x;
        if (pt.y < minY) minY = pt.y;
        if (pt.y > maxY) maxY = pt.y;
    }
    const pw = (maxX - minX) * 0.18, ph = (maxY - minY) * 0.18;
    const x = Math.max(0, Math.floor((minX - pw) * W));
    const y = Math.max(0, Math.floor((minY - ph) * H));
    const x2 = Math.min(W, Math.ceil((maxX + pw) * W));
    const y2 = Math.min(H, Math.ceil((maxY + ph) * H));
    return { x, y, w: x2 - x, h: y2 - y };
}

// ─── MODAL ───────────────────────────────────────────────────────────────────
function openModal() {
    $('overlay').classList.add('open');
    runDeepAnalysis();
}
function closeModal() { $('overlay').classList.remove('open'); }
function maybeClose(e) { if (e.target === $('overlay')) closeModal(); }

async function runDeepAnalysis() {
    $('modal-body').innerHTML = `
    <div class="loading">
      <div class="spinner"></div>
      <div class="loading-label">Preparing analysis…</div>
    </div>`;

    // ── Crop to face bounding box — eliminates background colour noise ──
    let b64 = canvas.toDataURL('image/jpeg', 0.88).split(',')[1];
    let analysisImgData = S.imgData;
    let analysisLm = S.lastLandmarks;
    let analysisW = canvas.width, analysisH = canvas.height;

    if (S.lastLandmarks && S.imgData) {
        const W = canvas.width, H = canvas.height;
        const crop = getFaceCrop(S.lastLandmarks, W, H);
        if (crop.w > 0 && crop.h > 0) {
            const off = document.createElement('canvas');
            off.width = crop.w;
            off.height = crop.h;
            const offCtx = off.getContext('2d');
            offCtx.drawImage(canvas, crop.x, crop.y, crop.w, crop.h, 0, 0, crop.w, crop.h);
            b64 = off.toDataURL('image/jpeg', 0.88).split(',')[1];
            analysisImgData = offCtx.getImageData(0, 0, crop.w, crop.h);
            analysisLm = S.lastLandmarks.map(pt => ({
                ...pt,
                x: (pt.x * W - crop.x) / crop.w,
                y: (pt.y * H - crop.y) / crop.h,
            }));
            analysisW = crop.w;
            analysisH = crop.h;
        }
    }

    // ── Compute undertone deterministically on-device (never let AI re-derive) ──
    let undertone = 'Neutral', skinCtx = '';
    if (analysisImgData && analysisLm) {
        const skin = analyzeSkinTone(analysisImgData, analysisLm, analysisW, analysisH);
        undertone = deriveUndertone(skin);
        if (skin) skinCtx = `${skin.fitzpatrick} · ${skin.label}`;
    }
    const recs = UNDERTONE_RECS[undertone];

    $('modal-body').innerHTML = `
    <div class="loading">
      <div class="spinner"></div>
      <div class="loading-label">Sending to AI model…</div>
    </div>`;

    const prompt = `You are a professional beauty and grooming advisor.

Confirmed on-device readings — treat as absolute ground truth, do NOT contradict:
• Undertone: ${undertone}
• Skin tone: ${skinCtx || 'unknown'}

Step 1 — Identify the apparent gender presentation from the image.
Step 2 — Fill the "tips" array with EXACTLY 4 actionable items based on gender. The array must NEVER be empty:
  • Male/masculine → 4 practical male grooming tips tailored to a ${undertone} undertone, such as: moisturizer and SPF routine, face wash type, beard/shaving care, dark-spot or oil-control advice, hair styling. Do NOT suggest any makeup.
  • Female/feminine → 4 makeup and beauty tips suited to a ${undertone} undertone.
  • Gender-neutral → 4 versatile skin-care and appearance tips for a ${undertone} undertone.

Return ONLY a JSON object (no markdown, no preamble):
{
  "description": "Write 1–2 short sentences in a natural, conversational tone. Describe the overall appearance and vibe warmly. Do not mention or re-analyze undertone. Do not mention technical terms. Avoid generic phrases and repetition. Make each response feel slightly unique based on the image context. Do not start with 'This person' or 'The individual'.",
  "ornaments": ["list only ornaments visually present in the image — e.g. Glasses, Earrings, Necklace, Bindi. Empty array if none."],
  "tips_label": "Grooming Tips or Beauty Tips — match the person's apparent gender presentation",
  "tips": ["tip 1", "tip 2", "tip 3", "tip 4"],
  "skincare_note": "One brief, specific skincare observation"
}`;

    try {
        const res = await fetch(`${BACKEND_URL}/v1/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_b64: b64, prompt }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const data = await res.json();
        let analysis;
        try { analysis = JSON.parse(data.answer.replace(/```json|```/g, '').trim()); }
        catch { analysis = { _raw: data.answer }; }
        renderAnalysis(analysis, undertone, recs);
    } catch (err) {
        $('modal-body').innerHTML = `
      <div class="error-box">
        <div class="error-title">Analysis Unavailable</div>
        <div class="error-msg">${err.message}<br><br>
          Make sure the backend is running:<br>
          <code>python main.py</code>
        </div>
      </div>`;
    }
}

function mkList(items) {
    if (!Array.isArray(items) || !items.length) return `<p style="color:var(--muted);font-size:.78rem">${items}</p>`;
    return `<ul class="rec-list">${items.map(i => `<li>${i}</li>`).join('')}</ul>`;
}

function mkTags(items) {
    return `<div class="tag-list">${items.map(i => `<span class="tag">${i}</span>`).join('')}</div>`;
}

function renderAnalysis(a, undertone, recs) {
    if (a._raw) {
        $('modal-body').innerHTML = `<div class="m-section"><div class="m-text" style="white-space:pre-wrap">${a._raw}</div></div>`;
        return;
    }

    const ornaments = Array.isArray(a.ornaments) ? a.ornaments : [];

    const ornSection = ornaments.length
        ? `<hr class="m-divider">
    <div class="m-section">
      <span class="m-label">Detected Ornaments</span>
      ${mkTags(ornaments)}
    </div>`
        : '';

    $('modal-body').innerHTML = `
    <div class="m-section">
      <span class="m-label">Overview</span>
      <div class="m-text">${a.description || '—'}</div>
    </div>

    <hr class="m-divider">

    <div class="m-section">
      <span class="m-label">Undertone</span>
      <div class="f-value" style="font-size:1.4rem;margin:.3rem 0 .75rem">${undertone}</div>
      <div style="display:flex;flex-direction:column;gap:.55rem">
        <div>
          <div class="f-label" style="margin-bottom:.3rem">Metals</div>
          ${mkTags(recs.metals)}
        </div>
        <div>
          <div class="f-label" style="margin-bottom:.3rem">Colors</div>
          ${mkTags(recs.colors)}
        </div>
      </div>
    </div>

    ${ornSection}

    <hr class="m-divider">

    <div class="m-section">
      <span class="m-label">${a.tips_label || 'Beauty & Grooming Tips'}</span>
      ${mkList(a.tips || a.makeup_tips)}
    </div>

    <hr class="m-divider">

    <div class="m-section">
      <span class="m-label">Skincare</span>
      <div class="m-text" style="font-size:.8rem">${a.skincare_note || '—'}</div>
    </div>
  `;
}

// ─── THEME TOGGLE ────────────────────────────────────────────────────────────
function toggleTheme() {
    const isLight = document.documentElement.classList.toggle('light-mode');
    localStorage.setItem('theme', isLight ? 'light' : 'dark');
}

// Apply saved theme before first paint
(function () {
    if (localStorage.getItem('theme') === 'light') {
        document.documentElement.classList.add('light-mode');
    }
})();

// ─── BOOT ────────────────────────────────────────────────────────────────────
setDistUIVisible(false);
setStatus('processing', 'Loading MediaPipe Tasks Vision…');
initAllTasks();
