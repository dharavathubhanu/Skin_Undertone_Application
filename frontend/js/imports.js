import {
    FaceDetector,
    FaceLandmarker,
    FilesetResolver,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs';

window.FaceDetector = FaceDetector;
window.FaceLandmarker = FaceLandmarker;
window.FilesetResolver = FilesetResolver;
window.dispatchEvent(new Event('mp-tasks-ready'));