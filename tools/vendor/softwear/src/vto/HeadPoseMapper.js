import * as THREE from 'three';

const EYE_LEFT = 33;
const EYE_RIGHT = 263;
const FOREHEAD = 10;
const EAR_LEFT = 234;
const EAR_RIGHT = 454;
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

/** Maps an accessory from image landmarks into the render camera's plane. */
export class HeadPoseMapper {
    constructor() {
        this.lastPosition = new THREE.Vector3();
        this.lastQuaternion = new THREE.Quaternion();
        this.lastScale = new THREE.Vector3(1, 1, 1);
        this.isInitialised = false;
        this.modelWidth = 0.16;
        this.depth = 1.4;
    }

    setModelWidth(width) {
        if (Number.isFinite(width) && width > 0.001) this.modelWidth = width;
        this.isInitialised = false;
    }

    getLandmark(landmarks, index) {
        const point = landmarks?.[index];
        return point && Number.isFinite(point.x) && Number.isFinite(point.y) ? point : null;
    }

    calculateProportionalScale(faceWidth, garmentDetails, camera = null) {
        const fov = (camera?.fov || 50) * Math.PI / 180;
        const aspect = camera?.aspect || 1;
        const planeWidth = 2 * Math.tan(fov / 2) * this.depth * aspect;
        const fit = garmentDetails?.scale || 1;
        return Math.max(0.01, faceWidth * planeWidth * fit / this.modelWidth);
    }

    update(faceLandmarks, garmentDetails, camera = null) {
        const leftEye = this.getLandmark(faceLandmarks, EYE_LEFT);
        const rightEye = this.getLandmark(faceLandmarks, EYE_RIGHT);
        if (!leftEye || !rightEye) return null;

        const glasses = garmentDetails?.headType === 'glasses' || garmentDetails?.category === 'glasses';
        const leftEar = this.getLandmark(faceLandmarks, EAR_LEFT);
        const rightEar = this.getLandmark(faceLandmarks, EAR_RIGHT);
        const forehead = this.getLandmark(faceLandmarks, FOREHEAD);
        if (!glasses && (!leftEar || !rightEar || !forehead)) return null;

        const a = glasses ? leftEye : leftEar;
        const b = glasses ? rightEye : rightEar;
        const centerX = (a.x + b.x) / 2;
        const centerY = glasses ? (leftEye.y + rightEye.y) / 2 : forehead.y;
        const featureWidth = Math.hypot(a.x - b.x, a.y - b.y);
        if (featureWidth < 0.005) return null;

        const fov = (camera?.fov || 50) * Math.PI / 180;
        const aspect = camera?.aspect || 1;
        const planeHeight = 2 * Math.tan(fov / 2) * this.depth;
        const planeWidth = planeHeight * aspect;
        const position = new THREE.Vector3(
            (centerX - 0.5) * planeWidth,
            (0.5 - centerY) * planeHeight,
            (camera?.position.z || 2.5) - this.depth
        );

        // Landmark order is anatomical; sort by image x to keep frontal roll
        // near zero instead of rotating the frame by 180 degrees.
        const sign = b.x >= a.x ? 1 : -1;
        const roll = Math.atan2(-(b.y - a.y) * sign, Math.abs(b.x - a.x));
        const quaternion = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), clamp(roll, -0.7, 0.7));
        const depthDifference = Number.isFinite(a.z) && Number.isFinite(b.z) ? (b.z - a.z) * sign : 0;
        const yaw = clamp(Math.atan2(depthDifference, Math.abs(b.x - a.x)), -0.7, 0.7);
        quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw));
        const trueWidth = featureWidth / Math.max(0.75, Math.cos(yaw));
        const scaleValue = this.calculateProportionalScale(trueWidth * (glasses ? 1.06 : 1.12), garmentDetails, camera);

        if (!this.isInitialised) {
            this.lastPosition.copy(position);
            this.lastQuaternion.copy(quaternion);
            this.lastScale.setScalar(scaleValue);
            this.isInitialised = true;
        } else {
            this.lastPosition.lerp(position, 0.45);
            this.lastQuaternion.slerp(quaternion, 0.35);
            this.lastScale.lerp(new THREE.Vector3(scaleValue, scaleValue, scaleValue), 0.35);
        }
        return { position: this.lastPosition, quaternion: this.lastQuaternion, scale: this.lastScale };
    }
}
