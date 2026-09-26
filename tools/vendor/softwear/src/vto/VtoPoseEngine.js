// src/vto/VtoPoseEngine.js
import * as THREE from 'three';
import { computeBodyBasis, POSE } from './bodyFrame';

const clamp = (v, min, max) => Math.min(max, Math.max(min, v));
const WORLD_UP = new THREE.Vector3(0, 1, 0);
const NOSE = 0;

/**
 * Owns the garment ROOT transform: depth (Z), screen position, and yaw.
 * Arm/bone articulation is owned by SMPLXBoneMapper.
 *
 *  - ORIENTATION: the garment hangs VERTICAL (like real clothing) and YAWS with
 *    you. The yaw is the real chest-forward direction from the metric 3D body
 *    frame (Tasks Vision world landmarks), amplified — monocular depth makes the
 *    raw turn subtle, so a gain makes it read while staying on real data (not the
 *    noisy foreshortening guess). Pitch/roll are dropped so it never goes
 *    top-down or tips over.
 *  - SIZE/DEPTH: yaw-invariant (nose → shoulder-mid), so size holds as you turn.
 *
 * TUNING:
 *   - fitFactor : overall garment size.
 *   - yawGain   : how strongly it turns (1 = raw real turn, higher = more).
 *   - yawSign   : flip if it turns the wrong way.
 *   - maxYaw    : cap on turn (radians).
 */
export class VtoPoseEngine {
    constructor() {
        this.isMobile = window.innerWidth <= 768;

        this.currentPosition = new THREE.Vector3();
        this.currentScale = new THREE.Vector3(1, 1, 1);
        this.currentQuaternion = new THREE.Quaternion();
        this.hasState = false;

        this.lostFrames = 0;
        this.lostFrameThreshold = 15;

        this.visibilityThreshold = this.isMobile ? 0.1 : 0.5;
        this.positionLerp = this.isMobile ? 0.6 : 0.4;
        this.rotationSlerp = 0.35;

        this.shoulderSpan = 0.42;
        this.garmentScale = 1.0;
        this.minDepth = 0.4;
        this.maxDepth = 6.0;

        // Yaw (real 3D chest direction, amplified, vertical garment)
        this.yawGain = 1.8;
        this.yawSign = 1;
        this.maxYaw = 1.0;          // ~57°
        this.yawSmooth = 0.25;
        this.smoothYaw = 0;

        this._tmp = new THREE.Vector3();
        this._scaleVec = new THREE.Vector3();
        this._yawQuat = new THREE.Quaternion();
        this.smoothDepth = 0;
    }

    update(landmarks, worldLandmarks, garmentModel, camera) {
        if (!landmarks || !garmentModel || !camera) {
            if (garmentModel) garmentModel.visible = false;
            return;
        }

        const ls = landmarks[POSE.L_SHOULDER];
        const rs = landmarks[POSE.R_SHOULDER];
        const shouldersVisible = ls && rs &&
            (ls.visibility ?? 1) > this.visibilityThreshold &&
            (rs.visibility ?? 1) > this.visibilityThreshold;

        if (!shouldersVisible) {
            this.lostFrames++;
            garmentModel.visible = this.lostFrames <= this.lostFrameThreshold && this.hasState;
            return;
        }
        this.lostFrames = 0;
        garmentModel.visible = true;

        const shoulderMidX = (ls.x + rs.x) / 2;
        const shoulderMidY = (ls.y + rs.y) / 2;
        const shoulderW = Math.max(Math.hypot(ls.x - rs.x, ls.y - rs.y), 0.001);
        // Normalized tracked shoulder midpoint, for the fit harness/scorer.
        this.lastTracked = { x: shoulderMidX, y: shoulderMidY, w: shoulderW };
        const nose2 = landmarks[0];
        if (nose2) this.lastNose = { x: nose2.x, y: nose2.y };
        const lh2 = landmarks[23];
        const rh2 = landmarks[24];
        if (lh2 && rh2 && (lh2.visibility ?? 1) > this.visibilityThreshold && (rh2.visibility ?? 1) > this.visibilityThreshold) {
            this.lastHips = { x: (lh2.x + rh2.x) / 2, y: (lh2.y + rh2.y) / 2 };
        }

        const vFOV = (camera.fov * Math.PI) / 180;
        const tanHalf = Math.tan(vFOV / 2);
        // Match the model's actual rig shoulder span to the tracked shoulders.
        // Model origin is moved to its shoulder midpoint at load time.
        const measuredDepth = clamp(
            this.shoulderSpan / (shoulderW * 2 * tanHalf * camera.aspect),
            this.minDepth, this.maxDepth
        );
        this.smoothDepth = this.smoothDepth
            ? this.smoothDepth + (measuredDepth - this.smoothDepth) * 0.35
            : measuredDepth;
        const depthZ = camera.position.z - this.smoothDepth;
        const target = this._projectAtDepth(shoulderMidX, shoulderMidY, depthZ, camera, tanHalf, this._tmp);

        // Real yaw from the 3D chest-forward axis, amplified; garment stays vertical.
        const basis = computeBodyBasis(worldLandmarks, this.visibilityThreshold, 0);
        if (basis) {
            const z = basis.z; // chest-forward in scene axes
            // Image-space landmarks (the Holistic fallback) carry only tiny,
            // noisy z values, so their cross-product forward axis is mostly
            // x/y noise: a frontal subject read as a ~57 degree turn. Only
            // trust large yaw from metric world landmarks; clamp the rest.
            const lms = worldLandmarks;
            const isMetric = lms && lms.some((lm) => Math.abs(lm.x) > 1.5 || Math.abs(lm.y) > 1.5);
            const maxYaw = isMetric ? this.maxYaw : 0.15;
            let yaw = Math.atan2(z.x, z.z) * this.yawSign * (isMetric ? this.yawGain : 1.0);
            yaw = clamp(yaw, -maxYaw, maxYaw);
            this.smoothYaw += (yaw - this.smoothYaw) * this.yawSmooth;
        }
        const targetQuat = this._yawQuat.setFromAxisAngle(WORLD_UP, this.smoothYaw);

        if (!this.hasState) {
            this.currentPosition.copy(target);
            this.currentScale.setScalar(this.garmentScale);
            this.currentQuaternion.copy(targetQuat);
            this.hasState = true;
        } else {
            this.currentPosition.lerp(target, this.positionLerp);
            this.currentScale.lerp(this._scaleVec.setScalar(this.garmentScale), 0.3);
            this.currentQuaternion.slerp(targetQuat, this.rotationSlerp);
        }

        garmentModel.position.copy(this.currentPosition);
        garmentModel.scale.copy(this.currentScale);
        garmentModel.quaternion.copy(this.currentQuaternion);

        // Debug hook for the headless fit harness (mirrors softWearPerformance).
        const perfHook = window.softWearPerformance;
        if (perfHook) {
            perfHook.fitDebug = {
                depth: +this.smoothDepth.toFixed(3),
                position: {
                    x: +this.currentPosition.x.toFixed(3),
                    y: +this.currentPosition.y.toFixed(3),
                    z: +this.currentPosition.z.toFixed(3),
                },
                scale: +this.currentScale.x.toFixed(3),
                yaw: +this.smoothYaw.toFixed(3),
                shoulderSpan: +this.shoulderSpan.toFixed(3),
                measuredRigSpan: this.measuredRigSpan != null ? +this.measuredRigSpan.toFixed(3) : null,
                measuredClothWidth: this.measuredClothWidth != null ? +this.measuredClothWidth.toFixed(3) : null,
                trackedShoulderMid: this.lastTracked || null,
                trackedHipsMid: this.lastHips || null,
                trackedNose: this.lastNose || null,
                garmentBox: this._fitBox(garmentModel),
            };
        }
    }

    // World-space size/center of the garment, for the fit harness.
    _fitBox(model) {
        const box = new THREE.Box3().setFromObject(model);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        return {
            size: { x: +size.x.toFixed(3), y: +size.y.toFixed(3), z: +size.z.toFixed(3) },
            center: { x: +center.x.toFixed(3), y: +center.y.toFixed(3), z: +center.z.toFixed(3) },
        };
    }

    _projectAtDepth(nx, ny, z, camera, tanHalf, out) {
        const dist = Math.abs(z - camera.position.z);
        const height = 2 * tanHalf * dist;
        const width = height * camera.aspect;
        out.set((nx - 0.5) * width, -(ny - 0.5) * height, z);
        return out;
    }
}
