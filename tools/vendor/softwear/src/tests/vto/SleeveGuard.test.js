// src/tests/vto/SleeveGuard.test.js — regression tests for the SMPLXBoneMapper
// drape/degenerate guards, using the real harness feed values.
//
// Defect reproduced here: with arms-down feed landmarks, the tracked left arm
// direction is nearly vertical (elbow directly below the shoulder), which made
// the driven sleeve hang flush against the torso panel and disappear behind it,
// while the untracked right arm held the A-pose bind and stayed visible. The
// mapper now enforces a minimum outward drape tilt and rejects implausible
// swings / degenerate joints.
import { SMPLXPoseMapper } from '../../vto/SMPLXBoneMapper';
import { POSE } from '../../vto/bodyFrame';
import * as THREE from 'three';

// GLB-like chain (baseballjacket): left/right shoulders at +-x, elbows out-down.
function riggedModel() {
    const model = new THREE.Group();
    const root = new THREE.Bone(); root.name = 'root';
    const lShoulder = new THREE.Bone(); lShoulder.name = 'left_shoulder';
    lShoulder.position.set(0.131, 0, 0);
    const lElbow = new THREE.Bone(); lElbow.name = 'left_elbow';
    lElbow.position.set(0.269, -0.079, -0.037);
    const lWrist = new THREE.Bone(); lWrist.name = 'left_wrist';
    lWrist.position.set(0.25, -0.28, -0.02);
    const rShoulder = new THREE.Bone(); rShoulder.name = 'right_shoulder';
    rShoulder.position.set(-0.122, 0, 0);
    const rElbow = new THREE.Bone(); rElbow.name = 'right_elbow';
    rElbow.position.set(-0.286, -0.053, -0.031);
    const rWrist = new THREE.Bone(); rWrist.name = 'right_wrist';
    rWrist.position.set(-0.25, -0.28, -0.02);
    root.add(lShoulder); lShoulder.add(lElbow); lElbow.add(lWrist);
    root.add(rShoulder); rShoulder.add(rElbow); rElbow.add(rWrist);
    const mesh = new THREE.SkinnedMesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial());
    mesh.add(root);
    mesh.skeleton = new THREE.Skeleton([root, lShoulder, lElbow, lWrist, rShoulder, rElbow, rWrist]);
    model.add(mesh);
    model.updateWorldMatrix(true, true);
    return model;
}

// Exact values printed by tools/e2e/probe_holistic.py for the harness feed
// (preview_standing_dark.png): left arm confidently tracked, right arm below
// the visibility threshold.
function feedLandmarks() {
    const lm = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0, visibility: 0 }));
    lm[POSE.L_SHOULDER] = { x: 0.557, y: 0.223, z: -0.332, visibility: 1.0 };
    lm[POSE.R_SHOULDER] = { x: 0.449, y: 0.232, z: 0.041, visibility: 1.0 };
    lm[POSE.L_ELBOW] = { x: 0.576, y: 0.388, z: -0.324, visibility: 0.99 };
    lm[POSE.R_ELBOW] = { x: 0.449, y: 0.376, z: 0.178, visibility: 0.21 };
    lm[POSE.L_WRIST] = { x: 0.535, y: 0.554, z: -0.277, visibility: 0.97 };
    lm[POSE.R_WRIST] = { x: 0.413, y: 0.485, z: 0.130, visibility: 0.38 };
    lm[POSE.L_HIP] = { x: 0.531, y: 0.547, z: -0.104, visibility: 1.0 };
    lm[POSE.R_HIP] = { x: 0.467, y: 0.534, z: 0.104, visibility: 1.0 };
    return lm;
}

function tposeLandmarks() {
    const lm = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0, visibility: 0 }));
    lm[POSE.L_SHOULDER] = { x: -0.20, y: -0.35, z: 0, visibility: 0.99 };
    lm[POSE.R_SHOULDER] = { x: 0.20, y: -0.35, z: 0, visibility: 0.99 };
    lm[POSE.L_ELBOW] = { x: 0.45, y: -0.35, z: 0, visibility: 0.99 };
    lm[POSE.R_ELBOW] = { x: -0.45, y: -0.35, z: 0, visibility: 0.99 };
    lm[POSE.L_WRIST] = { x: 0.62, y: -0.35, z: 0, visibility: 0.99 };
    lm[POSE.R_WRIST] = { x: -0.62, y: -0.35, z: 0, visibility: 0.99 };
    lm[POSE.L_HIP] = { x: -0.12, y: 0, z: 0, visibility: 0.99 };
    lm[POSE.R_HIP] = { x: 0.12, y: 0, z: 0, visibility: 0.99 };
    return lm;
}

function worldDir(model, fromName, toName) {
    const a = model.getObjectByName(fromName).getWorldPosition(new THREE.Vector3());
    const b = model.getObjectByName(toName).getWorldPosition(new THREE.Vector3());
    return b.sub(a).normalize();
}

test('arms-down feed: driven left sleeve keeps a visible outward drape', () => {
    const model = riggedModel();
    const mapper = new SMPLXPoseMapper();
    mapper.initializeBones(model);
    for (let i = 0; i < 40; i++) {
        mapper.applyPoseToRiggedGarment(model, feedLandmarks(), i * 33);
        model.updateWorldMatrix(true, true);
    }
    const dir = worldDir(model, 'left_shoulder', 'left_elbow');
    // Must not hang vertical against the torso (that hid the sleeve): the
    // outward x component must exceed the drape tilt's sine (~0.34).
    expect(dir.x).toBeGreaterThan(Math.sin(THREE.MathUtils.degToRad(15)));
});

test('degenerate joints (elbow on the shoulder) are rejected-and-held', () => {
    const model = riggedModel();
    const mapper = new SMPLXPoseMapper();
    mapper.initializeBones(model);
    const lm = feedLandmarks();
    lm[POSE.L_ELBOW] = { x: 0.557, y: 0.223, z: -0.332, visibility: 0.99 }; // == shoulder
    const before = model.getObjectByName('left_shoulder').quaternion.clone();
    mapper.applyPoseToRiggedGarment(model, lm, 0);
    expect(model.getObjectByName('left_shoulder').quaternion.clone().angleTo(before)).toBe(0);
});

test('T-pose still articulates both sleeves outward', () => {
    const model = riggedModel();
    const mapper = new SMPLXPoseMapper();
    mapper.initializeBones(model);
    for (let i = 0; i < 40; i++) {
        mapper.applyPoseToRiggedGarment(model, tposeLandmarks(), i * 33);
        model.updateWorldMatrix(true, true);
    }
    const left = worldDir(model, 'left_shoulder', 'left_elbow');
    const right = worldDir(model, 'right_shoulder', 'right_elbow');
    // horizontal spread: |x| dominates over the drop
    expect(Math.abs(left.x)).toBeGreaterThan(Math.abs(left.y));
    expect(Math.abs(right.x)).toBeGreaterThan(Math.abs(right.y));
});
