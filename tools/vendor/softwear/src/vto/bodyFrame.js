import * as THREE from 'three';

export const POSE = Object.freeze({
  L_SHOULDER: 11, R_SHOULDER: 12,
  L_ELBOW: 13, R_ELBOW: 14,
  L_WRIST: 15, R_WRIST: 16,
  L_HIP: 23, R_HIP: 24,
});

const point = (landmarks, index, threshold) => {
  const lm = landmarks?.[index];
  if (!lm || (lm.visibility ?? 1) < threshold) return null;
  return new THREE.Vector3(lm.x, -lm.y, -lm.z);
};

export function limbDirection(landmarks, from, to, threshold = 0.5) {
  const a = point(landmarks, from, threshold);
  const b = point(landmarks, to, threshold);
  if (!a || !b) return null;
  const direction = b.sub(a);
  return direction.lengthSq() < 1e-8 ? null : direction.normalize();
}

export function computeBodyBasis(landmarks, threshold = 0.5) {
  const ls = point(landmarks, POSE.L_SHOULDER, threshold);
  const rs = point(landmarks, POSE.R_SHOULDER, threshold);
  const lh = point(landmarks, POSE.L_HIP, threshold);
  const rh = point(landmarks, POSE.R_HIP, threshold);
  if (!ls || !rs || !lh || !rh) return null;
  const x = rs.clone().sub(ls).normalize();
  const shoulderMid = ls.clone().add(rs).multiplyScalar(0.5);
  const hipMid = lh.clone().add(rh).multiplyScalar(0.5);
  const y = shoulderMid.sub(hipMid).normalize();
  // MediaPipe's anatomical left/right makes shoulder x point towards the
  // viewer's left in an unmirrored camera frame. y × x therefore points
  // towards the camera for a frontal pose, giving neutral yaw at rest.
  const z = y.clone().cross(x).normalize();
  if (z.lengthSq() < 1e-8) return null;
  return { x, y, z };
}
