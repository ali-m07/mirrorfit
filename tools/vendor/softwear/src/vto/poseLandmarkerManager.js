// Holistic already publishes poseWorldLandmarks in MainDisplay. Keep its
// non-fatal optional Tasks Vision branch disabled until a model is supplied.
export async function initPoseLandmarker() {
  throw new Error('No separate Tasks Vision pose model configured');
}

export function detectWorldLandmarks() {
  return null;
}
