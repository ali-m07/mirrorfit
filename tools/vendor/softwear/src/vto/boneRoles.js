const ROLES = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist'];

export function resolveBoneRoles(bones) {
  const result = {};
  for (const bone of bones || []) {
    const name = bone.name.toLowerCase().replace(/[^a-z0-9]/g, '');
    for (const role of ROLES) {
      const key = role.replace('_', '');
      if (name.includes(key)) result[role] = bone;
    }
  }
  return result;
}
