/**
 * Normalize angle to be between -180 and 180 degrees.
 *
 * @export
 * @param {number} angle
 * @return {*}  {number}
 */
export function normalizeAngle(angle: number): number {
  let normalizedAngle = angle % 360;
  if (normalizedAngle > 180) {
    normalizedAngle -= 360;
  } else if (normalizedAngle < -180) {
    normalizedAngle += 360;
  }
  return normalizedAngle;
}
