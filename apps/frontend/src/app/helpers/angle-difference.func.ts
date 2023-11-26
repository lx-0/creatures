export function angleDifference(angle1: number, angle2: number): number {
  let diff = angle2 - angle1;
  while (diff < -180) diff += 360;
  while (diff > 180) diff -= 360;
  return diff;
}
