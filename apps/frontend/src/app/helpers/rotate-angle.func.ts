import { normalizeAngle } from './normalize-angle.func';

export function rotateAngle(angle: number, degreeRotation: number): number {
  return normalizeAngle(angle + degreeRotation);
}
