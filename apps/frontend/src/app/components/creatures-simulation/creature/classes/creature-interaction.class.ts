import { normalizeAngle, radiansToDegrees } from '../../../../helpers';
import { Position } from '../../environment.class';
import { Creature } from './creature.class';

export class CreatureInteraction {
  static distanceBetweenCreatures(
    creature1: Creature,
    creature2: Creature,
    newPositionCreature1?: Position
  ): number {
    const dx =
      (newPositionCreature1?.x ?? creature1.attributes.x) -
      creature2.attributes.x;
    const dy =
      (newPositionCreature1?.y ?? creature1.attributes.y) -
      creature2.attributes.y;
    return (
      Math.hypot(dx, dy) - creature1.attributes.size - creature2.attributes.size
    );
  }

  static areCreaturesTouching(
    creature1: Creature,
    creature2: Creature,
    newPositionCreature1?: Position
  ) {
    return (
      CreatureInteraction.distanceBetweenCreatures(
        creature1,
        creature2,
        newPositionCreature1
      ) <= 0
    );
  }

  static angleToCreature(currentCreature: Creature, targetCreature: Creature) {
    const dx = targetCreature.attributes.x - currentCreature.attributes.x;
    const dy = targetCreature.attributes.y - currentCreature.attributes.y;
    const angleInRadians = Math.atan2(dy, dx);
    return normalizeAngle(radiansToDegrees(angleInRadians)); // Convert to degrees and normalize
  }

  static isFacingTarget(
    currentCreature: Creature,
    targetCreature: Creature
  ): boolean {
    // Deviation in degrees
    const targetAngle = CreatureInteraction.angleToCreature(
      currentCreature,
      targetCreature
    );
    const directionAngle = currentCreature.attributes.angle; // Assuming this is in degrees

    let angleDifference = Math.abs(targetAngle - directionAngle);
    angleDifference =
      angleDifference > 180 ? 360 - angleDifference : angleDifference; // Shortest path difference

    return angleDifference < Creature.VIEW_ANGLE;
  }
}
