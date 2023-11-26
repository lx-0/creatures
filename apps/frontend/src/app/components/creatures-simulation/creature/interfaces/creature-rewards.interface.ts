/**
 * Represents the rewards associated with different actions performed by a creature.
 */
export interface CreatureRewards {
  /**
   * Action: Creature successfully eats food.
   * Reward: Large positive value (e.g., +5) as this is a highly desirable outcome.
   */
  eating: number;

  /**
   * Action: Creature moves without purpose, not towards food or away from danger.
   * Reward: Zero or very small positive/negative value as there's no significant benefit or harm.
   */
  idleOrInefficientMovement: number;

  /**
   * Action: Creature exhausts itself by moving excessively.
   * Reward: Negative value (e.g., -3) to discourage this behavior.
   */
  exhaustingMovement: number;

  /**
   * Action: Creature moves towards food or away from danger.
   * Reward: Positive value (e.g., +1) to encourage this behavior.
   */
  movementTowardsFood: number;

  movementTowardsSafeArea: number;

  movementTowardsPredator: number;

  movementDirectionalChange: number;

  /**
   * Action: Creature collides with an obstacle or another creature.
   * Reward: Negative value (e.g., -1) to discourage this behavior.
   */
  colliding: number;
}
