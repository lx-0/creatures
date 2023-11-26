import * as tf from '@tensorflow/tfjs';
import { Scalar } from '@tensorflow/tfjs';
import { flattenDeep, inRange, pick, round, times } from 'lodash';
import { generate } from 'shortid';
import {
  degreesToRadians,
  normalizeAngle,
  radiansToDegrees,
} from '../../../../helpers';
import { Environment, Position } from '../../environment.class';
import {
  CreatureActions,
  CreatureAttributes,
  CreatureCreationAttributes,
  CreatureRewards,
  CreatureStatistics,
} from '../interfaces';
import { CreatureState } from '../types';
import { CreatureInteraction } from './creature-interaction.class';
import { ExperienceBuffer } from './experience-buffer.class';
import { VisualCreature } from './visual-creature.class';

export class Creature {
  environment: Environment;
  visualRepresentation!: VisualCreature;

  id: string;
  parent?: string;
  generation: number;
  createdAt: Date;
  isPredator: boolean;
  isDead: boolean = false;

  attributes: CreatureAttributes;

  brain: tf.Sequential;
  experiences: ExperienceBuffer;
  trainings: number;
  optimizer: tf.AdamOptimizer | undefined;

  averageReward: number;
  averageRewards: CreatureRewards;
  currentRewards: CreatureRewards;

  debug: number | undefined;

  // Constants for maximum and minimum values
  // action settings
  static KILL_SWITCH = 1; // Kill switch for debugging. If set to 0, creature will not die.
  static REPRODUCTION_SWITCH = 1; // Reproduction switch for debugging. If set to 0, creature will not reproduce.
  static COLLISION_SWITCH = 0; // Collision switch for debugging. If set to 0, creature will not collide.
  // movement settings
  static MAX_SPEED = 5; // Maximum speed
  static ACCELERATION_FACTOR = 2;
  static MOVEMENT_ANGLE_FACTOR = 30;
  // reproduction settings
  static REPRODUCTION_COOLDOWN_INITIAL = 100;
  static REPRODUCTION_COOLDOWN_FACTOR_PREY = 0.01; // Cooldown in update cycles, per 100 health
  static REPRODUCTION_COOLDOWN_FACTOR_PREDATOR = 0; // Cooldown in update cycles, per 100 health
  static REPRODUCTION_COOLDOWN_ON_EAT = 30; // Cooldown per eaten creature
  // energy and health settings
  static GAIN_ENERGY_PREY = 0.1; // Energy gained per update cycle
  static GAIN_ENERGY = 0; // Energy gained per update cycle
  static GAIN_ENERGY_ON_EAT = 100; // Energy gained per eaten creature (default: 100)
  static GAIN_HEALTH_ON_EAT = 10; // Health gained per eaten creature
  static LOOSE_ENERGY_PREDATOR = 0; // Energy loosing per update cycle
  static LOOSE_ENERGY_PREY = 0; // Energy loosing per update cycle
  static LOOSE_HEALTH_WHEN_WEAK = 0.01; // Health loosing per update cycle, if weak (no energy)
  static LOOSE_ENERGY_ON_MOVE_PER_SPEED = 0.01; // Energy loosing per update cycle, per speed
  static LOOSE_ENERGY_ON_ROTATE_PER_ANGLE = 0.002; // Energy loosing per update cycle, per degree of rotation
  // view settings
  static VIEW_ANGLE = 45;
  static VIEW_BEAMS = 6;
  static VIEW_DISTANCE = 100;
  // reward settings
  static REWARD_COLLIDING_WITH_BOUNDARIES = -10000; // negative reward for colliding with boundaries
  static REWARD_SPEED_FACTOR = -0.01; // negative reward for speed (factor by speed)
  static REWARD_ROTATION_FACTOR = -1000; // negative reward for directional change (factor by rotation angle)
  static REWARD_DIRECTIONAL_CHANGE = 1000; // positive reward for directional change
  static REWARD_PREDATOR_MOVE_TOWARDS_FOOD_FACTOR = 1000; // positive reward for moving towards food (factor by speed)
  static REWARD_PREY_MOVE_TOWARDS_SAFE_AREA_FACTOR = 1; // positive reward for moving towards safe area (factor by speed)
  static REWARD_PREY_MOVE_TOWARDS_PREDATOR_FACTOR = -50; // negative reward for moving towards predator (factor by speed)
  static REWARD_ON_EAT_FACTOR = 0; // positive reward for eating (factor by health gained)

  static BRAIN_INPUT_NEURONS = 10 + Creature.VIEW_BEAMS * 3;
  /** Minimum number of experiences to start training */
  static MIN_EXPERIENCE_BATCH_SIZE = 50;
  /** Number of training cycles */
  static TRAINING_EPOCHS = 10;

  constructor(
    attributes: CreatureCreationAttributes,
    environment: Environment,
    isPredator: boolean,
    parent?: Creature
  ) {
    this.id = generate();
    this.createdAt = new Date();

    this.attributes = {
      ...attributes,
      rotation: 0,
      acceleration: 0,
      lifetime: 0,
      reproductionCooldown: Creature.REPRODUCTION_COOLDOWN_INITIAL,
      touchedCreatures: [],
      facedCreatures: [],
      viewBeams: times(Creature.VIEW_BEAMS, () => ({
        distance: Creature.VIEW_DISTANCE,
      })),
    };
    this.isPredator = isPredator;

    this.environment = environment;

    if (parent) {
      this.brain = Creature.cloneBrain(parent);
      this.parent = parent.id;
      this.generation = parent.generation + 1;
      this.move(this.attributes.size, this.attributes.angle);
    } else {
      this.brain = Creature.createBrain();
      this.generation = 0;
    }
    this.experiences = new ExperienceBuffer();
    this.trainings = 0;
    this.averageReward = 0;
    this.averageRewards = Creature.defaultRewards();
    this.currentRewards = Creature.defaultRewards();

    this.generateVisualRepresentation();
  }

  generateVisualRepresentation() {
    this.visualRepresentation = new VisualCreature(this, this.environment);
  }

  update() {
    if (!this.isDead) {
      // * change states
      this.checkLifetime();
      this.checkEnergy();
      this.checkHealth();
      this.checkReproduction();
      this.determineClosestCreatureInViewBeams();

      // * get current state
      const state = this.getCurrentState();

      // * make decision
      const actions = Creature.defaultActions(this.makeDecision(state));

      // * take action
      const rewards = Creature.defaultRewards(
        this.behaveBasedOnDecision(actions)
      );
      this.currentRewards = {
        ...rewards,
        ...Object.fromEntries(
          Object.entries(rewards).map(([key, value]) => [key, round(value, 2)])
        ),
      };

      // * state changed after taking action
      const stateAfterAction = this.getCurrentState();

      // * store experience
      this.recordExperience(state, actions, stateAfterAction, rewards);
    }
  }

  // Additional methods for behavior, movement, learning etc.

  getCurrentState(): CreatureState {
    return [
      this.attributes.x,
      this.attributes.y,
      this.attributes.size,
      this.attributes.speed,
      this.attributes.acceleration,
      this.attributes.angle,
      this.attributes.rotation,
      this.attributes.energy,
      this.attributes.health,
      this.attributes.facedCreatures.length > 0 ? 1 : 0,
      ...flattenDeep(
        this.attributes.viewBeams.map((beam) => [
          beam.distance,
          beam.creature ? 1 : 0,
          beam.creature?.isPredator ? 1 : 0,
        ])
      ),
    ];
  }

  static createBrain(): tf.Sequential {
    const numInputNeurons = Creature.BRAIN_INPUT_NEURONS;
    const numOutputNeurons = 3;
    // Define the neural network architecture
    const model = tf.sequential();

    model.add(
      tf.layers.dense({
        inputShape: [numInputNeurons],
        units: 64,
        activation: 'relu',
      })
    );
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 8, activation: 'relu' }));
    // 'softmax': all output neurons have total value of 1. used for classification.
    // 'sigmoid': all output neurons are individually between 0 and 1. used for regression.
    model.add(
      tf.layers.dense({ units: numOutputNeurons, activation: 'sigmoid' })
    );

    // Compile the model
    model.compile({
      optimizer: 'adam',
      loss: 'meanSquaredError',
      metrics: ['accuracy'],
    });

    return model;
  }

  static cloneBrain(parent: Creature) {
    // Serialize the weights
    const weights = parent.brain.getWeights().map((weight) => weight.clone());

    // Create a new brain with the same architecture
    const newBrain = Creature.createBrain();

    // Set the new brain's weights
    newBrain.setWeights(weights);

    return newBrain;
  }

  static reshapeState(state: CreatureState): tf.Tensor {
    // Create a tensor representing the creature's state
    // Convert state to input tensor
    // Reshape the tensor to have a shape of [1, input_neurons]
    return tf.tensor1d(state).reshape([1, Creature.BRAIN_INPUT_NEURONS]);
  }

  makeDecision(state: CreatureState): CreatureActions | undefined {
    const decision = tf.tidy(() => {
      // Process the input and make a decision
      const reshapedState = Creature.reshapeState(state);

      if (this.isDead) {
        // failsafe: do not predict if brain is dead
        return undefined;
      }
      const prediction = this.brain.predict(reshapedState) as tf.Tensor;

      return prediction.dataSync();
    });

    if (!decision) {
      return undefined;
    }

    // Interpret decision to control creature's actions
    const [rawAcceleration, rawRotation, rawEat] = decision;

    // Scale acceleration from [0, 1] to [-0.4, 1.6] px at ACCELERATION_FACTOR = 2
    const acceleration = (rawAcceleration - 0.2) * Creature.ACCELERATION_FACTOR;

    // Scale rotation from [0, 1] to [-15, 15] deg at MOVEMENT_ANGLE_FACTOR = 30
    const rotation =
      rawRotation ** 2 *
      Math.sign(rawRotation) *
      Creature.MOVEMENT_ANGLE_FACTOR;

    const eat = rawEat >= 0.5;

    return { acceleration, rotation, eat };
  }

  recordExperience(
    state: CreatureState,
    actions: CreatureActions,
    nextState: CreatureState,
    rewards: CreatureRewards
  ) {
    this.experiences.addExperience(state, actions, nextState, rewards);
  }

  static defaultActions(actions?: Partial<CreatureActions>): CreatureActions {
    return {
      acceleration: 0,
      rotation: 0,
      eat: false,
      ...actions,
    };
  }

  static convertActionsToNumberArray(actions: CreatureActions): number[] {
    return [actions.acceleration, actions.rotation, actions.eat ? 1 : 0];
  }

  static defaultRewards(rewards?: Partial<CreatureRewards>): CreatureRewards {
    return {
      eating: 0,
      idleOrInefficientMovement: 0,
      exhaustingMovement: 0,
      movementTowardsFood: 0,
      movementTowardsPredator: 0,
      movementTowardsSafeArea: 0,
      movementDirectionalChange: 0,
      colliding: 0,
      ...rewards,
    };
  }

  static convertRewardsToNumberArray(actions: CreatureRewards): number[] {
    return [
      actions.eating,
      actions.idleOrInefficientMovement,
      actions.exhaustingMovement,
      actions.movementTowardsFood,
      actions.movementTowardsPredator,
      actions.movementTowardsSafeArea,
      actions.movementDirectionalChange,
      actions.colliding,
    ];
  }

  async learnFromExperience() {
    if (this.experiences.size() > Creature.MIN_EXPERIENCE_BATCH_SIZE) {
      if (this.environment.isTraining) {
        console.log(
          'Training is already in progress. Skipping new training call.'
        );
        return;
      }
      this.environment.isTraining = true;
      // console.log('TRAINING', this.id);

      this.averageReward = this.experiences.calculateAverageReward();
      this.averageRewards = this.experiences.calculateAverageRewards();

      // Randomly sample a batch of experiences
      const experiences = this.experiences.sample(
        Creature.MIN_EXPERIENCE_BATCH_SIZE
      );

      // Extract states, actions, and rewards from the batch
      const states = experiences.map((exp) => exp.state);
      const actions = experiences
        .map((exp) => exp.actions)
        .map(Creature.convertActionsToNumberArray);
      const rewards = experiences
        .map((exp) => exp.rewards)
        .map(Creature.convertRewardsToNumberArray);

      // Use Approach 2 if you're willing to invest more time into a complex implementation and
      // need a more sophisticated model that can learn the intricate relationships between actions,
      // states, and rewards. This approach is more suited to scenarios where the subtleties of
      // actions and their outcomes are crucial.
      // Custom training loop
      try {
        for (let epoch = 0; epoch < 1; epoch++) {
          // eslint-disable-next-line no-await-in-loop
          await tf.tidy(() => {
            const statesTensor = tf.tensor2d(states);
            const actionsTensor = tf.tensor2d(actions);
            const rewardsTensor = tf.tensor2d(rewards);
            const nextStatesTensor = tf.tensor2d(
              experiences.map((e) => e.nextState)
            );

            if (!this.optimizer) {
              this.optimizer = tf.train.adam();
            }
            this.optimizer.minimize(() => {
              const predictedActions = this.brain.predict(
                statesTensor
              ) as tf.Tensor<tf.Rank>;
              const predictedNextActions = this.brain.predict(
                nextStatesTensor
              ) as tf.Tensor<tf.Rank>;
              const loss = Creature.computeLoss(
                predictedActions,
                actionsTensor,
                rewardsTensor,
                predictedNextActions,
                0.9 // Example discount factor
              );
              return loss;
            });

            // Dispose tensors
            statesTensor.dispose();
            actionsTensor.dispose();
            rewardsTensor.dispose();
            nextStatesTensor.dispose();
          });
        }
        this.trainings++;
        // this.experiences.clear();
      } catch (error: unknown) {
        if (
          this.isDead &&
          error instanceof Error &&
          error.message.match(/Layer '\w+' is already disposed./i)
        ) {
          // do nothing as creature is dead
        } else {
          console.error(`Error during training:`, error);
        }
      } finally {
        this.environment.isTraining = false;
      }

      // Use Approach 1 if you're looking for a simpler implementation and are okay
      // with a potentially less nuanced understanding of actions and rewards. This
      // is a good starting point if you're new to machine learning or reinforcement learning.
      // // Train the model using states, actions, and rewards
      // const statesTensor = tf.tensor2d(states);
      // const actionsTensor = tf.tensor2d(actions);
      // try {
      //   await this.brain.fit(statesTensor, actionsTensor, {
      //     epochs: Creature.TRAINING_EPOCHS, // Number of iterations over the entire dataset
      //     batchSize: 32, // Number of samples per gradient update
      //   });
      //   this.trainings++;
      // } catch (error) {
      //   console.error('Error during training:', error);
      // } finally {
      //   // Dispose of tensors to free memory
      //   statesTensor.dispose();
      //   actionsTensor.dispose();
      //   this.environment.isTraining = false;
      // }
    }
  }

  // Example loss function
  static computeLoss(
    predictedActions: tf.Tensor<tf.Rank>,
    actualActions: tf.Tensor2D,
    rewards: tf.Tensor2D,
    predictedNextActions: tf.Tensor<tf.Rank>,
    discountFactor: number
  ): Scalar {
    // Assuming rewards is a 2D tensor where each row represents multiple reward components for a single example,
    // we sum these components to get a single reward value per example.
    const aggregatedRewards = rewards.sum(1, true); // Sum across the second dimension and keep the same rank.

    // Calculate the future discounted reward
    const futureRewards = predictedNextActions.mul(tf.scalar(discountFactor));

    // Incorporate the future rewards into the current (aggregated) rewards
    const totalRewards = aggregatedRewards.add(futureRewards);

    // Calculating the loss
    const loss = tf
      .square(tf.sub(predictedActions, actualActions))
      .mul(totalRewards);

    // Calculate the mean over all dimensions and coerce it to a scalar
    const meanLoss = tf.mean(loss).asScalar();

    return meanLoss;
  }

  // Method to move the creature based on decision
  behaveBasedOnDecision(actions: CreatureActions): Partial<CreatureRewards> {
    let rewards: Partial<CreatureRewards> = {};
    if (actions) {
      const { acceleration, rotation, eat } = actions;

      rewards = { ...rewards, ...this.move(acceleration, rotation) };

      if (this.isPredator && eat) {
        const facedPrey = this.attributes.facedCreatures
          .filter((c) => !c.isPredator)
          .pop();
        if (facedPrey) {
          rewards = { ...rewards, ...this.eat(facedPrey) };
        }
      }
    }

    return rewards;
  }

  /**
   * Moves the creature based on the provided acceleration and rotation.
   * Returns the partial rewards earned during the movement.
   *
   * @param {number} acceleration - The acceleration value.
   * @param {number} rotation - The rotation value.
   * @return {*}  {Partial<CreatureRewards>} The partial rewards earned during the movement.
   * @memberof Creature
   */
  move(acceleration: number, rotation: number): Partial<CreatureRewards> {
    let rewards: Partial<CreatureRewards> = {};

    // calculate new speed from acceleration
    const speed = Math.min(
      Math.max(acceleration + this.attributes.speed, -1),
      Creature.MAX_SPEED
    );

    // calculate new angle from rotation
    const angle = normalizeAngle(rotation + this.attributes.angle);
    const radians = degreesToRadians(angle);

    // Calculate the next position
    const deltaX = speed * Math.cos(radians);
    const deltaY = speed * Math.sin(radians);
    let newX = this.attributes.x + deltaX;
    let newY = this.attributes.y + deltaY;

    // * CHECK movement

    // Check if the new position is within the boundaries
    let collisionWithBoundaries = false;
    if (newX < this.attributes.size) {
      newX = this.attributes.size;
      collisionWithBoundaries = true;
    }
    if (newY < this.attributes.size) {
      newY = this.attributes.size;
      collisionWithBoundaries = true;
    }
    if (newX > this.environment.MAP_WIDTH - this.attributes.size) {
      newX = this.environment.MAP_WIDTH - this.attributes.size;
      collisionWithBoundaries = true;
    }
    if (newY > this.environment.MAP_HEIGHT - this.attributes.size) {
      newY = this.environment.MAP_HEIGHT - this.attributes.size;
      collisionWithBoundaries = true;
    }

    // check collision with other creature
    if (!Creature.COLLISION_SWITCH) {
      const collisionWithCreature = this.wouldTouchOnPosition({
        x: newX,
        y: newY,
      });
      if (collisionWithCreature) {
        newX = this.attributes.x + Math.cos(radians) * (speed > 0 ? -1 : 1);
        newY = this.attributes.y + Math.sin(radians) * (speed > 0 ? -1 : 1);
      }
    }

    // check where moving towards
    const viewBeamInMovementDirection =
      speed > 0
        ? this.attributes.viewBeams
            .map((beam, index) => ({
              ...beam,
              index,
              angle: this.calculateBeamAngleForIndex(index),
            }))
            .find((beam) =>
              inRange(
                angle,
                beam.angle - Creature.calculateBeamWidth() / 2,
                beam.angle + Creature.calculateBeamWidth() / 2
              )
            )
        : undefined;

    // * ENERGY consumption

    // if the creature moves or turns, it looses energy
    this.attributes.energy -=
      Math.abs(speed) * Creature.LOOSE_ENERGY_ON_MOVE_PER_SPEED +
      Math.abs(rotation) * Creature.LOOSE_ENERGY_ON_ROTATE_PER_ANGLE;

    // * REWARDS

    // negative REWARD for colliding with boundaries
    rewards = {
      ...rewards,
      colliding: collisionWithBoundaries
        ? Creature.REWARD_COLLIDING_WITH_BOUNDARIES
        : 0,
    };

    // negative REWARD for exhausting itself
    rewards = {
      ...rewards,
      exhaustingMovement:
        Math.abs(speed) * Creature.REWARD_SPEED_FACTOR +
        Math.abs(rotation) * Creature.REWARD_ROTATION_FACTOR,
    };

    // positive REWARD for directional change
    rewards = {
      ...rewards,
      movementDirectionalChange:
        this.attributes.rotation * rotation < 0
          ? Creature.REWARD_DIRECTIONAL_CHANGE
          : 0,
    };

    // positive REWARD for moving towards food. the faster or closer, the higher the reward.
    if (
      this.isPredator &&
      viewBeamInMovementDirection?.creature?.isPredator === false
    ) {
      rewards = {
        ...rewards,
        movementTowardsFood:
          (Creature.VIEW_DISTANCE - viewBeamInMovementDirection.distance) *
          speed *
          Creature.REWARD_PREDATOR_MOVE_TOWARDS_FOOD_FACTOR,
      };
    }

    if (!this.isPredator) {
      // small positive REWARD for moving towards safe areas. the faster, the higher the reward.
      if (viewBeamInMovementDirection?.creature === undefined) {
        rewards = {
          ...rewards,
          movementTowardsSafeArea:
            speed * Creature.REWARD_PREY_MOVE_TOWARDS_SAFE_AREA_FACTOR,
        };
      }

      // negative REWARD for moving towards predators. the faster, the higher the penalty.
      if (viewBeamInMovementDirection?.creature?.isPredator) {
        rewards = {
          ...rewards,
          movementTowardsPredator:
            speed * Creature.REWARD_PREY_MOVE_TOWARDS_PREDATOR_FACTOR,
        };
      }
    }

    // * UPDATE POSITION

    this.attributes.x = newX;
    this.attributes.y = newY;
    this.attributes.angle = angle;
    this.attributes.rotation = rotation;
    this.attributes.speed = speed;
    this.attributes.acceleration = acceleration;

    this.checkTouching();
    this.checkFacing();

    return rewards;
  }

  // * CHECKS

  checkLifetime() {
    this.attributes.lifetime =
      (new Date().getTime() - this.createdAt.getTime()) / 1000;
  }

  checkEnergy() {
    this.attributes.energy += Creature.GAIN_ENERGY;
    if (this.isPredator) {
      this.attributes.energy -= Creature.LOOSE_ENERGY_PREDATOR;
    } else {
      this.attributes.energy += Creature.GAIN_ENERGY_PREY;
      this.attributes.energy -= Creature.LOOSE_ENERGY_PREY;
    }
    if (this.attributes.energy <= 0) {
      this.attributes.health -= Creature.LOOSE_HEALTH_WHEN_WEAK;
      this.attributes.energy = 0;
    }
  }

  checkHealth() {
    if (this.attributes.health <= 0) {
      this.die();
    }
  }

  checkReproduction() {
    if (
      this.attributes.health >= 50 &&
      this.attributes.energy >= 80 &&
      this.attributes.lifetime > 10 &&
      this.attributes.reproductionCooldown <= 0
    ) {
      this.reproduce();
      this.attributes.reproductionCooldown =
        Creature.REPRODUCTION_COOLDOWN_INITIAL;
    } else {
      this.attributes.reproductionCooldown -=
        (this.attributes.health / 100) *
        (this.isPredator
          ? Creature.REPRODUCTION_COOLDOWN_FACTOR_PREDATOR
          : Creature.REPRODUCTION_COOLDOWN_FACTOR_PREY);
    }
  }

  checkTouching() {
    this.attributes.touchedCreatures = this.environment.creatures.filter(
      (creature) =>
        !creature.isDead &&
        creature.id !== this.id &&
        CreatureInteraction.areCreaturesTouching(this, creature)
    );
  }

  wouldTouchOnPosition(position: Position): boolean {
    return (
      this.environment.creatures.filter(
        (creature) =>
          !creature.isDead &&
          creature.id !== this.id &&
          CreatureInteraction.areCreaturesTouching(this, creature, position)
      ).length > 0
    );
  }

  checkFacing() {
    this.attributes.facedCreatures = this.attributes.touchedCreatures.filter(
      (creature) => CreatureInteraction.isFacingTarget(this, creature)
    );
  }

  // * ACTIONS

  /**
   * Makes the creature eat another creature.
   *
   * @param creature The creature to be eaten.
   * @returns {*}  {Partial<CreatureRewards>} An object containing the rewards gained from eating.
   */
  eat(creature: Creature): Partial<CreatureRewards> {
    creature.kill();
    this.attributes.energy += Creature.GAIN_ENERGY_ON_EAT;
    this.attributes.health += Creature.GAIN_HEALTH_ON_EAT;
    this.attributes.reproductionCooldown -=
      Creature.REPRODUCTION_COOLDOWN_ON_EAT;

    // console.log(`Creature ${this.id} ate creature ${creature.id}`);
    return {
      eating: Creature.GAIN_HEALTH_ON_EAT * Creature.REWARD_ON_EAT_FACTOR,
    };
  }

  reproduce() {
    if (Creature.REPRODUCTION_SWITCH !== 0) {
      this.environment.createCreature(
        { x: this.attributes.x, y: this.attributes.y },
        this.isPredator,
        this
      );
    }
  }

  die() {
    if (this.isDead) {
      return;
    }
    if (Creature.KILL_SWITCH !== 0) {
      // Remove creature from simulation
      this.attributes.health = 0;
      this.isDead = true;
      this.brain.dispose();
    }
  }

  kill() {
    this.die();
  }

  // * HELPERS for VIEW

  determineClosestCreatureInViewBeams() {
    this.attributes.viewBeams = times(Creature.VIEW_BEAMS, (i) => ({
      distance: Creature.VIEW_DISTANCE,
    }));

    this.attributes.viewBeams.forEach((beam, index) => {
      this.environment.creatures
        .filter((creature) => !creature.isDead && creature.id !== this.id)
        .forEach((creature) => {
          const distanceToCreature =
            CreatureInteraction.distanceBetweenCreatures(this, creature);

          if (
            distanceToCreature < beam.distance &&
            this.isIntersectingWithBeamIndex(
              creature,
              distanceToCreature,
              index
            )
          ) {
            this.attributes.viewBeams[index] = {
              creature,
              distance: distanceToCreature,
            };
          }
        });
    });
  }

  isIntersectingWithBeamIndex(
    creature: Creature,
    distanceToCreature: number,
    beamIndex: number
  ) {
    const beamAngle = this.calculateBeamAngleForIndex(beamIndex); // in degrees

    const angleToCreature = CreatureInteraction.angleToCreature(this, creature);
    const angularSize = radiansToDegrees(
      Math.atan2(creature.attributes.size, distanceToCreature)
    );

    const startAngleWithSize = beamAngle - angularSize;
    const endAngleWithSize = beamAngle + angularSize;

    return (
      angleToCreature >= startAngleWithSize &&
      angleToCreature <= endAngleWithSize
    );
  }

  static calculateBeamWidth(): number {
    return Creature.VIEW_ANGLE / (Creature.VIEW_BEAMS - 1); // in degrees
  }

  calculateBeamAngleForIndex(beamIndex: number): number {
    return (
      this.attributes.angle -
      Creature.VIEW_ANGLE / 2 +
      Creature.calculateBeamWidth() * beamIndex
    ); // in degrees
  }

  // * STATISTICS

  getStatistics(): CreatureStatistics {
    return {
      ...this.attributes,
      ...pick(this, [
        'id',
        'generation',
        'parent',
        'isPredator',
        'isDead',
        'trainings',
        'averageReward',
        'averageRewards',
        'currentRewards',
        'debug',
      ]),
    };
  }
}
