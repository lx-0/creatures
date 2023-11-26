import { round, sum } from 'lodash';
import {
  CreatureActions,
  CreatureExperience,
  CreatureRewards,
} from '../interfaces';
import { CreatureState } from '../types';
import { Creature } from './creature.class';

export class ExperienceBuffer {
  private buffer: Array<CreatureExperience> = [];

  private maxSize: number;

  constructor(maxSize: number = 1000) {
    this.maxSize = maxSize;
  }

  clear() {
    this.buffer = [];
  }

  addExperience(
    state: CreatureState,
    actions: CreatureActions,
    nextState: CreatureState,
    rewards: CreatureRewards
  ) {
    if (this.buffer.length >= this.maxSize) {
      this.buffer.shift(); // Remove the oldest experience if the buffer is full
    }
    this.buffer.push({
      state,
      actions,
      nextState,
      rewards,
      reward: sum(Object.values(rewards)),
    });
  }

  sample(batchSize: number): Array<CreatureExperience> {
    // Calculate the total sum of rewards for weighting
    const totalReward = this.buffer.reduce((acc, exp) => acc + exp.reward, 0);
    const weightedSamples = this.buffer.map((exp) => ({
      ...exp,
      probability: exp.reward / totalReward,
    }));

    // Sample based on the probability weights
    const sampledExperiences = [];
    for (let i = 0; i < batchSize; i++) {
      const randomNum = Math.random();
      let probabilitySum = 0;
      // eslint-disable-next-line no-restricted-syntax
      for (const exp of weightedSamples) {
        probabilitySum += exp.probability;
        if (randomNum < probabilitySum) {
          sampledExperiences.push(exp);
          break;
        }
      }
    }
    return sampledExperiences;
  }

  size(): number {
    return this.buffer.length;
  }

  calculateAverageReward(): number {
    return (
      this.buffer.reduce((acc, exp) => acc + exp.reward, 0) / this.buffer.length
    );
  }

  calculateAverageRewards(): CreatureRewards {
    const totalRewards = this.buffer.reduce((acc, exp) => {
      Object.entries(exp.rewards).forEach(([key, value]) => {
        acc[key as keyof CreatureRewards] += value;
      });
      return acc;
    }, Creature.defaultRewards());
    return {
      ...totalRewards,
      ...Object.fromEntries(
        Object.entries(totalRewards).map(([key, value]) => [
          key,
          round(value / this.buffer.length, 2),
        ])
      ),
    };
  }
}
