import { CreatureState } from '../types';
import { CreatureActions } from './creature-actions.interface';
import { CreatureRewards } from './creature-rewards.interface';

export interface CreatureExperience {
  state: CreatureState;
  actions: CreatureActions;
  nextState: CreatureState;
  rewards: CreatureRewards;
  reward: number;
}
