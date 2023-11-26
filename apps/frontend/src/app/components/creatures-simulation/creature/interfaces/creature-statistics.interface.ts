import { Creature } from '../classes/creature.class';
import { CreatureAttributes } from './creature-attributes.interface';

export interface CreatureStatistics
  extends CreatureAttributes,
    Pick<
      Creature,
      | 'id'
      | 'generation'
      | 'parent'
      | 'isPredator'
      | 'isDead'
      | 'trainings'
      | 'averageReward'
      | 'averageRewards'
      | 'currentRewards'
      | 'debug'
    > {}
