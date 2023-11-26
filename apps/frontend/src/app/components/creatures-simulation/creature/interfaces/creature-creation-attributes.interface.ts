import { CreatureAttributes } from './creature-attributes.interface';

export interface CreatureCreationAttributes
  extends Omit<
    CreatureAttributes,
    | 'acceleration'
    | 'rotation'
    | 'lifetime'
    | 'reproductionCooldown'
    | 'touchedCreatures'
    | 'wasTouchingBefore'
    | 'facedCreatures'
    | 'viewBeams'
  > {}
