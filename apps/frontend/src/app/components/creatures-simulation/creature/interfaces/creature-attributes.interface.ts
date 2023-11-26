import { Creature } from '../classes/creature.class';
import { ViewBeam } from './view-beam.interface';

export interface CreatureAttributes {
  /**
   * The size (radius) of the creature in pixels.
   */
  size: number;
  color: number;
  speed: number;
  acceleration: number;
  /**
   * The angle in degrees.
   */
  angle: number;
  /**
   * The rotation in degrees.
   */
  rotation: number;
  x: number;
  y: number;
  energy: number;
  health: number;
  lifetime: number;
  reproductionCooldown: number;
  touchedCreatures: Creature[];
  facedCreatures: Creature[];
  viewBeams: ViewBeam[];
}
