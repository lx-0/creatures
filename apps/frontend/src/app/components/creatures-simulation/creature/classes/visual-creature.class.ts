import { times } from 'lodash';
import * as PIXI from 'pixi.js';
import { degreesToRadians } from '../../../../helpers';
import { Environment } from '../../environment.class';
import { Creature } from './creature.class';

export class VisualCreature {
  private environment: Environment;
  private creature: Creature;

  color: number;
  baseColor: number;

  body!: PIXI.Sprite;
  leftEye!: PIXI.Graphics;
  rightEye!: PIXI.Graphics;

  viewBeams: PIXI.Graphics[] = [];
  facingLine: PIXI.Graphics | undefined;
  facingLine2: PIXI.Graphics | undefined;

  private static SHOW_VIEW_ANGLE = false;
  private static SHOW_VIEW_BEAMS = false;
  private static VIEW_ALPHA = 0.3;
  private static SHOW_TOUCHES = true;
  private static SHOW_FACES = false;
  private static DEAD_ALPHA = 0.3;

  constructor(creature: Creature, environment: Environment) {
    this.creature = creature;
    this.environment = environment;

    this.baseColor = creature.attributes.color;
    this.color = this.baseColor;

    if (VisualCreature.SHOW_VIEW_ANGLE) {
      // Draw the facing line
      this.facingLine = new PIXI.Graphics();
      this.facingLine2 = new PIXI.Graphics();
      this.environment.app.stage.addChild(this.facingLine);
      this.environment.app.stage.addChild(this.facingLine2);
    }

    if (VisualCreature.SHOW_VIEW_BEAMS) {
      // Draw the facing line
      this.viewBeams = times(Creature.VIEW_BEAMS, () => new PIXI.Graphics());
      this.viewBeams.forEach((beam) => {
        this.environment.app.stage.addChild(beam);
      });
    }

    this.create();
  }

  toFront() {
    this.environment.app.stage.setChildIndex(
      this.body,
      this.environment.app.stage.children.length - 1
    );
    this.environment.app.stage.setChildIndex(
      this.leftEye,
      this.environment.app.stage.children.length - 1
    );
    this.environment.app.stage.setChildIndex(
      this.rightEye,
      this.environment.app.stage.children.length - 1
    );
  }

  toBack() {
    this.environment.app.stage.setChildIndex(this.body, 0);
    this.environment.app.stage.setChildIndex(this.leftEye, 0);
    this.environment.app.stage.setChildIndex(this.rightEye, 0);
  }

  determineColor(isSelected?: boolean): number {
    if (this.creature.isDead) {
      return 0x666666;
    }
    if (isSelected) {
      return 0xff33ff;
    }
    if (
      VisualCreature.SHOW_FACES &&
      this.creature.attributes.facedCreatures.length > 0
    ) {
      return 0x0033ff;
    }
    if (
      VisualCreature.SHOW_TOUCHES &&
      this.creature.attributes.touchedCreatures.length > 0
    ) {
      return 0xffcc00;
    }

    return this.baseColor;
  }

  create() {
    const color = this.determineColor();

    // Update body position
    this.drawBody(color);

    // Create eyes
    this.leftEye = new PIXI.Graphics();
    this.rightEye = new PIXI.Graphics();

    // Update eye positions
    this.drawEyes();

    // Add the body and eyes to the PIXI application
    this.environment.app.stage.addChild(this.leftEye);
    this.environment.app.stage.addChild(this.rightEye);

    if (VisualCreature.SHOW_VIEW_ANGLE) {
      this.drawFacingLine();
    }

    if (VisualCreature.SHOW_VIEW_BEAMS) {
      this.drawViewBeams();
    }
  }

  update(isSelected: boolean) {
    const color = this.determineColor(isSelected);

    if (this.color !== color) {
      this.drawBody(color);
      this.color = color;
      this.toFront();
    }
    // Update body position
    this.moveBody();

    // Update eye positions
    this.moveEyes();

    if (VisualCreature.SHOW_VIEW_ANGLE) {
      this.drawFacingLine();
    }

    if (VisualCreature.SHOW_VIEW_BEAMS) {
      this.drawViewBeams();
    }

    if (this.creature.isDead) {
      this.toBack();
    }
    if (isSelected) {
      this.toFront();
    }
  }

  drawBody(color: number) {
    const body = new PIXI.Graphics();

    body.clear();
    body.beginFill(color, this.creature.isDead ? VisualCreature.DEAD_ALPHA : 1);
    body.drawCircle(
      this.creature.attributes.x,
      this.creature.attributes.y,
      this.creature.attributes.size
    );
    body.endFill();

    const texture = this.environment.app.renderer.generateTexture(body);
    const sprite = new PIXI.Sprite(texture);
    sprite.eventMode = 'static';
    sprite.on('pointerdown', (_event) => {
      this.environment.selectCreature(this.creature.id);
    });

    if (this.body) {
      this.environment.app.stage.removeChild(this.body);
    }
    this.body = sprite;
    this.environment.app.stage.addChild(this.body);
  }

  moveBody() {
    this.body.x = this.creature.attributes.x - this.creature.attributes.size;
    this.body.y = this.creature.attributes.y - this.creature.attributes.size;
  }

  drawEyes() {
    const eyeSize = this.creature.attributes.size / 4; // Adjust size as needed

    // Position and draw left eye
    this.leftEye.clear();
    this.leftEye.beginFill(
      0xffffff,
      this.creature.isDead ? VisualCreature.DEAD_ALPHA : 1
    );
    this.leftEye.drawCircle(0, 0, eyeSize);
    this.leftEye.endFill();

    // Position and draw right eye
    this.rightEye.clear();
    this.rightEye.beginFill(
      0xffffff,
      this.creature.isDead ? VisualCreature.DEAD_ALPHA : 1
    );
    this.rightEye.drawCircle(0, 0, eyeSize);
    this.rightEye.endFill();
  }

  moveEyes() {
    const radians = degreesToRadians(this.creature.attributes.angle);

    const eyeSize = this.creature.attributes.size / 4; // Adjust size as needed
    const eyeOffsetX = Math.cos(radians) * (this.creature.attributes.size / 2);
    const eyeOffsetY = Math.sin(radians) * (this.creature.attributes.size / 2);

    // Position and draw left eye
    this.leftEye.x = this.creature.attributes.x + eyeOffsetX - eyeSize;
    this.leftEye.y = this.creature.attributes.y + eyeOffsetY;

    // Position and draw right eye
    this.rightEye.x = this.creature.attributes.x + eyeOffsetX + eyeSize;
    this.rightEye.y = this.creature.attributes.y + eyeOffsetY;
  }

  drawFacingLine() {
    if (!this.facingLine || !this.facingLine2) {
      return;
    }
    this.facingLine.clear();
    this.facingLine2.clear();

    if (this.creature.isDead) {
      return;
    }

    // Convert angle to radians
    const angleRadians = degreesToRadians(this.creature.attributes.angle);

    // Calculate the end point of the line
    const lineLength = Creature.VIEW_DISTANCE;
    const halfViewAngle = degreesToRadians(Creature.VIEW_ANGLE / 2);
    const endX =
      this.creature.attributes.x +
      lineLength * Math.cos(angleRadians - halfViewAngle);
    const endY =
      this.creature.attributes.y +
      lineLength * Math.sin(angleRadians - halfViewAngle);
    const endX2 =
      this.creature.attributes.x +
      lineLength * Math.cos(angleRadians + halfViewAngle);
    const endY2 =
      this.creature.attributes.y +
      lineLength * Math.sin(angleRadians + halfViewAngle);

    // Draw the line
    this.facingLine.lineStyle(1, 0xff00ff, VisualCreature.VIEW_ALPHA);
    this.facingLine2.lineStyle(1, 0xff00ff, VisualCreature.VIEW_ALPHA);
    this.facingLine.moveTo(
      this.creature.attributes.x,
      this.creature.attributes.y
    );
    this.facingLine2.moveTo(
      this.creature.attributes.x,
      this.creature.attributes.y
    );
    this.facingLine.lineTo(endX, endY);
    this.facingLine2.lineTo(endX2, endY2);
  }

  drawViewBeams() {
    const beamWidth = Creature.VIEW_ANGLE / (Creature.VIEW_BEAMS - 1);

    this.viewBeams.forEach((beam, index) => {
      beam.clear();

      if (this.creature.isDead) {
        return;
      }

      const angleRadians = degreesToRadians(
        this.creature.attributes.angle -
          Creature.VIEW_ANGLE / 2 +
          beamWidth * index
      );
      const lineLength = Creature.VIEW_DISTANCE;
      const endX =
        this.creature.attributes.x + lineLength * Math.cos(angleRadians);
      const endY =
        this.creature.attributes.y + lineLength * Math.sin(angleRadians);

      if (this.creature.attributes.viewBeams[index].creature) {
        beam.lineStyle(1, 0xff0000, VisualCreature.VIEW_ALPHA);
      } else {
        beam.lineStyle(1, 0xffffff, VisualCreature.VIEW_ALPHA);
      }
      beam.moveTo(this.creature.attributes.x, this.creature.attributes.y);
      beam.lineTo(endX, endY);
    });
  }

  // private createCreatureWithSprite(
  //   x: number,
  //   y: number,
  //   texture: PIXI.Texture
  // ): void {
  //   const sprite = new PIXI.Sprite(texture);
  //   sprite.x = x;
  //   sprite.y = y;
  //   sprite.anchor.set(0.5); // Center anchor if needed
  //   this.app.stage.addChild(sprite);
  // }
}
