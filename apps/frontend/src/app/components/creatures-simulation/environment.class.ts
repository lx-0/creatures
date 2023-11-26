import { EventEmitter } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { random } from 'lodash';
import * as PIXI from 'pixi.js';
import { Creature, CreatureStatistics } from './creature';

export interface EnvironmentStatistics {
  countCreatures: number;
  countPredatorCreatures: number;
  countPreyCreatures: number;
  countDeadCreatures: number;

  creatures: CreatureStatistics[];

  tfMemory: number;
  tfTensors: number;
  tfDataBuffers: number;
}

export interface Position {
  x: number;
  y: number;
}

export class Environment {
  public app: PIXI.Application;
  public creatures: Creature[] = [];

  public isTraining: boolean;
  lastTrainedAt: number;

  public selectedCreatureId: string | undefined;

  creatureSelected = new EventEmitter<string>();

  public static CREATURES_PREDATORS_INITIAL_COUNT: number = 20;
  public static CREATURES_PREY_INITIAL_COUNT: number = 20;
  public static CREATURE_INITIAL_SIZE: number = 10;
  public static CREATURE_TRAINING_INTERVAL: number = 100; // in milliseconds

  public MAP_WIDTH: number;
  public MAP_HEIGHT: number;

  constructor(app: PIXI.Application) {
    this.app = app;
    this.isTraining = false;
    this.lastTrainedAt = 0;
    this.MAP_WIDTH = app.renderer.width;
    this.MAP_HEIGHT = app.renderer.height;
  }

  initial() {
    // Initialize creatures
    for (let i = 0; i < Environment.CREATURES_PREDATORS_INITIAL_COUNT; i++) {
      this.createCreature(this.getFreeRandomPosition(), true);
    }
    for (let i = 0; i < Environment.CREATURES_PREY_INITIAL_COUNT; i++) {
      this.createCreature(this.getFreeRandomPosition(), false);
    }
  }

  getFreeRandomPosition(): Position {
    const x = random(
      Environment.CREATURE_INITIAL_SIZE,
      this.MAP_WIDTH - Environment.CREATURE_INITIAL_SIZE
    );
    const y = random(
      Environment.CREATURE_INITIAL_SIZE,
      this.MAP_HEIGHT - Environment.CREATURE_INITIAL_SIZE
    );
    return { x, y };
  }

  update() {
    this.creatures.forEach((creature) => {
      creature.update();
    });
    if (
      this.isTraining === false &&
      Date.now() - this.lastTrainedAt > Environment.CREATURE_TRAINING_INTERVAL
    ) {
      // * learn from experiences
      const livingCreatures = this.creatures.filter((c) => !c.isDead);
      const randomIndex = Math.floor(Math.random() * livingCreatures.length);
      this.creatures
        .find((c) => c.id === livingCreatures[randomIndex].id)
        ?.learnFromExperience();
      this.lastTrainedAt = Date.now();
    }
    // Update food sources, hazards, etc.
  }

  updateVisuals() {
    this.creatures.forEach((creature) => {
      creature.visualRepresentation.update(
        this.selectedCreatureId === creature.id
      );
    });
  }

  createCreature(
    position: Position,
    isPredator: boolean,
    parent?: Creature
  ): void {
    const creature = new Creature(
      {
        size: Environment.CREATURE_INITIAL_SIZE,
        speed: 5,
        angle: random(-180, 180),
        x: position.x,
        y: position.y,
        color: isPredator ? 0xff0000 : 0x009900,
        health: 100,
        energy: 100,
      },
      this,
      isPredator,
      parent
    );
    this.addCreature(creature);
  }

  addCreature(creature: Creature) {
    this.creatures.push(creature);
  }

  selectCreature(creatureId: string) {
    this.creatureSelected.emit(creatureId);
    this.selectedCreatureId = creatureId;
  }

  // * HELPERS for VIEW

  getStatistics(): EnvironmentStatistics {
    return {
      countCreatures: this.creatures.filter((c) => !c.isDead).length,
      countPredatorCreatures: this.creatures.filter(
        (c) => !c.isDead && c.isPredator
      ).length,
      countPreyCreatures: this.creatures.filter(
        (c) => !c.isDead && !c.isPredator
      ).length,
      countDeadCreatures: this.creatures.filter((c) => c.isDead).length,
      creatures: this.creatures.map((creature) => creature.getStatistics()),
      tfMemory: tf.memory().numBytes,
      tfTensors: tf.memory().numTensors,
      tfDataBuffers: tf.memory().numDataBuffers,
    };
  }
}
