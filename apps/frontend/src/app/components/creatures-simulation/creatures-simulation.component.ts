import { CommonModule } from '@angular/common';
import {
  AfterViewInit,
  Component,
  ElementRef,
  OnInit,
  ViewChild,
} from '@angular/core';
import { MatSort, MatSortModule } from '@angular/material/sort';
import { MatTableDataSource, MatTableModule } from '@angular/material/table';
import { MatTooltipModule } from '@angular/material/tooltip';
import { throttle } from 'lodash';
import * as PIXI from 'pixi.js';
import { ChartData, ChartModule } from './chart-component';
import { CreatureStatistics } from './creature';
import { Environment, EnvironmentStatistics } from './environment.class';

@Component({
  selector: 'frontend-creatures-simulation',
  standalone: true,
  imports: [
    CommonModule,
    ChartModule,
    MatTableModule,
    MatSortModule,
    MatTooltipModule,
  ],
  templateUrl: './creatures-simulation.component.html',
  styleUrls: ['./creatures-simulation.component.scss'],
})
export class CreaturesSimulationComponent implements OnInit, AfterViewInit {
  private app!: PIXI.Application;
  private environment!: Environment;
  statistics!: EnvironmentStatistics;
  chartData: {
    name: string;
    series: {
      name: string;
      value: number;
    }[];
  }[] = [];

  private static UPDATE_INTERVAL = 30; // Update every x milliseconds

  @ViewChild('creaturesCanvas', { static: true }) canvasContainer!: ElementRef;

  @ViewChild(MatSort) sort!: MatSort;
  dataSource: MatTableDataSource<CreatureStatistics>;

  public selectedCreatureId: string | undefined;

  protected displayedColumns: string[] = [
    'isPredator',
    'generation',
    'lifetime',
    'health',
    'energy',
    'reproductionCooldown',
    'speed',
    'angle',
    'trainings',
    'averageReward',
    'debug',
  ];

  constructor() {
    this.dataSource = new MatTableDataSource([] as CreatureStatistics[]);
  }

  ngOnInit(): void {
    this.initPixiApp();
    this.initializeEnvironment();
    this.startSimulationLoop();
  }

  ngAfterViewInit() {
    this.dataSource.sort = this.sort;
  }

  private initPixiApp(): void {
    this.app = new PIXI.Application({
      width: window.innerWidth - 480,
      height: window.innerHeight * 0.8,
    });

    if (this.canvasContainer) {
      this.canvasContainer.nativeElement.appendChild(this.app.view);
    } else {
      console.error('Canvas container not found');
    }
  }

  private initializeEnvironment(): void {
    this.environment = new Environment(this.app);
    this.environment.initial();

    this.environment.creatureSelected.subscribe((creatureId) => {
      this.selectedCreatureId = creatureId;
    });
  }

  private async updateEnvironment(): Promise<void> {
    this.environment.update();
  }

  selectCreature(creatureId: string) {
    this.environment.selectCreature(creatureId);
  }

  private startSimulationLoop(): void {
    // Update creature state based on TensorFlow.js model predictions
    this.app.ticker.add(
      throttle(
        this.updateEnvironment.bind(this),
        CreaturesSimulationComponent.UPDATE_INTERVAL
      )
    );
    this.app.ticker.add(throttle(this.updateStatistics.bind(this), 1000));
    this.app.ticker.add(throttle(this.updateChart.bind(this), 1000));

    this.app.ticker.add((delta) => {
      // Update and render your creatures in each tick
      this.environment.updateVisuals();
    });
  }

  // eslint-disable-next-line class-methods-use-this
  stringify(value: unknown): string {
    return JSON.stringify(value, null, 2);
  }

  private static getLastSeriesValues(
    currentSeries: ChartData['series']
  ): ChartData['series'] {
    const MAX_DATA_ENTRIES = 360;
    return currentSeries
      .slice(
        Math.max(currentSeries.length - MAX_DATA_ENTRIES - 1, 0),
        Math.max(currentSeries.length, 0)
      )
      .map((series, index) => ({
        name: (index + 1).toString(),
        value: series.value,
      }));
  }

  private updateStatistics(): void {
    this.statistics = this.environment.getStatistics();
    this.dataSource.data = this.statistics.creatures.filter((c) => !c.isDead);
    this.dataSource.sort = this.sort;
  }

  private updateChart(): void {
    const countPredatorsSeries =
      CreaturesSimulationComponent.getLastSeriesValues(
        this.chartData[0]?.series ?? []
      );
    const countPreysSeries = CreaturesSimulationComponent.getLastSeriesValues(
      this.chartData[1]?.series ?? []
    );
    this.chartData = [
      {
        name: 'Predators',
        series: [
          ...countPredatorsSeries,
          {
            name: (countPredatorsSeries.length + 1).toString(),
            value: this.statistics.countPredatorCreatures,
          },
        ],
      },
      {
        name: 'Preys',
        series: [
          ...countPreysSeries,
          {
            name: (countPreysSeries.length + 1).toString(),
            value: this.statistics.countPreyCreatures,
          },
        ],
      },
    ];
  }
}
