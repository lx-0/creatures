import { Component, Input } from '@angular/core';
import { ScaleType } from '@swimlane/ngx-charts';

// Example data structure for a line chart
export interface ChartData {
  name: string;
  series: { name: string; value: number }[];
}

@Component({
  selector: 'frontend-chart',
  template: `
    <ngx-charts-line-chart
      [view]="view"
      [scheme]="colorScheme"
      [results]="lineChartData"
      [gradient]="gradient"
      [xAxis]="xAxis"
      [yAxis]="yAxis"
      [legend]="legend"
      [showXAxisLabel]="showXAxisLabel"
      [showYAxisLabel]="showYAxisLabel"
      [xAxisLabel]="xAxisLabel"
      [yAxisLabel]="yAxisLabel"
      [timeline]="timeline"
      [legend]="false"
      (select)="onSelect($event)"
    >
    </ngx-charts-line-chart>
  `,
})
export class ChartComponent {
  // In your component
  @Input() public lineChartData: ChartData[] = [
    // Initial data goes here
  ];

  view: [number, number] = [600, 150]; // Width and height of the chart

  // Chart options
  colorScheme = {
    name: 'custom',
    selectable: false,
    group: ScaleType.Ordinal,
    domain: [
      '#FF5733', // Red
      '#33FF57', // Green
    ],
  };
  gradient: boolean = false;
  legend: boolean = true;
  showLabels: boolean = true;
  animations: boolean = true;
  xAxis: boolean = false;
  yAxis: boolean = true;
  showYAxisLabel: boolean = true;
  showXAxisLabel: boolean = false;
  xAxisLabel: string = '';
  yAxisLabel: string = 'Creatures';
  timeline: boolean = true;
  autoScale: boolean = true;

  onSelect(event: unknown): void {
    console.log(event);
  }
}
