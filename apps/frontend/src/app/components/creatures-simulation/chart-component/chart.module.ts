import { NgModule } from '@angular/core';
import { NgxChartsModule } from '@swimlane/ngx-charts';
import { ChartComponent } from './chart.compontent';

@NgModule({
  imports: [NgxChartsModule],
  declarations: [ChartComponent],
  exports: [ChartComponent],
})
export class ChartModule {}
