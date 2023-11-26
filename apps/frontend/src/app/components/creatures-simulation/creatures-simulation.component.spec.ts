import { ComponentFixture, TestBed } from '@angular/core/testing';
import { CreaturesSimulationComponent } from './creatures-simulation.component';

describe('CreaturesSimulationComponent', () => {
  let component: CreaturesSimulationComponent;
  let fixture: ComponentFixture<CreaturesSimulationComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CreaturesSimulationComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(CreaturesSimulationComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
