import { Component } from '@angular/core';
import { ComparisonService } from '../../../core/comparison_service';
import { ModelStatsResponse } from '../../../core/comparison_models';
import { CommonModule, DecimalPipe } from '@angular/common';

@Component({
  selector: 'app-comparison',
  standalone: true,
  imports: [CommonModule, DecimalPipe],
  templateUrl: './comparisonpage.html',
  styleUrls: ['./comparisonpage.scss']
})

export class ComparisonComponent {
  customStats: ModelStatsResponse | null = null;
  pretrainedStats: ModelStatsResponse | null = null;

  readonly serverUrl = 'http://localhost:8000';

  constructor(private comparisonService: ComparisonService) {
    this.loadModelStats();
  }

  compare(val1: number, val2: number): boolean {
    return val1 > val2;
  }

  getImgUrl(path: string | undefined): string {
    if (!path) return '';
    if (path.startsWith('http')) return path;
    return `${this.serverUrl}${path}`;
  }

  private loadModelStats(): void {
    this.comparisonService.getCustomModelStats().subscribe(stats => {
      this.customStats = stats;
    });

    this.comparisonService.getPretrainedModelStats().subscribe(stats => {
      this.pretrainedStats = stats;
    });
  }
}
