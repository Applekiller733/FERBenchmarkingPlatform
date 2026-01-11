import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, map } from 'rxjs';
import {
  ModelStatsResponse,
  ComparisonResponse,
  RawModelStats
} from './comparison_models';
import { parseReportContent } from './fer_parser';

@Injectable({
  providedIn: 'root'
})
export class ComparisonService {

  /**
   * Base API URL
   * Adjust if your backend runs on a different host/port
   */
  private readonly baseUrl = 'http://localhost:8000/stats';

  constructor(private http: HttpClient) {}

  /**
   * Fetch stats for the custom-trained model
   */
  getCustomModelStats(): Observable<ModelStatsResponse> {
    return this.http.get<RawModelStats>(
      `${this.baseUrl}/custom`
    ).pipe(
        map(res =>
          parseReportContent(res.model, res.report_content!, res.plots) 
        )
      );
  }

  /**
   * Fetch stats for the pretrained model
   */
  getPretrainedModelStats(): Observable<ModelStatsResponse> {
    return this.http.get<RawModelStats>(
      `${this.baseUrl}/pretrained`
    ).pipe(
        map(res =>
          parseReportContent(res.model, res.report_content!, res.plots) 
        )
      );
  }

  /**
   * Fetch comparison-level stats (optional)
   */
  getComparisonStats(): Observable<ComparisonResponse> {
    return this.http.get<ComparisonResponse>(
      `${this.baseUrl}/comparison`
    );
  }
}
