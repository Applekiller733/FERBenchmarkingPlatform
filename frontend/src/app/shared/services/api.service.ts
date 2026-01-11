import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface PredictionResult {
  filename: string;
  result: {
    top_emotion: string;
    confidence: number;
    predictions: { [key: string]: number };
  } | null;
}

export interface PretrainedPredictionResult {
  count: number;
  results: {
    filename: string;
    top_emotion: string;
    confidence: number;
    full_analysis: { [key: string]: number };
    error?: string;
  }[];
}

export interface StatsResult {
  model: string;
  report_content: string | null;
  plots: { [key: string]: string };
}

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private http = inject(HttpClient);
  private baseUrl = 'http://localhost:8000';

  constructor() { }

  predictCustom(file: File): Observable<PredictionResult> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post<PredictionResult>(`${this.baseUrl}/custom/predict_custom`, formData);
  }

  predictPretrained(file: File): Observable<PretrainedPredictionResult> {
    const formData = new FormData();
    formData.append('images', file); // API expects 'images' list but works with single file upload usually if key matches
    return this.http.post<PretrainedPredictionResult>(`${this.baseUrl}/pretrained/predict_pretrained`, formData);
  }

  getCustomStats(): Observable<StatsResult> {
    return this.http.get<StatsResult>(`${this.baseUrl}/stats/custom`);
  }

  getPretrainedStats(): Observable<StatsResult> {
    return this.http.get<StatsResult>(`${this.baseUrl}/stats/pretrained`);
  }
}
