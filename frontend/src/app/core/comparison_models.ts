export interface ModelPlots {
    before_after_accuracy?: string;
    confusion_matrix: string;
    confusion_matrix_normalized: string;
    per_class_metrics: string;
    roc_curve: string;
  }

  export interface EmotionMetrics {
    emotion: string;
    precision: number;
    recall: number;
    f1: number;
    support: number;
  }

  export interface OverallMetrics {
    accuracy: number;
    macroAvg: Omit<EmotionMetrics, 'emotion'>;
    weightedAvg: Omit<EmotionMetrics, 'emotion'>;
  }
  
  export interface ModelStatsResponse {
    model: string;
    emotions: EmotionMetrics[];
    overall: OverallMetrics;
    plots: ModelPlots;
  }
  
  export interface ComparisonResponse {
    comparison_plot?: string;
    message: string;
  }

  export interface RawModelStats {
    model: string;
    report_content: string | null;
    plots: ModelPlots;
  }
  


  