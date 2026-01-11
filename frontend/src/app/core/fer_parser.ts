import { ModelStatsResponse, EmotionMetrics, OverallMetrics, ModelPlots } from './comparison_models';

export function parseReportContent(
  modelName: string,
  reportContent: string,
  plots: ModelPlots
): ModelStatsResponse {

  const lines = reportContent
    .split('\n')
    .map(l => l.trim())
    .filter(l => l.length > 0);

  const emotions: EmotionMetrics[] = [];
  let accuracy = 0;
  let macroAvg!: Omit<EmotionMetrics, 'emotion'>;
  let weightedAvg!: Omit<EmotionMetrics, 'emotion'>;

  for (const line of lines) {
    // Match per-class rows: emotion + 4 numbers
    const classMatch = line.match(
      /^([a-zA-Z]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)$/
    );
    if (classMatch) {
      emotions.push({
        emotion: classMatch[1],
        precision: parseFloat(classMatch[2]),
        recall: parseFloat(classMatch[3]),
        f1: parseFloat(classMatch[4]),
        support: parseInt(classMatch[5], 10)
      });
      continue;
    }

    // Accuracy row
    const accMatch = line.match(/^accuracy\s+([\d.]+)\s+(\d+)$/);
    if (accMatch) {
      accuracy = parseFloat(accMatch[1]);
      continue;
    }

    // Macro / weighted average rows
    const avgMatch = line.match(
      /^(macro avg|weighted avg)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)$/
    );
    if (avgMatch) {
      const avg = {
        precision: parseFloat(avgMatch[2]),
        recall: parseFloat(avgMatch[3]),
        f1: parseFloat(avgMatch[4]),
        support: parseInt(avgMatch[5], 10)
      };
      if (avgMatch[1] === 'macro avg') macroAvg = avg;
      else weightedAvg = avg;
    }
  }

  const overall: OverallMetrics = {
    accuracy,
    macroAvg,
    weightedAvg
  };

  return {
    model: modelName,
    emotions,
    overall,
    plots
  };
}
