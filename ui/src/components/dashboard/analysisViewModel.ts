import type { AgentAnalysis, AnalysisResponse } from '../../types/api';
import {
  buildChartSelectorOptions,
  normaliseChartSpec,
  toScatterDataset,
  toSnapshotDataset,
  toTimeseriesDataset,
  type ChartSelectorOption,
} from '../charting/fundamentalsChartUtils';

export type DashboardLayoutVariant =
  | 'news_synthesis'
  | 'fundamentals_synthesis'
  | 'synthesis_only'
  | 'multi_agent';

export type KnownAgentId = 'news' | 'fundamental';

export function isKnownAgentId(id: string): id is KnownAgentId {
  return id === 'news' || id === 'fundamental';
}

export function getAgentById(agents: AgentAnalysis[] | undefined, id: KnownAgentId): AgentAnalysis | null {
  if (!agents?.length) return null;
  return agents.find((agent) => agent.id === id) ?? null;
}

export function deriveLayoutVariant(data: AnalysisResponse): DashboardLayoutVariant {
  const hasNews = Boolean(getAgentById(data.agents, 'news'));
  const hasFundamental = Boolean(getAgentById(data.agents, 'fundamental'));

  if (hasNews && hasFundamental) return 'multi_agent';
  if (hasNews) return 'news_synthesis';
  if (hasFundamental) return 'fundamentals_synthesis';
  return 'synthesis_only';
}

function isRenderableFundamentalsOption(option: ChartSelectorOption, data: AnalysisResponse): boolean {
  if (option.kind !== 'fundamentals' || !option.spec) return false;
  const financialData = data.fundamentalData;
  if (!financialData) return false;

  const chartSpec = normaliseChartSpec(option.spec);
  if (chartSpec.data_mode === 'snapshot') {
    const snapshot = toSnapshotDataset(financialData, chartSpec.row_labels, chartSpec.snapshot_period);
    return snapshot.points.length > 0;
  }

  if (chartSpec.chart_type === 'scatter') {
    return toScatterDataset(financialData, chartSpec.row_labels).length > 0;
  }

  const timeseries = toTimeseriesDataset(financialData, chartSpec.row_labels);
  return timeseries.points.length > 0 && timeseries.series.length > 0;
}

export function getRenderableChartOptions(data: AnalysisResponse): ChartSelectorOption[] {
  const allOptions = buildChartSelectorOptions(data.chartData, data.fundamentalsVisualization);
  return allOptions.filter((option) => {
    if (option.kind === 'market') return (data.chartData?.length ?? 0) > 0;
    return isRenderableFundamentalsOption(option, data);
  });
}

export function getChartAvailability(data: AnalysisResponse) {
  const options = getRenderableChartOptions(data);
  const hasMarketChart = options.some((option) => option.kind === 'market');
  const hasFundamentalChart = options.some((option) => option.kind === 'fundamentals');
  return {
    options,
    hasAnyChart: options.length > 0,
    hasMarketChart,
    hasFundamentalChart,
  };
}
