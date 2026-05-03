import type {
  ChartDataPoint,
  DataFramePayload,
  FundamentalsChartSpecPayload,
  FundamentalsChartType,
  FundamentalsVisualizationPayload
} from '../../types/api';

export interface ChartSelectorOption {
  id: string;
  label: string;
  kind: 'market' | 'fundamentals';
  spec?: FundamentalsChartSpecPayload;
}

export interface TimeseriesChartDataset {
  points: Array<Record<string, number | string | null>>;
  series: Array<{ key: string; label: string }>;
}

export interface ScatterPoint {
  x: number;
  y: number;
  period: string;
  row: string;
}

export interface SnapshotPoint {
  name: string;
  value: number;
}

const SUPPORTED_CHART_TYPES: Set<string> = new Set([
  'line',
  'bar',
  'area',
  'scatter',
  'stacked_bar',
  'stacked_area',
  'pie'
]);

const SNAPSHOT_UNSUPPORTED_TYPES: Set<string> = new Set([
  'line',
  'area',
  'scatter',
  'stacked_area'
]);

function dedupeRowLabels(rowLabels: string[]): string[] {
  const deduped: string[] = [];
  const seen = new Set<string>();
  rowLabels.forEach((label) => {
    if (seen.has(label)) return;
    seen.add(label);
    deduped.push(label);
  });
  return deduped;
}

function canCoalesceSplitFragments(
  previous: FundamentalsChartSpecPayload,
  current: FundamentalsChartSpecPayload
): boolean {
  if (previous.group_rows || current.group_rows) return false;
  if (previous.row_labels.length !== 1 || current.row_labels.length !== 1) return false;
  return (
    previous.title === current.title &&
    previous.chart_type === current.chart_type &&
    previous.data_mode === current.data_mode &&
    previous.snapshot_period === current.snapshot_period &&
    previous.rationale === current.rationale
  );
}

export function getNormalisedFundamentalsCharts(
  visualization: FundamentalsVisualizationPayload | null | undefined
): FundamentalsChartSpecPayload[] {
  const charts = (visualization?.charts ?? []).map((chart) => normaliseChartSpec(chart));
  if (!charts.length) return [];

  const coalesced: FundamentalsChartSpecPayload[] = [];
  charts.forEach((chart) => {
    const normalizedRows = dedupeRowLabels(chart.row_labels);
    const normalizedChart: FundamentalsChartSpecPayload = {
      ...chart,
      row_labels: normalizedRows
    };

    const previous = coalesced[coalesced.length - 1];
    if (!previous || !canCoalesceSplitFragments(previous, normalizedChart)) {
      coalesced.push(normalizedChart);
      return;
    }

    coalesced[coalesced.length - 1] = {
      ...previous,
      row_labels: dedupeRowLabels([...previous.row_labels, ...normalizedChart.row_labels])
    };
  });

  return coalesced;
}

export function buildChartSelectorOptions(
  marketData: ChartDataPoint[] | undefined,
  visualization: FundamentalsVisualizationPayload | null | undefined
): ChartSelectorOption[] {
  const options: ChartSelectorOption[] = [];
  if ((marketData?.length ?? 0) > 0) {
    options.push({ id: 'market-price', label: 'Market Price', kind: 'market' });
  }

  const charts = getNormalisedFundamentalsCharts(visualization);
  charts.forEach((chart, index) => {
    const label =
      chart.title?.trim() ||
      chart.row_labels.join(', ').slice(0, 48) ||
      `Fundamentals ${index + 1}`;
    options.push({
      id: `fundamentals-${index}`,
      label,
      kind: 'fundamentals',
      spec: chart
    });
  });
  return options;
}

export function normaliseChartSpec(chart: FundamentalsChartSpecPayload): FundamentalsChartSpecPayload {
  const rawMode = (chart.data_mode || 'timeseries').toLowerCase();
  const dataMode = rawMode === 'snapshot' ? 'snapshot' : 'timeseries';
  const rawType = (chart.chart_type || 'line').toLowerCase();
  let chartType: FundamentalsChartType = SUPPORTED_CHART_TYPES.has(rawType)
    ? (rawType as FundamentalsChartType)
    : (dataMode === 'snapshot' ? 'bar' : 'line');

  if (chartType === 'pie') {
    return {
      ...chart,
      chart_type: 'pie',
      data_mode: 'snapshot',
      snapshot_period: chart.snapshot_period || 'latest'
    };
  }

  if (dataMode === 'snapshot' && SNAPSHOT_UNSUPPORTED_TYPES.has(chartType)) {
    chartType = 'bar';
  }

  return {
    ...chart,
    chart_type: chartType,
    data_mode: dataMode,
    snapshot_period: chart.snapshot_period || 'latest'
  };
}

function buildRowKey(rowLabel: string, index: number): string {
  const cleaned = rowLabel
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
  return `${cleaned || 'series'}_${index}`;
}

export function toTimeseriesDataset(
  payload: DataFramePayload | null | undefined,
  rowLabels: string[]
): TimeseriesChartDataset {
  if (!payload || !payload.columns.length) return { points: [], series: [] };
  const selectedRows = rowLabels.filter((row) => payload.index.includes(row));
  const series = selectedRows.map((label, index) => ({ key: buildRowKey(label, index), label }));
  const points = payload.columns.map((period, colIndex) => {
    const point: Record<string, number | string | null> = { period };
    selectedRows.forEach((rowLabel, rowIndex) => {
      const payloadRowIndex = payload.index.indexOf(rowLabel);
      point[series[rowIndex].key] = payloadRowIndex >= 0 ? payload.data[payloadRowIndex]?.[colIndex] ?? null : null;
    });
    return point;
  });
  return { points, series };
}

function resolveSnapshotColumnIndex(payload: DataFramePayload, snapshotPeriod: string): number {
  if (!payload.columns.length) return -1;
  const target = (snapshotPeriod || '').trim().toLowerCase();
  if (!target || target === 'latest') return payload.columns.length - 1;
  const index = payload.columns.findIndex((col) => col.toLowerCase() === target);
  return index >= 0 ? index : payload.columns.length - 1;
}

export function toSnapshotDataset(
  payload: DataFramePayload | null | undefined,
  rowLabels: string[],
  snapshotPeriod: string
): { points: SnapshotPoint[]; periodLabel: string } {
  if (!payload || !payload.columns.length) return { points: [], periodLabel: '' };
  const colIndex = resolveSnapshotColumnIndex(payload, snapshotPeriod);
  if (colIndex < 0) return { points: [], periodLabel: '' };
  const periodLabel = payload.columns[colIndex] ?? '';
  const points: SnapshotPoint[] = [];
  rowLabels.forEach((rowLabel) => {
    const rowIndex = payload.index.indexOf(rowLabel);
    if (rowIndex < 0) return;
    const value = payload.data[rowIndex]?.[colIndex];
    if (typeof value === 'number' && Number.isFinite(value)) {
      points.push({ name: rowLabel, value });
    }
  });
  return { points, periodLabel };
}

export function toScatterDataset(
  payload: DataFramePayload | null | undefined,
  rowLabels: string[]
): ScatterPoint[] {
  if (!payload || !payload.columns.length) return [];
  const points: ScatterPoint[] = [];
  rowLabels.forEach((rowLabel) => {
    const rowIndex = payload.index.indexOf(rowLabel);
    if (rowIndex < 0) return;
    payload.columns.forEach((period, colIndex) => {
      const value = payload.data[rowIndex]?.[colIndex];
      if (typeof value === 'number' && Number.isFinite(value)) {
        points.push({ x: colIndex, y: value, period, row: rowLabel });
      }
    });
  });
  return points;
}

export function shouldFallbackPieToBar(points: SnapshotPoint[]): boolean {
  if (!points.length) return true;
  return points.every((point) => !(point.value > 0));
}
