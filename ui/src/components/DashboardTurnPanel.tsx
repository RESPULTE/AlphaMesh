import { useEffect, useMemo, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  Area,
  AreaChart as RechartsAreaChart,
  Bar,
  BarChart as RechartsBarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart as RechartsLineChart,
  Pie,
  PieChart as RechartsPieChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart as RechartsScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  Star,
  TrendingUp,
  ArrowRight,
  Download,
  Sparkles,
  User,
  FileText,
  BarChart2,
  CheckCircle2,
  Building2,
  X,
  ChevronDown,
  ChevronUp,
  ExternalLink,
} from 'lucide-react';
import clsx from 'clsx';
import Markdown from 'react-markdown';
import type { AgentAnalysis, AnalysisResponse } from '../types/api';
import type { StreamPhase } from '../hooks/useAnalysisStream';
import {
  normaliseChartSpec,
  shouldFallbackPieToBar,
  toScatterDataset,
  toSnapshotDataset,
  toTimeseriesDataset,
} from './charting/fundamentalsChartUtils';
import {
  deriveLayoutVariant,
  getAgentById,
  getChartAvailability,
  type DashboardLayoutVariant,
} from './dashboard/analysisViewModel';

interface DashboardTurnPanelProps {
  query: string;
  data: AnalysisResponse;
  isStreaming?: boolean;
  streamPhase?: StreamPhase;
}

const CHART_COLORS = ['#007a01', '#2b9f30', '#6ecf72', '#87d98a', '#b6e8b9', '#d1f2d3'];

function AgentModal({ agent, onClose }: { agent: AgentAnalysis; onClose: () => void }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const hasExtraContent = !!agent.tableData || !!agent.references;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 md:p-8">
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
      />
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 20 }}
        className={clsx(
          'relative w-full max-h-[90vh] bg-surface rounded-3xl md:rounded-[2.5rem] shadow-2xl border border-outline-variant/20 flex flex-col overflow-hidden',
          hasExtraContent ? 'max-w-5xl' : 'max-w-3xl'
        )}
      >
        <div className="flex items-center justify-between p-6 md:p-8 border-b border-outline-variant/10 bg-surface-container-lowest shrink-0">
          <div className="flex items-center gap-4">
            <div className="bg-primary/10 p-3 rounded-xl">
              {agent.id === 'news' ? (
                <FileText className="w-6 h-6 text-primary" />
              ) : (
                <BarChart2 className="w-6 h-6 text-primary" />
              )}
            </div>
            <div>
              <div className="text-[10px] md:text-xs font-black font-label tracking-[0.2em] text-primary uppercase mb-1">
                {agent.category}
              </div>
              <h2 className="font-headline text-xl md:text-2xl font-extrabold text-on-surface">
                {agent.name}
              </h2>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-full hover:bg-surface-container-high transition-colors text-on-surface-variant"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        <div className="flex flex-col md:flex-row overflow-y-auto custom-scrollbar bg-surface flex-grow">
          <div
            className={clsx(
              'p-6 md:p-8',
              hasExtraContent ? 'md:w-1/2 md:border-r border-outline-variant/10' : 'w-full'
            )}
          >
            <div className="prose prose-sm md:prose-base prose-invert max-w-none prose-headings:font-headline prose-headings:font-bold prose-a:text-primary hover:prose-a:text-primary/80 prose-strong:text-on-surface prose-p:text-on-surface-variant prose-li:text-on-surface-variant">
              <Markdown>{agent.fullReport || 'No detailed report available.'}</Markdown>
            </div>
          </div>

          {hasExtraContent && (
            <div className="md:w-1/2 flex flex-col bg-surface-container-lowest md:bg-transparent">
              <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="md:hidden flex items-center justify-between p-6 border-t border-outline-variant/10 font-bold text-on-surface"
              >
                <span>{agent.tableData ? 'View Financial Data' : 'View References'}</span>
                {isExpanded ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
              </button>

              <div
                className={clsx(
                  'p-6 md:p-10 flex-grow bg-surface-container-lowest/30',
                  isExpanded ? 'block' : 'hidden md:flex flex-col justify-center'
                )}
              >
                {agent.tableData && (
                  <div className="w-full max-w-lg mx-auto">
                    <div className="bg-surface-container-lowest rounded-2xl border border-outline-variant/20 overflow-hidden shadow-sm">
                      <div className="p-4 md:p-5 border-b border-outline-variant/20 bg-surface-container-low/30 flex items-center gap-3">
                        <div className="p-2 bg-primary/10 rounded-lg">
                          <BarChart2 className="w-4 h-4 text-primary" />
                        </div>
                        <h3 className="font-headline font-bold text-sm md:text-base text-on-surface">
                          {agent.tableData.title}
                        </h3>
                      </div>
                      <div className="overflow-x-auto">
                        <table className="w-full text-left border-collapse">
                          <thead>
                            <tr className="bg-surface-container-low/10">
                              {agent.tableData.headers.map((header, i) => (
                                <th
                                  key={i}
                                  className="py-3 px-4 md:px-5 text-xs font-black font-label tracking-wider text-on-surface-variant/50 uppercase border-b border-outline-variant/10"
                                >
                                  {header}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-outline-variant/10">
                            {agent.tableData.rows.map((row, i) => (
                              <tr key={i} className="hover:bg-surface-container-low/40 transition-colors group">
                                {row.map((cell, j) => (
                                  <td
                                    key={j}
                                    className={clsx(
                                      'py-3 md:py-4 px-4 md:px-5 text-sm',
                                      j === 0
                                        ? 'font-medium text-on-surface'
                                        : 'text-on-surface-variant font-mono group-hover:text-primary transition-colors'
                                    )}
                                  >
                                    {cell}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                )}

                {agent.references && (
                  <div className="w-full max-w-lg mx-auto">
                    <div className="flex items-center gap-3 mb-6">
                      <div className="p-2 bg-primary/10 rounded-lg">
                        <FileText className="w-4 h-4 text-primary" />
                      </div>
                      <h3 className="font-headline font-bold text-lg text-on-surface">Sources & References</h3>
                    </div>
                    <div className="space-y-3">
                      {agent.references.map((ref, index) => (
                        <a
                          key={ref.id}
                          href={ref.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="group flex items-start gap-4 p-4 rounded-2xl bg-surface-container-lowest border border-outline-variant/20 hover:border-primary/40 hover:bg-surface-container-low/50 hover:shadow-md hover:-translate-y-0.5 transition-all duration-300"
                        >
                          <div className="flex-shrink-0 w-7 h-7 rounded-full bg-surface-container-high group-hover:bg-primary/10 group-hover:text-primary text-on-surface-variant flex items-center justify-center text-xs font-bold font-mono mt-0.5 transition-colors">
                            {index + 1}
                          </div>
                          <div className="flex-grow">
                            <h4 className="text-sm font-medium text-on-surface group-hover:text-primary transition-colors line-clamp-2 mb-1.5 leading-snug">
                              {ref.title}
                            </h4>
                            <div className="flex items-center gap-2 text-[11px] text-on-surface-variant/60 font-mono uppercase tracking-wider">
                              <span>{ref.source}</span>
                            </div>
                          </div>
                          <ExternalLink className="w-4 h-4 text-on-surface-variant/30 group-hover:text-primary transition-colors flex-shrink-0 mt-1" />
                        </a>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </motion.div>
    </div>
  );
}

function AgentInsightCard({ agent, onSelect }: { agent: AgentAnalysis; onSelect: (agent: AgentAnalysis) => void }) {
  const isNews = agent.id === 'news';

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="relative flex flex-col h-full">
      <div
        className="absolute -top-4 left-0 bg-surface-container-low h-8 md:h-10 w-40 md:w-52 z-10 flex items-center px-5 md:px-7 border-t border-l border-outline-variant/20"
        style={{ clipPath: 'polygon(0% 0%, 75% 0%, 85% 100%, 0% 100%)' }}
      >
        <span className="text-[8px] md:text-[9px] font-black font-label tracking-[0.2em] text-primary uppercase">
          {agent.category}
        </span>
      </div>
      <div
        onClick={() => onSelect(agent)}
        className="bg-surface rounded-tr-2xl rounded-b-2xl md:rounded-tr-[2.5rem] md:rounded-b-[2.5rem] p-5 pt-8 md:p-8 md:pt-12 h-full flex flex-col transition-all border border-outline-variant/20 shadow-[0_4px_6px_-1px_rgba(0,0,0,0.03),0_20px_30px_-5px_rgba(0,0,0,0.06)] hover:shadow-2xl duration-500 cursor-pointer"
      >
        <div className="flex justify-between items-start mb-4 md:mb-6">
          <h2 className="font-headline text-base md:text-lg font-extrabold mb-1 text-on-surface">{agent.name}</h2>
          <div className="bg-primary/10 p-2 md:p-2.5 rounded-lg md:rounded-xl">
            {isNews ? (
              <FileText className="w-4 h-4 md:w-5 md:h-5 text-primary" />
            ) : (
              <BarChart2 className="w-4 h-4 md:w-5 md:h-5 text-primary" />
            )}
          </div>
        </div>

        {isNews ? (
          <div className="space-y-3 md:space-y-4 flex-grow">
            <div className="space-y-1.5 md:space-y-2">
              <div className="flex justify-between text-[8px] md:text-[9px] font-black font-label uppercase tracking-widest text-on-surface-variant/60">
                <span>Recent Catalyst</span>
                <span>{agent.recentCatalyst.timeAgo}</span>
              </div>
              <h3 className="font-bold text-sm md:text-base leading-tight text-on-surface">{agent.recentCatalyst.title}</h3>
              <p className="text-[10px] md:text-xs text-on-surface-variant leading-relaxed font-medium">{agent.recentCatalyst.description}</p>
            </div>
            <div className="space-y-2 md:space-y-3 border-t border-outline-variant/20 pt-3 md:pt-4">
              <div className="flex justify-between text-[8px] md:text-[9px] font-black font-label uppercase tracking-widest text-on-surface-variant/60">
                <span>Sentiment Score</span>
                <span className="text-primary font-black">{agent.sentiment.label}</span>
              </div>
              <div className="h-1.5 md:h-2 w-full bg-surface-container-highest rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${agent.sentiment.score}%` }}
                  transition={{ duration: 1, delay: 0.5 }}
                  className="h-full bg-primary rounded-full shadow-[0_0_10px_rgba(0,110,1,0.3)]"
                />
              </div>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-3 md:gap-4 flex-grow content-start">
            {agent.metrics?.map((metric, i) => (
              <div key={i} className="bg-surface-container-lowest p-3 md:p-4 rounded-lg md:rounded-xl border border-outline-variant/20 shadow-sm">
                <span className="text-[7px] md:text-[8px] font-black font-label tracking-[0.15em] text-on-surface-variant/60 uppercase block">
                  {metric.label}
                </span>
                <span className="text-base md:text-xl font-headline font-black text-on-surface">{metric.value}</span>
              </div>
            ))}
            {agent.quote && (
              <div className="col-span-2 bg-surface-container-lowest p-3 md:p-4 rounded-lg md:rounded-xl border border-outline-variant/20 shadow-sm">
                <p className="text-[10px] md:text-[11px] text-on-surface-variant leading-relaxed italic font-medium">{agent.quote}</p>
              </div>
            )}
          </div>
        )}

        <button className="mt-6 md:mt-8 flex items-center justify-between w-full text-primary font-bold text-[10px] md:text-xs border-t border-outline-variant/20 pt-3 md:pt-4">
          <span>{isNews ? 'Full Report' : 'Financial Deep-Dive'}</span>
          <ArrowRight className="w-3.5 h-3.5 md:w-4 md:h-4" />
        </button>
      </div>
    </motion.div>
  );
}

function SummaryPanel({ data, isStreaming }: { data: AnalysisResponse; isStreaming: boolean }) {
  return (
    <div className="bg-surface-container-lowest rounded-2xl md:rounded-[2.5rem] p-6 md:p-10 h-full shadow-[0_4px_6px_-1px_rgba(0,0,0,0.03),0_20px_30px_-5px_rgba(0,0,0,0.06)] flex flex-col border border-outline-variant/20 relative overflow-hidden">
      <div className="absolute -top-12 -right-12 md:-top-24 md:-right-24 w-32 h-32 md:w-64 md:h-64 bg-primary/5 rounded-full blur-[50px] md:blur-[100px]"></div>

      <div className="relative z-10 flex flex-col h-full">
        <div className="flex items-center gap-2 md:gap-3 mb-6 md:mb-10 justify-between w-full">
          <h2 className="font-headline text-lg md:text-xl font-bold tracking-tight text-on-surface">Summary of Findings</h2>
          <div className="w-8 h-8 md:w-10 md:h-10 rounded-lg md:rounded-xl bg-primary/10 flex items-center justify-center border border-primary/10 ml-auto">
            <Sparkles className="w-4 h-4 md:w-5 md:h-5 text-primary" />
          </div>
        </div>

        <div className="flex-grow space-y-6 md:space-y-10">
          <div>
            <h3 className="text-[9px] md:text-[10px] font-black uppercase tracking-[0.25em] text-primary mb-2 md:mb-4">Core Narrative</h3>
            <p className="text-on-surface-variant text-xs md:text-sm leading-relaxed font-medium min-h-[60px] md:min-h-[80px]">
              {data.summary?.coreNarrative || (
                <span className="animate-pulse text-on-surface-variant/40">Synthesizing data...</span>
              )}
            </p>
          </div>

          {data.summary?.agentConsensus && data.summary.agentConsensus.length > 0 && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4 md:space-y-6">
              <h3 className="text-[9px] md:text-[10px] font-black uppercase tracking-[0.25em] text-primary">Agent Consensus</h3>
              <div className="space-y-3 md:space-y-4">
                {data.summary.agentConsensus.map((consensus, i) => (
                  <div key={i} className="flex items-start gap-3 md:gap-4 p-3 md:p-4 rounded-xl md:rounded-2xl bg-surface border border-outline-variant/20 shadow-sm">
                    {consensus.icon === 'verified' ? (
                      <CheckCircle2 className="w-4 h-4 md:w-5 md:h-5 text-primary mt-0.5 shrink-0" />
                    ) : (
                      <Building2 className="w-4 h-4 md:w-5 md:h-5 text-primary mt-0.5 shrink-0" />
                    )}
                    <div>
                      <p className="text-[11px] md:text-xs font-bold text-on-surface mb-0.5 md:mb-1">{consensus.title}</p>
                      <p className="text-[10px] md:text-[11px] text-on-surface-variant/80 leading-tight">{consensus.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          )}

          {data.summary?.verdict?.label && (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="pt-4 md:pt-6 border-t border-outline-variant/10">
              <div className="flex items-center justify-between mb-1.5 md:mb-2">
                <span className="text-[9px] md:text-[10px] font-black uppercase tracking-widest text-on-surface-variant/60">AlphaMesh Verdict</span>
                <span className="text-xs md:text-sm font-black text-primary">{data.summary.verdict.label}</span>
              </div>
              <p className="text-[10px] md:text-xs text-on-surface-variant/80 font-medium leading-relaxed">{data.summary.verdict.description}</p>
            </motion.div>
          )}
        </div>

        <button
          disabled={isStreaming}
          className={clsx(
            'mt-8 md:mt-12 w-full py-4 md:py-5 rounded-xl md:rounded-2xl font-black text-[10px] md:text-xs flex items-center justify-center gap-1.5 md:gap-2 group shadow-lg transition-all',
            isStreaming
              ? 'bg-surface-container-high text-on-surface-variant/50 cursor-not-allowed'
              : 'bg-inverse-surface text-inverse-on-surface hover:bg-inverse-surface/90'
          )}
        >
          EXPORT REPORT
          <Download className="w-3.5 h-3.5 md:w-4 md:h-4" />
        </button>
      </div>
    </div>
  );
}

function SynthesisHeroCard({ data, isStreaming }: { data: AnalysisResponse; isStreaming: boolean }) {
  return (
    <div className="rounded-2xl md:rounded-[2.5rem] border border-outline-variant/20 bg-surface-container-lowest overflow-hidden shadow-[0_12px_30px_rgba(0,0,0,0.08)]">
      <div className="bg-gradient-to-r from-[#003c08] via-[#005d0b] to-[#007a01] p-[1px]">
        <div className="bg-surface-container-lowest px-6 md:px-8 py-6 md:py-8 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="bg-primary/15 p-3 rounded-xl border border-primary/20">
              <Sparkles className="w-5 h-5 text-primary" />
            </div>
            <div>
              <div className="text-[10px] md:text-xs font-black font-label tracking-[0.2em] text-primary uppercase">AlphaMesh Agent</div>
              <h2 className="font-headline text-2xl md:text-3xl font-extrabold text-on-surface">Response</h2>
            </div>
          </div>
          {data.summary?.verdict?.label && (
            <span className="text-xs md:text-sm font-black text-primary bg-primary/10 border border-primary/20 px-3 py-1.5 rounded-full">
              {data.summary.verdict.label}
            </span>
          )}
        </div>
      </div>

      <div className="px-6 md:px-8 py-7 md:py-10">
        <p className="text-on-surface text-base md:text-2xl leading-relaxed md:leading-relaxed font-medium">
          {data.summary?.coreNarrative || (
            <span className="animate-pulse text-on-surface-variant/40">Synthesizing data...</span>
          )}
        </p>

        {data.summary?.verdict?.description && (
          <p className="mt-6 text-on-surface-variant text-sm md:text-base leading-relaxed">
            {data.summary.verdict.description}
          </p>
        )}

        {isStreaming && (
          <div className="mt-6 inline-flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-on-surface-variant/60">
            <motion.div
              animate={{ opacity: [0.3, 1, 0.3] }}
              transition={{ repeat: Infinity, duration: 1.2 }}
              className="w-2 h-2 rounded-full bg-primary"
            />
            Analyzing...
          </div>
        )}
      </div>
    </div>
  );
}

function renderMarketChart(chartData: AnalysisResponse['chartData']) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <RechartsLineChart data={chartData}>
        <YAxis domain={['dataMin - 5', 'dataMax + 5']} hide />
        <Line
          type="monotone"
          dataKey="price"
          stroke="#006e01"
          strokeWidth={5}
          dot={false}
          isAnimationActive={true}
          animationDuration={1500}
        />
      </RechartsLineChart>
    </ResponsiveContainer>
  );
}

export default function DashboardTurnPanel({
  query,
  data,
  isStreaming = false,
  streamPhase = 'completed',
}: DashboardTurnPanelProps) {
  const [selectedAgent, setSelectedAgent] = useState<AgentAnalysis | null>(null);
  const [selectedChartId, setSelectedChartId] = useState<string>('market-price');

  const chartAvailability = useMemo(() => getChartAvailability(data), [data]);
  const chartOptions = chartAvailability.options;

  const layoutVariant: DashboardLayoutVariant = useMemo(() => deriveLayoutVariant(data), [data]);
  const newsAgent = useMemo(() => getAgentById(data.agents, 'news'), [data.agents]);
  const fundamentalAgent = useMemo(() => getAgentById(data.agents, 'fundamental'), [data.agents]);

  const visibleAgents = useMemo(() => {
    const items: AgentAnalysis[] = [];
    if (newsAgent) items.push(newsAgent);
    if (fundamentalAgent) items.push(fundamentalAgent);
    return items;
  }, [newsAgent, fundamentalAgent]);

  useEffect(() => {
    if (!chartOptions.some((option) => option.id === selectedChartId)) {
      setSelectedChartId(chartOptions[0]?.id ?? 'market-price');
    }
  }, [chartOptions, selectedChartId]);

  const selectedChart = chartOptions.find((option) => option.id === selectedChartId) ?? chartOptions[0];

  const priceText = data?.currentPrice != null ? data.currentPrice.toFixed(2) : '--';
  const changeText =
    data?.priceChange != null && data?.priceChangePercent != null
      ? `+${data.priceChange.toFixed(2)} (${data.priceChangePercent.toFixed(2)}%)`
      : '--';
  const hasIdentifiedCompany =
    Boolean(data.ticker?.trim()) &&
    Boolean(data.companyName?.trim()) &&
    data.companyName.trim().toLowerCase() !== 'company';
  const canRevealAnalytics = hasIdentifiedCompany || (!isStreaming && streamPhase !== 'awaiting_ticker');
  const showAnalyticsSection =
    canRevealAnalytics &&
    (hasIdentifiedCompany || chartAvailability.hasAnyChart || visibleAgents.length > 0 || layoutVariant !== 'synthesis_only');

  // Latch: once the analytics section has been revealed during this stream, keep it
  // visible even if computed state briefly dips back to false (e.g. ticker resolves,
  // then agents array is momentarily empty before the first agent event arrives).
  const analyticsEverRevealedRef = useRef(false);
  if (showAnalyticsSection) analyticsEverRevealedRef.current = true;
  // Reset latch when a brand-new stream begins so a fresh query starts clean.
  useEffect(() => {
    if (streamPhase === 'starting') analyticsEverRevealedRef.current = false;
  }, [streamPhase]);
  const stableShowAnalytics = showAnalyticsSection || analyticsEverRevealedRef.current;

  const renderSelectedChart = () => {
    if (!selectedChart) return null;

    if (selectedChart.kind === 'market') {
      return renderMarketChart(data.chartData);
    }

    const chartSpec = normaliseChartSpec(selectedChart.spec!);
    const financialData = data?.fundamentalData;
    if (!financialData) return null;

    if (chartSpec.data_mode === 'snapshot') {
      const snapshot = toSnapshotDataset(financialData, chartSpec.row_labels, chartSpec.snapshot_period);
      if (!snapshot.points.length) return null;

      const renderPie = chartSpec.chart_type === 'pie' && !shouldFallbackPieToBar(snapshot.points);
      if (renderPie) {
        return (
          <ResponsiveContainer width="100%" height="100%">
            <RechartsPieChart>
              <Tooltip />
              <Legend />
              <Pie data={snapshot.points} dataKey="value" nameKey="name" outerRadius={110} label>
                {snapshot.points.map((point, index) => (
                  <Cell key={`${point.name}-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                ))}
              </Pie>
            </RechartsPieChart>
          </ResponsiveContainer>
        );
      }

      return (
        <ResponsiveContainer width="100%" height="100%">
          <RechartsBarChart data={snapshot.points} margin={{ top: 16, right: 24, left: 8, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#d9d9d9" />
            <XAxis dataKey="name" tick={{ fill: '#4a4a4a', fontSize: 12 }} />
            <YAxis tick={{ fill: '#4a4a4a', fontSize: 12 }} />
            <Tooltip />
            <Bar dataKey="value" fill="#007a01" radius={[6, 6, 0, 0]} />
          </RechartsBarChart>
        </ResponsiveContainer>
      );
    }

    if (chartSpec.chart_type === 'scatter') {
      const scatterPoints = toScatterDataset(financialData, chartSpec.row_labels);
      if (!scatterPoints.length) return null;

      return (
        <ResponsiveContainer width="100%" height="100%">
          <RechartsScatterChart margin={{ top: 16, right: 24, left: 8, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#d9d9d9" />
            <XAxis dataKey="x" tick={{ fill: '#4a4a4a', fontSize: 12 }} />
            <YAxis dataKey="y" tick={{ fill: '#4a4a4a', fontSize: 12 }} />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Scatter data={scatterPoints} fill="#007a01" />
          </RechartsScatterChart>
        </ResponsiveContainer>
      );
    }

    const timeseries = toTimeseriesDataset(financialData, chartSpec.row_labels);
    if (!timeseries.points.length || !timeseries.series.length) return null;

    const seriesEls = timeseries.series.map((series, index) => {
      const color = CHART_COLORS[index % CHART_COLORS.length];
      if (chartSpec.chart_type === 'area' || chartSpec.chart_type === 'stacked_area') {
        return (
          <Area
            key={series.key}
            type="monotone"
            dataKey={series.key}
            name={series.label}
            stroke={color}
            fill={color}
            fillOpacity={0.2}
            strokeWidth={2}
            stackId={chartSpec.chart_type === 'stacked_area' ? 'stack' : undefined}
          />
        );
      }
      if (chartSpec.chart_type === 'bar' || chartSpec.chart_type === 'stacked_bar') {
        return (
          <Bar
            key={series.key}
            dataKey={series.key}
            name={series.label}
            fill={color}
            radius={[4, 4, 0, 0]}
            stackId={chartSpec.chart_type === 'stacked_bar' ? 'stack' : undefined}
          />
        );
      }
      return (
        <Line
          key={series.key}
          type="monotone"
          dataKey={series.key}
          name={series.label}
          stroke={color}
          strokeWidth={2.5}
          dot={false}
        />
      );
    });

    if (chartSpec.chart_type === 'area' || chartSpec.chart_type === 'stacked_area') {
      return (
        <ResponsiveContainer width="100%" height="100%">
          <RechartsAreaChart data={timeseries.points} margin={{ top: 16, right: 24, left: 8, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#d9d9d9" />
            <XAxis dataKey="period" tick={{ fill: '#4a4a4a', fontSize: 12 }} />
            <YAxis tick={{ fill: '#4a4a4a', fontSize: 12 }} />
            <Tooltip />
            <Legend />
            {seriesEls}
          </RechartsAreaChart>
        </ResponsiveContainer>
      );
    }

    if (chartSpec.chart_type === 'bar' || chartSpec.chart_type === 'stacked_bar') {
      return (
        <ResponsiveContainer width="100%" height="100%">
          <RechartsBarChart data={timeseries.points} margin={{ top: 16, right: 24, left: 8, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#d9d9d9" />
            <XAxis dataKey="period" tick={{ fill: '#4a4a4a', fontSize: 12 }} />
            <YAxis tick={{ fill: '#4a4a4a', fontSize: 12 }} />
            <Tooltip />
            <Legend />
            {seriesEls}
          </RechartsBarChart>
        </ResponsiveContainer>
      );
    }

    return (
      <ResponsiveContainer width="100%" height="100%">
        <RechartsLineChart data={timeseries.points}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#d9d9d9" />
          <XAxis dataKey="period" tick={{ fill: '#4a4a4a', fontSize: 12 }} />
          <YAxis tick={{ fill: '#4a4a4a', fontSize: 12 }} />
          <Tooltip />
          <Legend />
          {seriesEls}
        </RechartsLineChart>
      </ResponsiveContainer>
    );
  };

  return (
    <motion.section
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.25, ease: 'easeOut' }}
    >
      <div className="bg-surface-container-lowest rounded-3xl md:rounded-[3rem] p-5 md:p-12 shadow-[0_20px_40px_rgba(0,0,0,0.06)] border border-outline-variant/20 relative">
        <div className="flex justify-end mb-6 md:mb-8">
          <div className="flex items-start gap-2 md:gap-3 max-w-md">
            <div className="flex flex-col items-end gap-1.5 md:gap-2">
              <div className="bg-surface-container-low/80 backdrop-blur-sm rounded-xl md:rounded-2xl rounded-tr-none px-3 py-2 md:px-5 md:py-3 border border-outline-variant/20 shadow-sm">
                <p className="text-xs md:text-sm font-semibold text-on-surface leading-relaxed">{query}</p>
              </div>
              <span className="text-[8px] md:text-[10px] font-bold text-on-surface-variant/60 uppercase tracking-widest mr-1">
                {isStreaming ? 'ANALYZING...' : 'JUST NOW'}
              </span>
            </div>
            <div className="w-6 h-6 md:w-8 md:h-8 rounded-full bg-surface-container-high flex items-center justify-center shrink-0 border border-surface-container-lowest shadow-sm">
              <User className="w-3 h-3 md:w-4 md:h-4 text-on-surface-variant" />
            </div>
          </div>
        </div>

        {!stableShowAnalytics && <SynthesisHeroCard data={data} isStreaming={isStreaming} />}

        <AnimatePresence mode="sync">
          {stableShowAnalytics && (
            <motion.div
              key="analytics-section"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.35, ease: [0.22, 0.61, 0.36, 1] }}
              className="mt-8 md:mt-10"
            >
              {hasIdentifiedCompany && (
                <motion.div
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.16, ease: [0.22, 0.61, 0.36, 1] }}
                  className="flex flex-col md:flex-row justify-between items-start md:items-end mb-8 md:mb-14 gap-4 md:gap-6"
                >
                  <div>
                    <div className="flex items-center gap-2 md:gap-3 mb-2 md:mb-4">
                      <span className="bg-surface-container-low text-on-surface-variant px-2 py-0.5 md:px-3 md:py-1 rounded-full text-[8px] md:text-[10px] font-black font-label tracking-[0.15em] uppercase border border-outline-variant/20">
                        {data.ticker}
                      </span>
                      <h1 className="font-headline text-xl md:text-2xl font-bold tracking-tight text-on-surface">
                        {data.companyName}
                      </h1>
                    </div>
                    <div className="flex items-baseline gap-3 md:gap-6">
                      <span className="font-headline text-4xl md:text-7xl font-black leading-none tracking-tighter text-on-surface filter drop-shadow-sm">
                        {priceText}
                      </span>
                      <div className="flex flex-col">
                        <span className="text-primary font-black flex items-center text-sm md:text-xl">
                          <TrendingUp className="w-4 h-4 md:w-5 md:h-5 mr-1" />
                          {changeText}
                        </span>
                        <span className="text-on-surface-variant/60 text-[8px] md:text-[10px] font-bold font-label uppercase tracking-[0.2em] mt-0.5 md:mt-1">
                          {data.marketStatus || 'MARKET DATA UNAVAILABLE'}
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="flex gap-2 md:gap-4 w-full md:w-auto mt-2 md:mt-0">
                    <button className="flex-1 md:flex-none flex items-center justify-center gap-1.5 md:gap-2 px-4 py-3 md:px-8 md:py-4 rounded-xl md:rounded-2xl border-2 border-surface-container-high hover:bg-surface-container-low hover:border-outline-variant/30 transition-all font-bold text-xs md:text-sm text-on-surface-variant">
                      <Star className="w-4 h-4 md:w-5 md:h-5" />
                      Wishlist
                    </button>
                    <button className="flex-1 md:flex-none bg-gradient-to-br from-[#007a01] to-[#00d905] text-white px-6 py-3 md:px-12 md:py-4 rounded-xl md:rounded-2xl font-black text-xs md:text-sm shadow-[0_0_30px_rgba(0,200,5,0.35)] active:scale-[0.98] transition-all transform hover:-translate-y-0.5">
                      Trade Now
                    </button>
                  </div>
                </motion.div>
              )}

              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 md:gap-10">
                <div className="lg:col-span-8 space-y-6 md:space-y-10">
                  {chartAvailability.hasAnyChart && (
                    <motion.section
                      layout
                      initial={{ opacity: 0, scale: 0.985 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.99 }}
                      transition={{ duration: 0.2, ease: [0.22, 0.61, 0.36, 1] }}
                    >
                      <div className="mb-3 md:mb-4 flex flex-col md:flex-row md:items-center md:justify-between gap-2">
                        <div className="text-[9px] md:text-[10px] font-black font-label tracking-[0.2em] text-on-surface-variant uppercase">Chart View</div>
                        {chartOptions.length > 1 && (
                          <div className="relative">
                            <select
                              value={selectedChart?.id ?? chartOptions[0]?.id ?? ''}
                              onChange={(event) => setSelectedChartId(event.target.value)}
                              className="appearance-none bg-surface-container-low border border-outline-variant/30 text-on-surface text-xs md:text-sm font-semibold rounded-xl px-4 py-2 pr-10 max-w-[360px] w-full md:w-auto focus:outline-none focus:ring-2 focus:ring-primary/30"
                            >
                              {chartOptions.map((option) => (
                                <option key={option.id} value={option.id}>
                                  {option.label}
                                </option>
                              ))}
                            </select>
                            <ChevronDown className="w-4 h-4 text-on-surface-variant absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none" />
                          </div>
                        )}
                      </div>
                      <div className="relative h-[250px] md:h-[360px] w-full bg-surface/80 rounded-2xl md:rounded-[2.5rem] overflow-hidden border border-outline-variant/20">
                        {renderSelectedChart()}
                      </div>
                    </motion.section>
                  )}

                  {visibleAgents.length > 0 && (
                    <motion.div
                      layout
                      initial="hidden"
                      animate="show"
                      variants={{
                        hidden: {},
                        show: {
                          transition: {
                            staggerChildren: 0.07,
                          },
                        },
                      }}
                      className={clsx(
                        'grid gap-6 md:gap-10 items-stretch',
                        visibleAgents.length > 1 ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1'
                      )}
                    >
                      {visibleAgents.map((agent) => (
                        <motion.div
                          key={agent.id}
                          variants={{
                            hidden: { opacity: 0, y: 14 },
                            show: {
                              opacity: 1,
                              y: 0,
                              transition: { duration: 0.2, ease: [0.22, 0.61, 0.36, 1] },
                            },
                          }}
                        >
                          <AgentInsightCard agent={agent} onSelect={setSelectedAgent} />
                        </motion.div>
                      ))}
                    </motion.div>
                  )}
                </div>

                <div className="lg:col-span-4 flex flex-col h-full">
                  <SummaryPanel data={data} isStreaming={isStreaming} />
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <AnimatePresence>{selectedAgent && <AgentModal agent={selectedAgent} onClose={() => setSelectedAgent(null)} />}</AnimatePresence>
    </motion.section>
  );
}
