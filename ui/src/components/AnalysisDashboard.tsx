import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { useAnalysisStream } from '../hooks/useAnalysisStream';
import { LineChart as RechartsLineChart, Line, ResponsiveContainer, YAxis } from 'recharts';
import { Star, TrendingUp, ArrowRight, Download, Sparkles, User, FileText, BarChart2, CheckCircle2, Building2, ArrowLeft, X, ChevronDown, ChevronUp, ExternalLink } from 'lucide-react';
import clsx from 'clsx';
import Markdown from 'react-markdown';
import { AgentAnalysis } from '../types/api';

interface AnalysisDashboardProps {
  query: string;
  onBack?: () => void;
}

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
          "relative w-full max-h-[90vh] bg-surface rounded-3xl md:rounded-[2.5rem] shadow-2xl border border-outline-variant/20 flex flex-col overflow-hidden",
          hasExtraContent ? "max-w-5xl" : "max-w-3xl"
        )}
      >
        <div className="flex items-center justify-between p-6 md:p-8 border-b border-outline-variant/10 bg-surface-container-lowest shrink-0">
          <div className="flex items-center gap-4">
            <div className="bg-primary/10 p-3 rounded-xl">
              {agent.icon === 'news' ? (
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
          <div className={clsx("p-6 md:p-8", hasExtraContent ? "md:w-1/2 md:border-r border-outline-variant/10" : "w-full")}>
            <div className="prose prose-sm md:prose-base prose-invert max-w-none prose-headings:font-headline prose-headings:font-bold prose-a:text-primary hover:prose-a:text-primary/80 prose-strong:text-on-surface prose-p:text-on-surface-variant prose-li:text-on-surface-variant">
              <Markdown>
                {agent.fullReport || "No detailed report available."}
              </Markdown>
            </div>
          </div>

          {hasExtraContent && (
            <div className="md:w-1/2 flex flex-col bg-surface-container-lowest md:bg-transparent">
              {/* Mobile Expand Button */}
              <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="md:hidden flex items-center justify-between p-6 border-t border-outline-variant/10 font-bold text-on-surface"
              >
                <span>{agent.tableData ? "View Financial Data" : "View References"}</span>
                {isExpanded ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
              </button>

              {/* Extra Content Area */}
              <div className={clsx(
                "p-6 md:p-10 flex-grow bg-surface-container-lowest/30",
                isExpanded ? "block" : "hidden md:flex flex-col justify-center"
              )}>
                {agent.tableData && (
                  <div className="w-full max-w-lg mx-auto">
                    <div className="bg-surface-container-lowest rounded-2xl border border-outline-variant/20 overflow-hidden shadow-sm">
                      <div className="p-4 md:p-5 border-b border-outline-variant/20 bg-surface-container-low/30 flex items-center gap-3">
                        <div className="p-2 bg-primary/10 rounded-lg">
                          <BarChart2 className="w-4 h-4 text-primary" />
                        </div>
                        <h3 className="font-headline font-bold text-sm md:text-base text-on-surface">{agent.tableData.title}</h3>
                      </div>
                      <div className="overflow-x-auto">
                        <table className="w-full text-left border-collapse">
                          <thead>
                            <tr className="bg-surface-container-low/10">
                              {agent.tableData.headers.map((header, i) => (
                                <th key={i} className="py-3 px-4 md:px-5 text-xs font-black font-label tracking-wider text-on-surface-variant/50 uppercase border-b border-outline-variant/10">
                                  {header}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-outline-variant/10">
                            {agent.tableData.rows.map((row, i) => (
                              <tr key={i} className="hover:bg-surface-container-low/40 transition-colors group">
                                {row.map((cell, j) => (
                                  <td key={j} className={clsx("py-3 md:py-4 px-4 md:px-5 text-sm", j === 0 ? "font-medium text-on-surface" : "text-on-surface-variant font-mono group-hover:text-primary transition-colors")}>
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

export default function AnalysisDashboard({ query, onBack }: AnalysisDashboardProps) {
  const { data, isStreaming } = useAnalysisStream(query);
  const [selectedAgent, setSelectedAgent] = useState<AgentAnalysis | null>(null);

  if (!data) {
    return (
      <div className="pt-32 pb-24 px-6 md:px-12 flex flex-col items-center justify-center min-h-screen w-full max-w-[1600px] mx-auto">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
          className="w-12 h-12 border-4 border-primary/20 border-t-primary rounded-full"
        />
        <p className="mt-4 text-on-surface-variant font-medium">Initializing AlphaMesh Agents...</p>
      </div>
    );
  }

  return (
    <motion.main
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 1.02 }}
      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
      className="pt-24 md:pt-28 pb-32 md:pb-36 px-4 md:px-8 w-full max-w-[1600px] mx-auto flex-grow"
    >
      {onBack && (
        <button 
          onClick={onBack}
          className="mb-6 flex items-center gap-2 text-on-surface-variant hover:text-on-surface transition-colors font-medium text-sm md:text-base py-2 px-4 rounded-full hover:bg-surface-container-low w-fit"
        >
          <ArrowLeft className="w-4 h-4 md:w-5 md:h-5" />
          Back to History
        </button>
      )}
      <div className="bg-surface-container-lowest rounded-3xl md:rounded-[3rem] p-5 md:p-12 shadow-[0_20px_40px_rgba(0,0,0,0.06)] border border-outline-variant/20 relative">
        
        {/* Header Query Info */}
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

        {/* Header Info */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-end mb-8 md:mb-14 gap-4 md:gap-6">
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
                {data.currentPrice?.toFixed(2)}
              </span>
              <div className="flex flex-col">
                <span className="text-primary font-black flex items-center text-sm md:text-xl">
                  <TrendingUp className="w-4 h-4 md:w-5 md:h-5 mr-1" />
                  +{data.priceChange?.toFixed(2)} ({data.priceChangePercent?.toFixed(2)}%)
                </span>
                <span className="text-on-surface-variant/60 text-[8px] md:text-[10px] font-bold font-label uppercase tracking-[0.2em] mt-0.5 md:mt-1">
                  {data.marketStatus}
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
        </div>

        {/* Layout Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 md:gap-10">
          {/* Left Column: Chart and Agents */}
          <div className="lg:col-span-8 space-y-6 md:space-y-10">
            {/* Chart Area */}
            <section>
              <div className="relative h-[250px] md:h-[360px] w-full bg-surface/80 rounded-2xl md:rounded-[2.5rem] overflow-hidden group border border-outline-variant/20">
                {data.chartData && data.chartData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <RechartsLineChart data={data.chartData}>
                      <defs>
                        <linearGradient id="chartFill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#006e01" stopOpacity={0.15} />
                          <stop offset="100%" stopColor="#006e01" stopOpacity={0} />
                        </linearGradient>
                      </defs>
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
                ) : (
                  <div className="w-full h-full flex items-center justify-center">
                    <motion.div
                      animate={{ opacity: [0.5, 1, 0.5] }}
                      transition={{ repeat: Infinity, duration: 1.5 }}
                      className="text-on-surface-variant/50 font-medium"
                    >
                      Loading Chart Data...
                    </motion.div>
                  </div>
                )}
              </div>
            </section>

            {/* Agent Analysis Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-10 items-stretch">
              {/* Agent 1: News */}
              {data.agents && data.agents[0] ? (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="relative flex flex-col h-full"
                >
                  <div className="absolute -top-4 left-0 bg-surface-container-low h-8 md:h-10 w-40 md:w-52 z-10 flex items-center px-5 md:px-7 border-t border-l border-outline-variant/20" style={{ clipPath: 'polygon(0% 0%, 75% 0%, 85% 100%, 0% 100%)' }}>
                    <span className="text-[8px] md:text-[9px] font-black font-label tracking-[0.2em] text-primary uppercase">
                      {data.agents[0].category}
                    </span>
                  </div>
                  <div 
                    onClick={() => setSelectedAgent(data.agents[0])}
                    className="bg-surface rounded-tr-2xl rounded-b-2xl md:rounded-tr-[2.5rem] md:rounded-b-[2.5rem] p-5 pt-8 md:p-8 md:pt-12 h-full flex flex-col transition-all border border-outline-variant/20 shadow-[0_4px_6px_-1px_rgba(0,0,0,0.03),0_20px_30px_-5px_rgba(0,0,0,0.06)] hover:shadow-2xl duration-500 cursor-pointer group/card"
                  >
                    <div className="flex justify-between items-start mb-4 md:mb-6">
                      <div>
                        <h2 className="font-headline text-base md:text-lg font-extrabold mb-1 text-on-surface">
                          {data.agents[0].name}
                        </h2>
                      </div>
                      <div className="bg-primary/10 p-2 md:p-2.5 rounded-lg md:rounded-xl">
                        <FileText className="w-4 h-4 md:w-5 md:h-5 text-primary" />
                      </div>
                    </div>
                    <div className="space-y-3 md:space-y-4 flex-grow">
                      <div className="space-y-1.5 md:space-y-2">
                        <div className="flex justify-between text-[8px] md:text-[9px] font-black font-label uppercase tracking-widest text-on-surface-variant/60">
                          <span>Recent Catalyst</span>
                          <span>{data.agents[0].recentCatalyst.timeAgo}</span>
                        </div>
                        <h3 className="font-bold text-sm md:text-base leading-tight text-on-surface">
                          {data.agents[0].recentCatalyst.title}
                        </h3>
                        <p className="text-[10px] md:text-xs text-on-surface-variant leading-relaxed font-medium">
                          {data.agents[0].recentCatalyst.description}
                        </p>
                      </div>
                      <div className="space-y-2 md:space-y-3 border-t border-outline-variant/20 pt-3 md:pt-4">
                        <div className="flex justify-between text-[8px] md:text-[9px] font-black font-label uppercase tracking-widest text-on-surface-variant/60">
                          <span>Sentiment Score</span>
                          <span className="text-primary font-black">{data.agents[0].sentiment.label}</span>
                        </div>
                        <div className="h-1.5 md:h-2 w-full bg-surface-container-highest rounded-full overflow-hidden">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${data.agents[0].sentiment.score}%` }}
                            transition={{ duration: 1, delay: 0.5 }}
                            className="h-full bg-primary rounded-full shadow-[0_0_10px_rgba(0,110,1,0.3)]"
                          />
                        </div>
                      </div>
                    </div>
                    <button className="mt-6 md:mt-8 flex items-center justify-between w-full group/btn text-primary font-bold text-[10px] md:text-xs border-t border-outline-variant/20 pt-3 md:pt-4">
                      <span>Full Report</span>
                      <ArrowRight className="w-3.5 h-3.5 md:w-4 md:h-4 group-hover/btn:translate-x-1 transition-transform" />
                    </button>
                  </div>
                </motion.div>
              ) : (
                <div className="h-48 md:h-64 bg-surface/50 rounded-2xl md:rounded-[2.5rem] border border-outline-variant/10 animate-pulse" />
              )}

              {/* Agent 2: Fundamental */}
              {data.agents && data.agents[1] ? (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="relative flex flex-col h-full"
                >
                  <div className="absolute -top-4 left-0 bg-surface-container-low h-8 md:h-10 w-40 md:w-52 z-10 flex items-center px-5 md:px-7 border-t border-l border-outline-variant/20" style={{ clipPath: 'polygon(0% 0%, 75% 0%, 85% 100%, 0% 100%)' }}>
                    <span className="text-[8px] md:text-[9px] font-black font-label tracking-[0.2em] text-primary uppercase">
                      {data.agents[1].category}
                    </span>
                  </div>
                  <div 
                    onClick={() => setSelectedAgent(data.agents[1])}
                    className="bg-surface rounded-tr-2xl rounded-b-2xl md:rounded-tr-[2.5rem] md:rounded-b-[2.5rem] p-5 pt-8 md:p-8 md:pt-12 h-full flex flex-col transition-all border border-outline-variant/20 shadow-[0_4px_6px_-1px_rgba(0,0,0,0.03),0_20px_30px_-5px_rgba(0,0,0,0.06)] hover:shadow-2xl duration-500 cursor-pointer group/card"
                  >
                    <div className="flex justify-between items-start mb-4 md:mb-6">
                      <div>
                        <h2 className="font-headline text-base md:text-lg font-extrabold mb-1 text-on-surface">
                          {data.agents[1].name}
                        </h2>
                      </div>
                      <div className="bg-primary/10 p-2 md:p-2.5 rounded-lg md:rounded-xl">
                        <BarChart2 className="w-4 h-4 md:w-5 md:h-5 text-primary" />
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-3 md:gap-4 flex-grow content-start">
                      {data.agents[1].metrics?.map((m, i) => (
                        <div key={i} className="bg-surface-container-lowest p-3 md:p-4 rounded-lg md:rounded-xl border border-outline-variant/20 shadow-sm">
                          <span className="text-[7px] md:text-[8px] font-black font-label tracking-[0.15em] text-on-surface-variant/60 uppercase block">
                            {m.label}
                          </span>
                          <span className="text-base md:text-xl font-headline font-black text-on-surface">
                            {m.value}
                          </span>
                        </div>
                      ))}
                      {data.agents[1].quote && (
                        <div className="col-span-2 bg-surface-container-lowest p-3 md:p-4 rounded-lg md:rounded-xl border border-outline-variant/20 shadow-sm">
                          <p className="text-[10px] md:text-[11px] text-on-surface-variant leading-relaxed italic font-medium">
                            {data.agents[1].quote}
                          </p>
                        </div>
                      )}
                    </div>
                    <button className="mt-6 md:mt-8 flex items-center justify-between w-full group/btn text-primary font-bold text-[10px] md:text-xs border-t border-outline-variant/20 pt-3 md:pt-4">
                      <span>Financial Deep-Dive</span>
                      <ArrowRight className="w-3.5 h-3.5 md:w-4 md:h-4 group-hover/btn:translate-x-1 transition-transform" />
                    </button>
                  </div>
                </motion.div>
              ) : (
                <div className="h-48 md:h-64 bg-surface/50 rounded-2xl md:rounded-[2.5rem] border border-outline-variant/10 animate-pulse" />
              )}
            </div>
          </div>

          {/* Right Column: Summary of Findings */}
          <div className="lg:col-span-4 flex flex-col h-full">
            <div className="bg-surface-container-lowest rounded-2xl md:rounded-[2.5rem] p-6 md:p-10 h-full shadow-[0_4px_6px_-1px_rgba(0,0,0,0.03),0_20px_30px_-5px_rgba(0,0,0,0.06)] flex flex-col border border-outline-variant/20 relative overflow-hidden">
              {/* Aesthetic Accent */}
              <div className="absolute -top-12 -right-12 md:-top-24 md:-right-24 w-32 h-32 md:w-64 md:h-64 bg-primary/5 rounded-full blur-[50px] md:blur-[100px]"></div>
              
              <div className="relative z-10 flex flex-col h-full">
                <div className="flex items-center gap-2 md:gap-3 mb-6 md:mb-10 justify-between w-full">
                  <h2 className="font-headline text-lg md:text-xl font-bold tracking-tight text-on-surface">
                    Summary of Findings
                  </h2>
                  <div className="w-8 h-8 md:w-10 md:h-10 rounded-lg md:rounded-xl bg-primary/10 flex items-center justify-center border border-primary/10 ml-auto">
                    <Sparkles className="w-4 h-4 md:w-5 md:h-5 text-primary" />
                  </div>
                </div>

                <div className="flex-grow space-y-6 md:space-y-10">
                  <div>
                    <h3 className="text-[9px] md:text-[10px] font-black uppercase tracking-[0.25em] text-primary mb-2 md:mb-4">
                      Core Narrative
                    </h3>
                    <p className="text-on-surface-variant text-xs md:text-sm leading-relaxed font-medium min-h-[60px] md:min-h-[80px]">
                      {data.summary?.coreNarrative || (
                        <span className="animate-pulse text-on-surface-variant/40">Synthesizing data...</span>
                      )}
                    </p>
                  </div>

                  {data.summary?.agentConsensus && data.summary.agentConsensus.length > 0 && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="space-y-4 md:space-y-6"
                    >
                      <h3 className="text-[9px] md:text-[10px] font-black uppercase tracking-[0.25em] text-primary">
                        Agent Consensus
                      </h3>
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
                              <p className="text-[10px] md:text-[11px] text-on-surface-variant/80 leading-tight">
                                {consensus.description}
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </motion.div>
                  )}

                  {data.summary?.verdict?.label && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="pt-4 md:pt-6 border-t border-outline-variant/10"
                    >
                      <div className="flex items-center justify-between mb-1.5 md:mb-2">
                        <span className="text-[9px] md:text-[10px] font-black uppercase tracking-widest text-on-surface-variant/60">
                          AlphaMesh Verdict
                        </span>
                        <span className="text-xs md:text-sm font-black text-primary">
                          {data.summary.verdict.label}
                        </span>
                      </div>
                      <p className="text-[10px] md:text-xs text-on-surface-variant/80 font-medium leading-relaxed">
                        {data.summary.verdict.description}
                      </p>
                    </motion.div>
                  )}
                </div>

                <button
                  disabled={isStreaming}
                  className={clsx(
                    "mt-8 md:mt-12 w-full py-4 md:py-5 rounded-xl md:rounded-2xl font-black text-[10px] md:text-xs flex items-center justify-center gap-1.5 md:gap-2 group shadow-lg transition-all",
                    isStreaming
                      ? "bg-surface-container-high text-on-surface-variant/50 cursor-not-allowed"
                      : "bg-inverse-surface text-inverse-on-surface hover:bg-inverse-surface/90"
                  )}
                >
                  EXPORT REPORT
                  <Download className="w-3.5 h-3.5 md:w-4 md:h-4 group-hover:translate-y-0.5 transition-transform" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Agent Analysis Modal */}
      <AnimatePresence>
        {selectedAgent && (
          <AgentModal agent={selectedAgent} onClose={() => setSelectedAgent(null)} />
        )}
      </AnimatePresence>
    </motion.main>
  );
}
