import React, { useState } from 'react';
import { Plus, ArrowUp } from 'lucide-react';

interface ChatInputProps {
  onAnalyze: (query: string) => void;
  isStreaming: boolean;
}

export default function ChatInput({ onAnalyze, isStreaming }: ChatInputProps) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !isStreaming) {
      onAnalyze(query.trim());
      setQuery('');
    }
  };

  return (
    <div className="fixed bottom-[72px] md:bottom-0 left-0 right-0 p-4 md:p-8 z-[60] pointer-events-none">
      <div className="max-w-3xl mx-auto w-full pointer-events-auto">
        <form
          onSubmit={handleSubmit}
          className="bg-surface-container-lowest/80 backdrop-blur-2xl border border-outline-variant/20 rounded-[2.5rem] shadow-[0_4px_6px_-1px_rgba(0,0,0,0.03),0_20px_30px_-5px_rgba(0,0,0,0.06)] p-2 md:p-3 flex items-center gap-2 md:gap-4"
        >
          <button
            type="button"
            className="p-2 md:p-3.5 hover:bg-surface-container-low rounded-2xl text-on-surface-variant/60 transition-colors"
          >
            <Plus className="w-5 h-5" />
          </button>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={isStreaming}
            placeholder="Ask AlphaMesh about AAPL..."
            className="flex-grow bg-transparent border-none focus:ring-0 text-sm font-semibold py-2 px-1 text-on-surface placeholder:text-on-surface-variant/50 outline-none"
          />
          <button
            type="submit"
            disabled={!query.trim() || isStreaming}
            className="bg-primary text-on-primary w-10 h-10 md:w-12 md:h-12 rounded-[1.25rem] flex items-center justify-center hover:shadow-[0_0_30px_rgba(0,200,5,0.35)] hover:scale-[1.02] transition-all active:scale-95 shadow-sm disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
          >
            <ArrowUp className="w-5 h-5" />
          </button>
        </form>
        <p className="hidden md:block text-[9px] text-center text-on-surface-variant/50 mt-5 font-black uppercase tracking-[0.25em]">
          AlphaMesh Intelligence Unit • Ver 2.4.0
        </p>
      </div>
    </div>
  );
}
