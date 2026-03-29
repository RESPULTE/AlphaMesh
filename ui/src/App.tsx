import { useState } from 'react';
import { AnimatePresence } from 'motion/react';
import TopNav from './components/TopNav';
import BottomNav from './components/BottomNav';
import Chat from './components/Chat';
import Portfolio from './components/Portfolio';
import History from './components/History';
import ChatInput from './components/ChatInput';

export default function App() {
  const [currentTab, setCurrentTab] = useState('Chat');
  const [analysisQuery, setAnalysisQuery] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);

  const handleAnalyze = (query: string) => {
    setAnalysisQuery(query);
    setCurrentTab('History');
    // Simulate streaming state for the chat input
    setIsStreaming(true);
    setTimeout(() => setIsStreaming(false), 5000); // Match the mock stream duration
  };

  const handleSetTab = (tab: string) => {
    if (tab === 'History' && currentTab === 'History') {
      setAnalysisQuery(null);
    }
    setCurrentTab(tab);
  };

  return (
    <div className="min-h-screen bg-surface text-on-surface font-body selection:bg-primary-container/30 flex flex-col pb-20 md:pb-0">
      <TopNav currentTab={currentTab} setTab={handleSetTab} />
      
      <div className="flex-grow flex flex-col relative">
        <AnimatePresence mode="wait">
          {currentTab === 'Chat' && <Chat key="chat" onAnalyze={handleAnalyze} />}
          {currentTab === 'Portfolio' && <Portfolio key="portfolio" />}
          {currentTab === 'History' && (
            <History 
              key="history" 
              onAnalyze={handleAnalyze} 
              query={analysisQuery}
              onClearQuery={() => setAnalysisQuery(null)}
            />
          )}
        </AnimatePresence>
      </div>

      {currentTab === 'History' && analysisQuery && (
        <ChatInput onAnalyze={handleAnalyze} isStreaming={isStreaming} />
      )}
      
      <BottomNav currentTab={currentTab} setTab={handleSetTab} />
    </div>
  );
}
