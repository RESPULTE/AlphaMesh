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
  /** Conversation ID to auto-expand in History when navigating from Chat */
  const [openConversationId, setOpenConversationId] = useState<string | null>(null);

  const handleAnalyze = (query: string) => {
    setAnalysisQuery(query);
    setOpenConversationId(null); // clear any pending deep-link
    setCurrentTab('History');
    // Simulate streaming state for the chat input
    setIsStreaming(true);
    setTimeout(() => setIsStreaming(false), 5000); // Match the mock stream duration
  };

  const handleOpenConversation = (conversationId: string) => {
    setAnalysisQuery(null);        // don't start a new analysis
    setOpenConversationId(conversationId);
    setCurrentTab('History');
  };

  const handleSetTab = (tab: string) => {
    if (tab === 'History' && currentTab === 'History') {
      setAnalysisQuery(null);
    }
    // Reset the deep-link when manually switching tabs
    if (tab !== 'History') {
      setOpenConversationId(null);
    }
    setCurrentTab(tab);
  };

  return (
    <div className="min-h-screen bg-surface text-on-surface font-body selection:bg-primary-container/30 flex flex-col pb-20 md:pb-0">
      <TopNav currentTab={currentTab} setTab={handleSetTab} />
      
      <div className="flex-grow flex flex-col relative">
        <AnimatePresence mode="wait">
          {currentTab === 'Chat' && (
            <Chat
              key="chat"
              onAnalyze={handleAnalyze}
              onOpenConversation={handleOpenConversation}
            />
          )}
          {currentTab === 'Portfolio' && <Portfolio key="portfolio" />}
          {currentTab === 'History' && (
            <History
              key="history"
              query={analysisQuery}
              onClearQuery={() => setAnalysisQuery(null)}
              initialExpandedId={openConversationId}
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
