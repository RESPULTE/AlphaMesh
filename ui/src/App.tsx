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
  const [dashboardConversationId, setDashboardConversationId] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  /** Conversation ID to auto-expand in History when navigating from Chat */
  const [openConversationId, setOpenConversationId] = useState<string | null>(null);

  const handleAnalyze = (query: string) => {
    setAnalysisQuery(query);
    setDashboardConversationId(null);
    setOpenConversationId(null);
    setCurrentTab('History');
    setIsStreaming(true);
    setTimeout(() => setIsStreaming(false), 5000);
  };

  const handleOpenConversation = (conversationId: string) => {
    setAnalysisQuery(null);
    setDashboardConversationId(null);
    setOpenConversationId(conversationId);
    setCurrentTab('History');
  };

  /**
   * Called when the user types a new message inside the FullConversationView modal.
   * Sets the active conversation in localStorage so the backend continues in the
   * same thread, then kicks off a fresh analysis stream.
   */
  const handleContinueConversation = (conversationId: string) => {
    setOpenConversationId(null);
    setAnalysisQuery(null);
    setDashboardConversationId(conversationId);
    setCurrentTab('History');
  };

  const handleSetTab = (tab: string) => {
    if (tab === 'History' && currentTab === 'History') {
      setAnalysisQuery(null);
      setDashboardConversationId(null);
    }
    if (tab !== 'History') {
      setOpenConversationId(null);
      setDashboardConversationId(null);
      setAnalysisQuery(null);
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
              dashboardConversationId={dashboardConversationId}
              onClearQuery={() => setAnalysisQuery(null)}
              onClearDashboardConversation={() => setDashboardConversationId(null)}
              initialExpandedId={openConversationId}
              onContinueConversation={handleContinueConversation}
            />
          )}
        </AnimatePresence>
      </div>

      {currentTab === 'History' && (analysisQuery || dashboardConversationId) && (
        <ChatInput onAnalyze={handleAnalyze} isStreaming={isStreaming} />
      )}

      <BottomNav currentTab={currentTab} setTab={handleSetTab} />
    </div>
  );
}
