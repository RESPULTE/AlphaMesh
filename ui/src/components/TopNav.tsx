import { Bell, LogOut, User } from 'lucide-react';
import { motion } from 'motion/react';
import clsx from 'clsx';

interface TopNavProps {
  currentTab: string;
  setTab: (tab: string) => void;
  userEmail: string;
  onLogout: () => void;
}

export default function TopNav({ currentTab, setTab, userEmail, onLogout }: TopNavProps) {
  const tabs = ['Chat', 'Portfolio', 'History'];

  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-surface/80 backdrop-blur-md border-b border-outline-variant/10">
      <nav className="flex items-center w-full px-6 py-4 max-w-7xl mx-auto relative">
        <div className="flex items-center gap-2 md:gap-3 text-xl md:text-2xl font-extrabold tracking-tighter text-on-surface font-headline antialiased mr-auto">
          <div className="relative w-6 h-6 md:w-8 md:h-8 flex items-center justify-center text-primary">
            <div className="absolute w-full h-full border-2 border-current rotate-45 opacity-30" />
            <div className="absolute w-[60%] h-[60%] bg-current rotate-45" style={{ clipPath: 'polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)' }} />
          </div>
          <span>AlphaMesh</span>
        </div>

        <div className="hidden md:flex items-center gap-10 font-headline antialiased tracking-tight absolute left-1/2 -translate-x-1/2">
          {tabs.map((tab) => (
            <button
              key={tab}
              onClick={() => setTab(tab)}
              className={clsx(
                'relative pb-1 font-semibold transition-colors',
                currentTab === tab
                  ? 'text-primary font-bold'
                  : 'text-on-surface-variant/60 hover:text-on-surface'
              )}
            >
              {tab}
              {currentTab === tab && (
                <motion.div
                  layoutId="activeTab"
                  className="absolute bottom-0 left-0 w-full h-0.5 bg-primary"
                  transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                />
              )}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-1 md:gap-3 ml-auto">
          <button className="p-2 hover:bg-surface-container-low rounded-lg transition-all text-on-surface-variant">
            <Bell className="w-5 h-5 md:w-5 md:h-5" />
          </button>
          <div className="hidden md:flex items-center gap-2 rounded-full border border-outline-variant/30 bg-surface-container-low px-3 py-1.5 text-xs">
            <User className="h-3.5 w-3.5 text-on-surface-variant" />
            <span className="max-w-[180px] truncate text-on-surface-variant">{userEmail}</span>
          </div>
          <button
            onClick={onLogout}
            className="inline-flex items-center gap-1 rounded-full bg-surface-container-low px-3 py-2 text-xs font-semibold text-on-surface-variant transition-all hover:bg-surface-container-high hover:text-on-surface"
          >
            <LogOut className="h-4 w-4" />
            <span className="hidden sm:inline">Sign Out</span>
          </button>
        </div>
      </nav>
    </header>
  );
}
