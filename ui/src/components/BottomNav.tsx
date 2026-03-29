import { MessageSquare, LineChart, Briefcase, History as HistoryIcon } from 'lucide-react';
import clsx from 'clsx';

interface BottomNavProps {
  currentTab: string;
  setTab: (tab: string) => void;
}

export default function BottomNav({ currentTab, setTab }: BottomNavProps) {
  const tabs = [
    { id: 'Chat', icon: MessageSquare },
    { id: 'Portfolio', icon: Briefcase },
    { id: 'History', icon: HistoryIcon },
  ];

  return (
    <nav className="md:hidden fixed bottom-0 left-0 right-0 z-50 bg-surface/90 backdrop-blur-md border-t border-outline-variant/10 pb-safe">
      <div className="flex items-center justify-around px-2 py-3">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = currentTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setTab(tab.id)}
              className={clsx(
                'flex flex-col items-center justify-center w-16 gap-1 transition-colors',
                isActive ? 'text-primary' : 'text-on-surface-variant/60 hover:text-on-surface'
              )}
            >
              <div className={clsx(
                'p-1.5 rounded-full transition-all',
                isActive ? 'bg-primary/10' : 'bg-transparent'
              )}>
                <Icon className={clsx("w-5 h-5", isActive && "fill-primary/20")} />
              </div>
              <span className={clsx(
                'text-[10px] font-medium',
                isActive ? 'font-bold' : ''
              )}>
                {tab.id}
              </span>
            </button>
          );
        })}
      </div>
    </nav>
  );
}
