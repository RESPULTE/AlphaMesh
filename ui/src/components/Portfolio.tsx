import { useCallback, useEffect, useMemo, useState } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import { Filter, Loader2, Plus, Search, TrendingDown, TrendingUp, X } from 'lucide-react';
import clsx from 'clsx';
import { useAuth } from '../auth/AuthContext';
import type {
  MarketQuote,
  PortfolioHolding,
  PortfolioResponse,
  TickerSearchResponse,
  TickerSearchResult,
  UpsertPortfolioHoldingRequest,
} from '../types/api';

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 2,
  }).format(value);
}

function formatShares(value: number): string {
  if (Number.isInteger(value)) {
    return value.toLocaleString('en-US');
  }
  return value.toLocaleString('en-US', { maximumFractionDigits: 4 });
}

function formatPercent(value: number): string {
  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(2)}%`;
}

interface PortfolioRow {
  holding: PortfolioHolding;
  quote?: MarketQuote;
  marketValue: number;
  dayChangeValue: number;
  dayChangePercent: number;
}

export default function Portfolio() {
  const { authFetch } = useAuth();
  const [holdings, setHoldings] = useState<PortfolioHolding[]>([]);
  const [quotesByTicker, setQuotesByTicker] = useState<Record<string, MarketQuote>>({});
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshingQuotes, setIsRefreshingQuotes] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<TickerSearchResult[]>([]);
  const [selectedTicker, setSelectedTicker] = useState<TickerSearchResult | null>(null);
  const [sharesInput, setSharesInput] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  const loadPortfolio = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await authFetch('/api/v1/portfolio');
      if (!response.ok) {
        throw new Error('Unable to load your portfolio holdings.');
      }
      const payload = (await response.json()) as PortfolioResponse;
      setHoldings(payload.holdings ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to load your portfolio holdings.');
      setHoldings([]);
    } finally {
      setIsLoading(false);
    }
  }, [authFetch]);

  useEffect(() => {
    void loadPortfolio();
  }, [loadPortfolio]);

  useEffect(() => {
    let cancelled = false;

    const loadQuotes = async () => {
      if (holdings.length === 0) {
        setQuotesByTicker({});
        setIsRefreshingQuotes(false);
        return;
      }

      setIsRefreshingQuotes(true);
      const quoteEntries = await Promise.all(
        holdings.map(async (holding) => {
          try {
            const response = await authFetch(`/api/market/${encodeURIComponent(holding.ticker)}/quote`);
            if (!response.ok) {
              return [holding.ticker, null] as const;
            }
            const quote = (await response.json()) as MarketQuote;
            return [holding.ticker, quote] as const;
          } catch {
            return [holding.ticker, null] as const;
          }
        }),
      );

      if (cancelled) {
        return;
      }

      const next: Record<string, MarketQuote> = {};
      for (const [ticker, quote] of quoteEntries) {
        if (quote) {
          next[ticker] = quote;
        }
      }
      setQuotesByTicker(next);
      setIsRefreshingQuotes(false);
    };

    void loadQuotes();
    return () => {
      cancelled = true;
    };
  }, [authFetch, holdings]);

  useEffect(() => {
    if (!isAddModalOpen) {
      return;
    }

    const trimmed = searchQuery.trim();
    if (trimmed.length < 2) {
      setSearchResults([]);
      setSearchError(null);
      setIsSearching(false);
      return;
    }

    const controller = new AbortController();
    const timeoutId = window.setTimeout(async () => {
      setIsSearching(true);
      setSearchError(null);
      try {
        const response = await authFetch(
          `/api/v1/tickers/search?q=${encodeURIComponent(trimmed)}&limit=8`,
          { signal: controller.signal }
        );
        if (!response.ok) {
          throw new Error('Ticker search failed.');
        }
        const payload = (await response.json()) as TickerSearchResponse;
        setSearchResults(payload.results ?? []);
      } catch (err) {
        if (controller.signal.aborted) {
          return;
        }
        setSearchError(err instanceof Error ? err.message : 'Ticker search failed.');
        setSearchResults([]);
      } finally {
        if (!controller.signal.aborted) {
          setIsSearching(false);
        }
      }
    }, 300);

    return () => {
      controller.abort();
      window.clearTimeout(timeoutId);
    };
  }, [authFetch, isAddModalOpen, searchQuery]);

  const rows = useMemo<PortfolioRow[]>(() => {
    return holdings.map((holding) => {
      const quote = quotesByTicker[holding.ticker];
      const currentPrice = quote?.currentPrice ?? 0;
      const dayChange = quote?.priceChange ?? 0;
      return {
        holding,
        quote,
        marketValue: currentPrice * holding.shares,
        dayChangeValue: dayChange * holding.shares,
        dayChangePercent: quote?.priceChangePercent ?? 0,
      };
    });
  }, [holdings, quotesByTicker]);

  const totals = useMemo(() => {
    return rows.reduce(
      (acc, row) => ({
        marketValue: acc.marketValue + row.marketValue,
        dayChangeValue: acc.dayChangeValue + row.dayChangeValue,
      }),
      { marketValue: 0, dayChangeValue: 0 },
    );
  }, [rows]);

  const dayChangePercent =
    totals.marketValue > 0 ? (totals.dayChangeValue / totals.marketValue) * 100 : 0;

  const openAddModal = () => {
    setSearchQuery('');
    setSearchResults([]);
    setSelectedTicker(null);
    setSharesInput('');
    setSearchError(null);
    setSaveError(null);
    setIsAddModalOpen(true);
  };

  const closeAddModal = () => {
    if (!isSaving) {
      setIsAddModalOpen(false);
    }
  };

  const handleSelectTicker = (result: TickerSearchResult) => {
    setSelectedTicker(result);
    setSearchQuery(`${result.ticker} - ${result.company_name}`);
    setSearchResults([]);
    setSearchError(null);
  };

  const handleSaveHolding = async () => {
    if (!selectedTicker) {
      setSaveError('Select a ticker before saving.');
      return;
    }

    const parsedShares = Number(sharesInput);
    if (!Number.isFinite(parsedShares) || parsedShares <= 0) {
      setSaveError('Enter a valid share amount greater than zero.');
      return;
    }

    setIsSaving(true);
    setSaveError(null);

    const payload: UpsertPortfolioHoldingRequest = {
      ticker: selectedTicker.ticker,
      company_name: selectedTicker.company_name,
      exchange: selectedTicker.exchange ?? null,
      asset_type: selectedTicker.asset_type,
      shares: parsedShares,
    };

    try {
      const response = await authFetch('/api/v1/portfolio/holding', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorPayload = await response.json().catch(() => null);
        const detail = errorPayload && typeof errorPayload.detail === 'string'
          ? errorPayload.detail
          : 'Failed to save holding.';
        throw new Error(detail);
      }

      const updated = (await response.json()) as PortfolioResponse;
      setHoldings(updated.holdings ?? []);
      setIsAddModalOpen(false);
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : 'Failed to save holding.');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <>
      <motion.main
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
        className="pt-24 md:pt-28 pb-24 md:pb-32 px-4 md:px-6 max-w-7xl mx-auto w-full"
      >
        <header className="mb-10 md:mb-16 flex flex-col items-center text-center">
          <span className="font-label text-[10px] md:text-[11px] font-bold uppercase tracking-widest text-primary mb-1.5 md:mb-2 block">
            Total Portfolio Value
          </span>
          <h1 className="font-headline text-4xl sm:text-5xl md:text-7xl font-extrabold tracking-tighter text-on-surface mb-2 md:mb-4">
            {formatCurrency(totals.marketValue)}
          </h1>
          <div
            className={clsx(
              'flex items-center justify-center gap-1.5 md:gap-2 font-bold text-xs sm:text-sm md:text-base mb-6 md:mb-8',
              totals.dayChangeValue >= 0 ? 'text-secondary' : 'text-error',
            )}
          >
            {totals.dayChangeValue >= 0 ? (
              <TrendingUp className="w-4 h-4 md:w-5 md:h-5 shrink-0" />
            ) : (
              <TrendingDown className="w-4 h-4 md:w-5 md:h-5 shrink-0" />
            )}
            <span>
              {`${totals.dayChangeValue >= 0 ? '+' : ''}${formatCurrency(totals.dayChangeValue)} (${formatPercent(dayChangePercent)}) Today`}
            </span>
            {isRefreshingQuotes && <Loader2 className="w-4 h-4 animate-spin ml-1" />}
          </div>

          <div className="flex gap-2 md:gap-3 w-full md:w-auto justify-center">
            <button
              onClick={openAddModal}
              className="flex-1 md:flex-none flex items-center justify-center gap-1.5 md:gap-2 bg-surface-container-high px-4 py-3 md:px-6 md:py-3 rounded-xl font-bold text-xs md:text-sm hover:bg-surface-container-highest transition-all active:scale-95 text-on-surface"
            >
              <Plus className="w-4 h-4 md:w-[18px] md:h-[18px]" />
              Add Asset
            </button>
            <button className="flex-1 md:flex-none flex items-center justify-center gap-1.5 md:gap-2 bg-primary text-on-primary px-4 py-3 md:px-6 md:py-3 rounded-xl font-bold text-xs md:text-sm hover:opacity-90 transition-all shadow-lg shadow-primary/20 active:scale-95">
              <Filter className="w-4 h-4 md:w-[18px] md:h-[18px]" />
              Filter
            </button>
          </div>
        </header>

        {error && (
          <div className="mb-6 rounded-xl border border-error/30 bg-error/10 text-error px-4 py-3 text-sm">
            {error}
          </div>
        )}

        <section className="space-y-4">
          <div className="flex items-center justify-between px-4 mb-4">
            <h2 className="font-label text-[11px] font-bold uppercase tracking-widest text-outline">
              Tracked Assets
            </h2>
            <div className="flex gap-8">
              <span className="hidden md:block font-label text-[11px] font-bold uppercase tracking-widest text-outline w-32 text-right">
                Position Amount
              </span>
              <span className="font-label text-[11px] font-bold uppercase tracking-widest text-outline w-32 text-right">
                Total Market Value
              </span>
            </div>
          </div>

          {isLoading && (
            <div className="bg-surface-container-lowest border border-outline-variant/20 rounded-xl p-6 text-outline text-sm">
              Loading portfolio...
            </div>
          )}

          {!isLoading && rows.length === 0 && (
            <div className="bg-surface-container-lowest border border-outline-variant/20 rounded-xl p-6 text-outline text-sm">
              No assets tracked yet. Click <span className="text-on-surface font-semibold">Add Asset</span> to set your first holding.
            </div>
          )}

          {rows.map((row, i) => (
            <motion.div
              key={row.holding.ticker}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.06 }}
              className="group relative bg-surface-container-lowest hover:bg-surface-container-low transition-all duration-300 rounded-xl overflow-hidden border border-transparent hover:border-outline-variant/20 shadow-[0_20px_40px_rgba(26,28,28,0.04)]"
            >
              <div className="flex items-center justify-between p-4 md:p-6">
                <div className="flex items-center gap-3 md:gap-6">
                  <div className="w-10 h-10 md:w-12 md:h-12 bg-surface-container-high rounded-full flex items-center justify-center shrink-0">
                    <span className="font-label text-xs md:text-sm text-primary font-bold">
                      {row.holding.ticker.slice(0, 3)}
                    </span>
                  </div>
                  <div>
                    <h3 className="font-headline text-base md:text-lg font-bold text-on-surface">
                      {row.holding.company_name}
                    </h3>
                    <span className="font-label text-[10px] md:text-xs font-semibold text-outline tracking-wider uppercase">
                      {row.holding.ticker} | {row.holding.exchange || 'UNKNOWN EXCHANGE'}
                    </span>
                  </div>
                </div>

                <div className="flex items-center gap-4 md:gap-12">
                  <div className="hidden md:block text-right w-32">
                    <div className="font-headline text-lg font-bold text-on-surface">
                      {formatShares(row.holding.shares)}
                    </div>
                    <div className="font-bold text-sm text-outline">
                      Shares
                    </div>
                  </div>
                  <div className="text-right w-24 md:w-32">
                    <div className="font-headline text-base md:text-lg font-bold text-on-surface">
                      {formatCurrency(row.marketValue)}
                    </div>
                    <div
                      className={clsx(
                        'font-bold text-xs md:text-sm',
                        row.dayChangeValue >= 0 ? 'text-secondary' : 'text-error',
                      )}
                    >
                      {formatPercent(row.dayChangePercent)}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </section>

        <footer className="mt-20 text-center">
          <p className="text-outline font-medium text-sm">
            Use <span className="text-on-surface font-bold">Add Asset</span> to set your holdings and keep this portfolio synced.
          </p>
        </footer>
      </motion.main>

      <AnimatePresence>
        {isAddModalOpen && (
          <div className="fixed inset-0 z-[110] flex items-end md:items-center justify-center p-0 md:p-8">
            <motion.button
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={closeAddModal}
              className="absolute inset-0 bg-black/60 backdrop-blur-sm"
              aria-label="Close add asset modal"
            />
            <motion.div
              initial={{ opacity: 0, y: 24, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 24, scale: 0.98 }}
              transition={{ duration: 0.22, ease: 'easeOut' }}
              className="relative w-full md:max-w-2xl bg-surface border border-outline-variant/30 rounded-t-3xl md:rounded-3xl shadow-2xl overflow-hidden"
            >
              <div className="px-5 md:px-8 py-4 md:py-6 border-b border-outline-variant/20 bg-surface-container-lowest/80">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-label text-[10px] md:text-xs uppercase tracking-[0.2em] text-primary mb-1">
                      Add Asset
                    </p>
                    <h2 className="font-headline text-xl md:text-2xl font-extrabold text-on-surface">
                      Set Your Holdings
                    </h2>
                  </div>
                  <button
                    onClick={closeAddModal}
                    className="p-2 rounded-full hover:bg-surface-container-high transition-colors text-on-surface-variant"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>
              </div>

              <div className="p-5 md:p-8 space-y-5">
                <label className="block">
                  <span className="text-xs font-label tracking-wider uppercase text-outline mb-2 block">
                    Search ticker
                  </span>
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-outline" />
                    <input
                      value={searchQuery}
                      onChange={(e) => {
                        setSearchQuery(e.target.value);
                        setSelectedTicker(null);
                        setSaveError(null);
                      }}
                      placeholder="Type ticker or company name (e.g., AAPL, Apple)"
                      className="w-full bg-surface-container-low border border-outline-variant/30 rounded-xl pl-10 pr-4 py-3 text-sm md:text-base text-on-surface placeholder:text-on-surface-variant/50 outline-none focus:border-primary/60"
                    />
                  </div>
                </label>

                <div className="min-h-20">
                  {isSearching && (
                    <div className="text-sm text-outline flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Searching tickers...
                    </div>
                  )}

                  {searchError && (
                    <div className="text-sm text-error bg-error/10 border border-error/30 rounded-lg px-3 py-2">
                      {searchError}
                    </div>
                  )}

                  {!isSearching && !selectedTicker && searchQuery.trim().length >= 2 && searchResults.length > 0 && (
                    <div className="rounded-xl border border-outline-variant/20 overflow-hidden max-h-56 overflow-y-auto">
                      {searchResults.map((result) => (
                        <button
                          key={result.ticker}
                          onClick={() => handleSelectTicker(result)}
                          className="w-full text-left px-4 py-3 bg-surface-container-lowest hover:bg-surface-container-low transition-colors border-b border-outline-variant/10 last:border-b-0"
                        >
                          <div className="font-semibold text-on-surface text-sm">
                            {result.ticker} - {result.company_name}
                          </div>
                          <div className="text-xs text-outline uppercase tracking-wider mt-1">
                            {result.exchange || 'Unknown'} | {result.asset_type}
                          </div>
                        </button>
                      ))}
                    </div>
                  )}

                  {!isSearching && !selectedTicker && searchQuery.trim().length >= 2 && searchResults.length === 0 && !searchError && (
                    <div className="text-sm text-outline">
                      No supported equity or ETF tickers found for this query.
                    </div>
                  )}

                  {selectedTicker && (
                    <div className="rounded-xl border border-primary/30 bg-primary/10 px-4 py-3">
                      <div className="font-semibold text-on-surface text-sm md:text-base">
                        {selectedTicker.ticker} - {selectedTicker.company_name}
                      </div>
                      <div className="text-xs text-outline uppercase tracking-wider mt-1">
                        {selectedTicker.exchange || 'Unknown'} | {selectedTicker.asset_type}
                      </div>
                    </div>
                  )}
                </div>

                <label className="block">
                  <span className="text-xs font-label tracking-wider uppercase text-outline mb-2 block">
                    Number of shares
                  </span>
                  <input
                    type="number"
                    inputMode="decimal"
                    min="0"
                    step="any"
                    value={sharesInput}
                    onChange={(e) => {
                      setSharesInput(e.target.value);
                      setSaveError(null);
                    }}
                    placeholder="e.g. 125"
                    className="w-full bg-surface-container-low border border-outline-variant/30 rounded-xl px-4 py-3 text-sm md:text-base text-on-surface placeholder:text-on-surface-variant/50 outline-none focus:border-primary/60"
                  />
                </label>

                {saveError && (
                  <div className="text-sm text-error bg-error/10 border border-error/30 rounded-lg px-3 py-2">
                    {saveError}
                  </div>
                )}

                <div className="flex gap-3 pt-2">
                  <button
                    onClick={closeAddModal}
                    className="flex-1 bg-surface-container-high text-on-surface rounded-xl py-3 text-sm font-semibold hover:bg-surface-container-highest transition-colors"
                    disabled={isSaving}
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleSaveHolding}
                    className="flex-1 bg-primary text-on-primary rounded-xl py-3 text-sm font-bold hover:opacity-90 transition-all shadow-lg shadow-primary/20 disabled:opacity-60"
                    disabled={isSaving}
                  >
                    {isSaving ? (
                      <span className="inline-flex items-center gap-2">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Saving...
                      </span>
                    ) : (
                      'Save Holding'
                    )}
                  </button>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </>
  );
}
