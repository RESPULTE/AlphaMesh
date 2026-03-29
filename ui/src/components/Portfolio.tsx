import { motion } from 'motion/react';
import { Plus, Filter, TrendingUp, TrendingDown } from 'lucide-react';
import clsx from 'clsx';

export default function Portfolio() {
  const assets = [
    { name: 'Apple Inc.', ticker: 'AAPL', exchange: 'NASDAQ', price: '$189.43', change: '+1.24%', up: true, shares: '150', marketValue: '$28,414.50', logo: 'https://lh3.googleusercontent.com/aida-public/AB6AXuCAsBQSs8I26WJyZrzAUC8sv9UOIYNUq7B2FGhixHxEMGSr2YMJyfkrNrYc2PKWsTgUpaN5fZthibf68NMAeZy5gkFWVB4qI-TJNWBHEaceiGLhv4abCdWrJt8ffWMN1L56hh20ptCtesP6YzGk8ShclftH2YKvwNwBjA9a6biAElpqx35GS9mICDG5pFHdMF8kTnDl86IZQeBc5Lhbd2Xzc6EH4Ha2CoC39UXsL1dHTe3CmvkHehV-5uBEfmSd1AkVZE-ewqpkdds' },
    { name: 'Bitcoin', ticker: 'BTC', exchange: 'CRYPTO', price: '$64,210.00', change: '-2.11%', up: false, shares: '0.5', marketValue: '$32,105.00', logo: 'https://lh3.googleusercontent.com/aida-public/AB6AXuDgXyBbMgAytCfo4fBUcd1qAE9G0DY7VtYpMrMWAdsyFUV2TYUxkd0Il7Of0fNIK4xZiWzax1XhIA4v2IiBfPbTjDGD5EdZ_ZzIPoECYzpAuDS5_hBCmAZaByWbf-q45nG-ZW4b4fm67wxY7ee74ho8YMbfhlvJdq2aGb4Ex6wC4kDZIfBom2sDcJwgiiTuujVp9j2Wy1HcWrLH-RW4-2xUtwWQNxqTByEPttQGtW7zrE-F5EyZvOlkWa6eerZk-acrLCc7v0ufYvw' },
    { name: 'Tesla, Inc.', ticker: 'TSLA', exchange: 'NASDAQ', price: '$175.43', change: '+5.67%', up: true, shares: '200', marketValue: '$35,086.00', logo: 'https://lh3.googleusercontent.com/aida-public/AB6AXuCRMcTte4Bc22_9lsC0HunOFy3jq90jEHfpAufN3cgGiD7ua7oQ-f__NoLtqL8GZ1pjq7zuMC9JyeeqiA7tJ--6kUhd27ofHDuD9e5zz3Efn1LmoPPL4yiDZvGZC6JEwPocfURfzEW5t6fX08LrYcGt_tCiy0QM9lU6Hbfb6uOXm3Zf7iTi9nOVNQFHdh0AmRwn_gvfcA23B4JSsz6S5WUWTkPgx5UUQO--PIOYd4pcdrXTNIipRqbREyWqY5m7j0i2hjocfhm22ow' },
    { name: 'NVIDIA Corp.', ticker: 'NVDA', exchange: 'NASDAQ', price: '$822.79', change: '+0.89%', up: true, shares: '50', marketValue: '$41,139.50', logo: 'https://lh3.googleusercontent.com/aida-public/AB6AXuCOSGRTe9T3epxGT2ERVkw1IHRq-oxfHQvR3Ry8uXu6ju4JOhYKwPxHZUKGDgfUtZe0EfmKryVdlMHeS1xnihj25ezrjqHgjMNumGPC7uo1y08kGC9RcbC5hxvbAkjdo-dBLnvZeCHg5Ahuq9nvw1e2ki93kYN7B0omVYGbcX6aCRN5fi5K7dz4E1JPSfO-3MffYzESILnXnGNgxgV0E8BlOpUWOpHoGWfgUF6vvKV83SSFI6W6NZOUskxu6hr7b6f7UQE7AZ3Fa3s' },
  ];

  return (
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
          $136,745.00
        </h1>
        <div className="flex items-center justify-center gap-1.5 md:gap-2 text-secondary font-bold text-xs sm:text-sm md:text-base mb-6 md:mb-8">
          <TrendingUp className="w-4 h-4 md:w-5 md:h-5 shrink-0" />
          <span>+$2,410.12 (1.72%) Today</span>
        </div>

        <div className="flex gap-2 md:gap-3 w-full md:w-auto justify-center">
          <button className="flex-1 md:flex-none flex items-center justify-center gap-1.5 md:gap-2 bg-surface-container-high px-4 py-3 md:px-6 md:py-3 rounded-xl font-bold text-xs md:text-sm hover:bg-surface-container-highest transition-all active:scale-95 text-on-surface">
            <Plus className="w-4 h-4 md:w-[18px] md:h-[18px]" />
            Add Asset
          </button>
          <button className="flex-1 md:flex-none flex items-center justify-center gap-1.5 md:gap-2 bg-primary text-on-primary px-4 py-3 md:px-6 md:py-3 rounded-xl font-bold text-xs md:text-sm hover:opacity-90 transition-all shadow-lg shadow-primary/20 active:scale-95">
            <Filter className="w-4 h-4 md:w-[18px] md:h-[18px]" />
            Filter
          </button>
        </div>
      </header>

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

        {assets.map((asset, i) => (
          <motion.div
            key={asset.ticker}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            className="group relative bg-surface-container-lowest hover:bg-surface-container-low transition-all duration-300 rounded-xl overflow-hidden cursor-pointer border border-transparent hover:border-outline-variant/20 shadow-[0_20px_40px_rgba(26,28,28,0.04)]"
          >
            <div className="flex items-center justify-between p-4 md:p-6">
              <div className="flex items-center gap-3 md:gap-6">
                <div className="w-10 h-10 md:w-12 md:h-12 bg-surface-container-high rounded-full flex items-center justify-center shrink-0">
                  <img src={asset.logo} alt={asset.name} className="w-5 h-5 md:w-6 md:h-6 object-contain grayscale" referrerPolicy="no-referrer" />
                </div>
                <div>
                  <h3 className="font-headline text-base md:text-lg font-bold text-on-surface">{asset.name}</h3>
                  <span className="font-label text-[10px] md:text-xs font-semibold text-outline tracking-wider uppercase">
                    {asset.ticker} • {asset.exchange}
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-4 md:gap-12">
                <div className="hidden md:block text-right w-32">
                  <div className="font-headline text-lg font-bold text-on-surface">{asset.shares}</div>
                  <div className="font-bold text-sm text-outline">
                    Shares
                  </div>
                </div>
                <div className="text-right w-24 md:w-32">
                  <div className="font-headline text-base md:text-lg font-bold text-on-surface">{asset.marketValue}</div>
                  <div className={clsx("font-bold text-xs md:text-sm", asset.up ? "text-secondary" : "text-error")}>
                    {asset.change}
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </section>

      <footer className="mt-20 text-center">
        <p className="text-outline font-medium text-sm">
          Want to track more assets? Use the <span className="text-on-surface font-bold">Search</span> bar or <span className="text-on-surface font-bold">Import Portfolio</span>.
        </p>
      </footer>
    </motion.main>
  );
}
