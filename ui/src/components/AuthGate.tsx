import { useState } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import { ArrowRight } from 'lucide-react';
import { useAuth } from '../auth/AuthContext';
import BrandMark from './BrandMark';

type AuthMode = 'login' | 'signup';

const MODE_LABELS: Record<AuthMode, { title: string; subtitle: string; cta: string }> = {
  login: {
    title: 'Welcome back',
    subtitle: 'Access your AlphaMesh workspace.',
    cta: 'Sign In',
  },
  signup: {
    title: 'Create account',
    subtitle: 'Set up secure workspace access.',
    cta: 'Create Account',
  },
};

export default function AuthGate() {
  const { authenticate } = useAuth();
  const [mode, setMode] = useState<AuthMode>('login');
  const [email, setEmail] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const labels = MODE_LABELS[mode];

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!email.trim()) {
      setError('Enter your email to continue.');
      return;
    }

    setIsSubmitting(true);
    setError(null);
    try {
      await authenticate(mode, email);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Authentication failed.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="relative min-h-screen overflow-hidden bg-surface text-on-surface">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute -top-32 left-1/2 h-80 w-80 -translate-x-1/2 rounded-full bg-primary/15 blur-3xl" />
        <div className="absolute bottom-0 right-[-10%] h-72 w-72 rounded-full bg-tertiary/10 blur-3xl" />
        <div className="absolute left-[-8%] top-1/3 h-64 w-64 rounded-full bg-secondary/10 blur-3xl" />
      </div>

      <div className="relative mx-auto flex min-h-screen w-full max-w-6xl items-center justify-center px-5 py-10 md:px-8">
        <motion.section
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, ease: 'easeOut' }}
          className="grid w-full max-w-5xl overflow-hidden rounded-3xl border border-outline-variant/25 bg-surface-container-lowest/90 shadow-[0_36px_120px_rgba(0,0,0,0.45)] md:grid-cols-[1.1fr_1fr]"
        >
          <div className="relative flex flex-col justify-between border-b border-outline-variant/20 p-7 md:border-b-0 md:border-r md:p-10">
            <div>
              <BrandMark size="auth" />
              <p className="mt-8 font-label text-[11px] uppercase tracking-[0.22em] text-primary">Analysis Workspace</p>
              <h1 className="mt-4 font-headline text-4xl font-black tracking-tight md:text-5xl">Secure Access</h1>
              <p className="mt-4 max-w-md text-sm text-on-surface-variant md:text-base">
                Continue with your email to access conversation and portfolio context.
              </p>
            </div>
          </div>

          <div className="p-7 md:p-10">
            <div className="relative mb-7 grid grid-cols-2 rounded-xl border border-outline-variant/25 bg-surface-container-low p-1">
              <motion.div
                aria-hidden="true"
                initial={false}
                animate={{ x: mode === 'login' ? '0%' : '100%', opacity: 1 }}
                transition={{ type: 'spring', stiffness: 320, damping: 30, mass: 0.8 }}
                className="absolute left-1 top-1 h-[calc(100%-0.5rem)] w-[calc(50%-0.25rem)] rounded-lg bg-primary shadow-[0_8px_24px_rgba(0,200,5,0.25)]"
              />
              {(['login', 'signup'] as const).map((entry) => {
                const active = mode === entry;
                return (
                  <button
                    key={entry}
                    type="button"
                    onClick={() => {
                      setMode(entry);
                      setError(null);
                    }}
                    className={`relative z-10 rounded-lg px-3 py-2 text-sm font-semibold transition-all duration-300 ${
                      active
                        ? 'text-on-primary opacity-100'
                        : 'text-on-surface-variant opacity-60 hover:opacity-100 hover:text-on-surface'
                    }`}
                  >
                    {entry === 'login' ? 'Sign In' : 'Sign Up'}
                  </button>
                );
              })}
            </div>

            <AnimatePresence mode="wait" initial={false}>
              <motion.div
                key={mode}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.2, ease: 'easeOut' }}
              >
                <h2 className="font-headline text-2xl font-extrabold tracking-tight md:text-3xl">{labels.title}</h2>
                <p className="mt-2 text-sm text-on-surface-variant">{labels.subtitle}</p>
              </motion.div>
            </AnimatePresence>

            <form onSubmit={handleSubmit} className="mt-8 space-y-5">
              <label className="block">
                <span className="mb-2 block font-label text-[11px] uppercase tracking-[0.18em] text-outline">Email</span>
                <input
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  type="email"
                  autoComplete="email"
                  placeholder="you@example.com"
                  className="w-full rounded-xl border border-outline-variant/30 bg-surface-container-low px-4 py-3 text-sm text-on-surface outline-none transition-colors placeholder:text-on-surface-variant/45 focus:border-primary/60"
                  disabled={isSubmitting}
                />
              </label>

              {error && (
                <div className="rounded-lg border border-error/30 bg-error/10 px-3 py-2 text-sm text-error">
                  {error}
                </div>
              )}

              <button
                type="submit"
                disabled={isSubmitting}
                className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-primary px-4 py-3 text-sm font-bold text-on-primary transition-all hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <span>{isSubmitting ? 'Working...' : labels.cta}</span>
                <ArrowRight className="h-4 w-4" />
              </button>
            </form>
          </div>
        </motion.section>
      </div>
    </main>
  );
}
