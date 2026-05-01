import { useState } from 'react';
import clsx from 'clsx';

type BrandMarkSize = 'nav' | 'auth';

interface BrandMarkProps {
  className?: string;
  logoSrc?: string;
  showWordmark?: boolean;
  size?: BrandMarkSize;
}

const SIZE_CLASSES: Record<BrandMarkSize, { icon: string; text: string; gap: string }> = {
  nav: {
    icon: 'h-6 w-6 md:h-8 md:w-8',
    text: 'text-xl md:text-2xl',
    gap: 'gap-2 md:gap-3',
  },
  auth: {
    icon: 'h-8 w-8 md:h-10 md:w-10',
    text: 'text-2xl md:text-3xl',
    gap: 'gap-3',
  },
};

export default function BrandMark({
  className,
  logoSrc = '/branding/alphamesh-logo.svg',
  showWordmark = true,
  size = 'nav',
}: BrandMarkProps) {
  const [imageFailed, setImageFailed] = useState(false);
  const styles = SIZE_CLASSES[size];

  return (
    <div
      className={clsx(
        'flex items-center font-headline font-extrabold tracking-tighter text-on-surface antialiased',
        styles.gap,
        className
      )}
    >
      <div className={clsx('relative flex items-center justify-center text-primary', styles.icon)}>
        {!imageFailed ? (
          <img
            src={logoSrc}
            alt="AlphaMesh logo"
            className="h-full w-full object-contain"
            onError={() => setImageFailed(true)}
          />
        ) : (
          <>
            <div className="absolute h-full w-full rotate-45 border-2 border-current opacity-30" />
            <div
              className="absolute h-[60%] w-[60%] rotate-45 bg-current"
              style={{ clipPath: 'polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)' }}
            />
          </>
        )}
      </div>
      {showWordmark && <span className={styles.text}>AlphaMesh</span>}
    </div>
  );
}
