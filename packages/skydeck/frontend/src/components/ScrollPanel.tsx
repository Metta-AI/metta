import { useRef, useEffect, useLayoutEffect } from 'react';
import type { ReactNode } from 'react';

interface ScrollPanelProps {
  children: ReactNode;
  className?: string;
  /** Dependencies that trigger scroll restoration after they change */
  deps?: unknown[];
  /** Max height for the panel (enables vertical scrolling) */
  maxHeight?: string | number;
  /** Additional styles */
  style?: React.CSSProperties;
}

/**
 * A panel that preserves scroll position across re-renders.
 * Useful for tables/lists that refresh data in the background.
 */
export function ScrollPanel({
  children,
  className,
  deps = [],
  maxHeight,
  style,
}: ScrollPanelProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const scrollPosRef = useRef<number>(0);

  // Save scroll position before any render
  useLayoutEffect(() => {
    if (containerRef.current) {
      scrollPosRef.current = containerRef.current.scrollTop;
    }
  });

  // Restore scroll position after deps change
  useEffect(() => {
    if (containerRef.current && scrollPosRef.current > 0) {
      containerRef.current.scrollTop = scrollPosRef.current;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  const combinedStyle: React.CSSProperties = {
    overflowY: 'auto',
    overflowX: 'auto',
    ...(maxHeight && { maxHeight }),
    ...style,
  };

  return (
    <div ref={containerRef} className={className} style={combinedStyle}>
      {children}
    </div>
  );
}
