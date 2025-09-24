"use client";

import { FC, useEffect, useRef, useCallback } from "react";

/**
 * InfiniteScroll Component
 * 
 * Automatically loads more content when the user scrolls near the bottom.
 * Uses Intersection Observer API for efficient scroll detection.
 */
export const InfiniteScroll: FC<{
  loadNext: (count: number) => void;
  hasMore: boolean;
  loading?: boolean;
  children: React.ReactNode;
}> = ({ loadNext, hasMore, loading = false, children }) => {
  const observerRef = useRef<HTMLDivElement>(null);
  const loadingRef = useRef<HTMLDivElement>(null);

  const handleIntersection = useCallback(
    (entries: IntersectionObserverEntry[]) => {
      const [entry] = entries;
      if (entry.isIntersecting && hasMore && !loading) {
        loadNext(10); // Load 10 more items
      }
    },
    [loadNext, hasMore, loading]
  );

  useEffect(() => {
    const observer = new IntersectionObserver(handleIntersection, {
      root: null,
      rootMargin: "100px", // Start loading when within 100px of the bottom
      threshold: 0.1,
    });

    if (loadingRef.current) {
      observer.observe(loadingRef.current);
    }

    return () => {
      if (loadingRef.current) {
        observer.unobserve(loadingRef.current);
      }
    };
  }, [handleIntersection]);

  return (
    <div ref={observerRef}>
      {children}
      
      {/* Loading indicator at the bottom */}
      {hasMore && (
        <div ref={loadingRef} className="p-4 text-center">
          {loading ? (
            <div className="flex items-center justify-center gap-2 text-gray-500">
              <div className="w-4 h-4 border-2 border-gray-300 border-t-blue-600 rounded-full animate-spin"></div>
              <span>Loading more posts...</span>
            </div>
          ) : (
            <div className="h-4" /> // Invisible element for intersection observer
          )}
        </div>
      )}
    </div>
  );
}; 