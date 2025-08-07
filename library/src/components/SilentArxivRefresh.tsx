"use client";

import { useEffect, useRef } from "react";
import { useRouter } from "next/navigation";

interface SilentArxivRefreshProps {
  postId: string;
  content: string;
  onInstitutionsAdded?: (institutions: string[]) => void;
}

export function SilentArxivRefresh({
  postId,
  content,
  onInstitutionsAdded,
}: SilentArxivRefreshProps) {
  const router = useRouter();
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    // Check if content has arXiv URL
    const hasArxivUrl = /https?:\/\/arxiv\.org\/abs\/\d+\.\d+/gi.test(content);

    if (!hasArxivUrl) {
      return;
    }

    // Check if we've already completed processing for this post
    const storageKey = `arxivCompleted:${postId}`;
    const alreadyCompleted =
      typeof window !== "undefined"
        ? window.localStorage.getItem(storageKey) === "true"
        : false;

    if (alreadyCompleted) {
      return;
    }

    // Poll silently for completion
    const checkStatus = async () => {
      try {
        const response = await fetch(`/api/papers/${postId}/institutions`);
        if (response.ok) {
          const { hasInstitutions } = await response.json();
          if (hasInstitutions) {
            // Mark as completed
            if (typeof window !== "undefined") {
              window.localStorage.setItem(storageKey, "true");
            }

            // Fetch the actual institution data and update local state
            if (onInstitutionsAdded) {
              try {
                const paperResponse = await fetch(`/api/papers/${postId}/data`);
                if (paperResponse.ok) {
                  const { institutions } = await paperResponse.json();
                  onInstitutionsAdded(institutions || []);
                }
              } catch (error) {
                console.error("Failed to fetch institution data:", error);
              }
            }

            // Also refresh as fallback
            router.refresh();

            if (intervalRef.current) {
              clearInterval(intervalRef.current);
              intervalRef.current = null;
            }
          }
        }
      } catch (error) {
        console.error("Failed to check institution status:", error);
      }
    };

    // Start polling every 3 seconds
    intervalRef.current = setInterval(checkStatus, 3000);

    // Cleanup after 2 minutes max (in case something goes wrong)
    const timeout = setTimeout(() => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }, 120000);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      clearTimeout(timeout);
    };
  }, [postId, content, router, onInstitutionsAdded]);

  // This component renders nothing
  return null;
}
