"use client";

import { useQuery } from "@tanstack/react-query";

interface StarData {
  totalStars: number;
  isStarredByCurrentUser: boolean;
}

interface UsePaperStarsProps {
  paperId: string;
  initialData?: StarData;
}

// This hook manages star data for a paper using TanStack Query
export function usePaperStars({ paperId, initialData }: UsePaperStarsProps) {
  return useQuery({
    queryKey: ["paper-stars", paperId],
    queryFn: async (): Promise<StarData> => {
      // In a real implementation, this would fetch from /api/papers/${paperId}/stars
      // For now, we throw an error since we rely on mutations and placeholderData
      throw new Error(
        `No cached data for paper ${paperId}. This should be set by mutations.`
      );
    },
    staleTime: Infinity, // Never refetch automatically since we rely on mutations
    enabled: false, // Disable automatic fetching, we rely on mutations
    placeholderData: initialData, // Use placeholder data instead of initial data
    refetchOnMount: false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  });
}
