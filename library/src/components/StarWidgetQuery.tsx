"use client";

import { FC, useEffect } from "react";
import { StarWidget } from "./StarWidget";
import { useStarMutation } from "@/hooks/useStarMutation";
import { useQueryClient } from "@tanstack/react-query";
import { queryKeys } from "@/lib/hooks/useServerMutation";

interface StarWidgetQueryProps {
  paperId: string;
  initialTotalStars: number;
  initialIsStarredByCurrentUser: boolean;
  size?: "sm" | "md" | "lg" | "xl";
  readonly?: boolean;
}

interface StarData {
  totalStars: number;
  isStarredByCurrentUser: boolean;
}

/**
 * StarWidgetQuery Component
 *
 * A wrapper around StarWidget that uses TanStack Query for state management.
 * Provides optimistic updates and automatic synchronization across components.
 */
export const StarWidgetQuery: FC<StarWidgetQueryProps> = ({
  paperId,
  initialTotalStars,
  initialIsStarredByCurrentUser,
  size = "md",
  readonly = false,
}) => {
  const queryClient = useQueryClient();

  const initialData: StarData = {
    totalStars: initialTotalStars,
    isStarredByCurrentUser: initialIsStarredByCurrentUser,
  };

  // Set initial data in cache if not already present
  useEffect(() => {
    const queryKey = queryKeys.papers.stars(paperId);
    const existingData = queryClient.getQueryData<StarData>(queryKey);
    if (!existingData) {
      queryClient.setQueryData<StarData>(queryKey, initialData);
    }
  }, [paperId, queryClient, initialData]);

  // Read current star data from cache
  const starData =
    queryClient.getQueryData<StarData>(queryKeys.papers.stars(paperId)) ??
    initialData;

  const starMutation = useStarMutation();

  const handleStarToggle = () => {
    if (readonly) return;
    starMutation.mutate({ paperId });
  };

  return (
    <StarWidget
      totalStars={starData.totalStars}
      isStarredByCurrentUser={starData.isStarredByCurrentUser}
      onClick={readonly ? undefined : handleStarToggle}
      size={size}
    />
  );
};
