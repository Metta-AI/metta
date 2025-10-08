"use client";

import { FC, useEffect } from "react";
import { StarWidget } from "./StarWidget";
import { usePaperStars } from "@/hooks/usePaperStars";
import { useStarMutation } from "@/hooks/useStarMutation";
import { useQueryClient } from "@tanstack/react-query";

interface StarWidgetQueryProps {
  paperId: string;
  initialTotalStars: number;
  initialIsStarredByCurrentUser: boolean;
  size?: "sm" | "md" | "lg" | "xl";
  readonly?: boolean;
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

  const initialData = {
    totalStars: initialTotalStars,
    isStarredByCurrentUser: initialIsStarredByCurrentUser,
  };

  // Set initial data in cache if not already present
  useEffect(() => {
    const existingData = queryClient.getQueryData(["paper-stars", paperId]);
    if (!existingData) {
      queryClient.setQueryData(["paper-stars", paperId], initialData);
      console.log(`Set initial star data for ${paperId}:`, initialData);
    }
  }, [paperId, queryClient, initialTotalStars, initialIsStarredByCurrentUser]);

  const { data: starData } = usePaperStars({
    paperId,
    initialData,
  });

  const starMutation = useStarMutation();

  const handleStarToggle = () => {
    if (readonly) return;
    starMutation.mutate({ paperId });
  };

  return (
    <StarWidget
      totalStars={starData?.totalStars ?? initialTotalStars}
      isStarredByCurrentUser={
        starData?.isStarredByCurrentUser ?? initialIsStarredByCurrentUser
      }
      onClick={readonly ? undefined : handleStarToggle}
      size={size}
    />
  );
};
