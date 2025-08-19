"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toggleStarAction } from "@/posts/actions/toggleStarAction";

interface StarData {
  totalStars: number;
  isStarredByCurrentUser: boolean;
}

export function useStarMutation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (paperId: string) => {
      const formData = new FormData();
      formData.append("paperId", paperId);
      return await toggleStarAction(formData);
    },
    onMutate: async (paperId: string) => {
      // Cancel any outgoing refetches for this paper
      await queryClient.cancelQueries({ queryKey: ["paper-stars", paperId] });

      // Snapshot the previous value
      const previousStars = queryClient.getQueryData<StarData>([
        "paper-stars",
        paperId,
      ]);

      // Optimistically update the star state
      if (previousStars) {
        const newData: StarData = {
          totalStars: previousStars.isStarredByCurrentUser
            ? Math.max(previousStars.totalStars - 1, 0)
            : previousStars.totalStars + 1,
          isStarredByCurrentUser: !previousStars.isStarredByCurrentUser,
        };

        queryClient.setQueryData<StarData>(["paper-stars", paperId], newData);
        console.log(`Updated star data for ${paperId}:`, newData);
      } else {
        console.warn(`No previous star data found for paper ${paperId}`);
      }

      // Return context for rollback
      return { previousStars };
    },
    onError: (err, paperId, context) => {
      // Revert optimistic update on error
      if (context?.previousStars) {
        queryClient.setQueryData(
          ["paper-stars", paperId],
          context.previousStars
        );
      }
    },
    onSettled: (data, error, paperId) => {
      // Always refetch to ensure we have the latest data
      queryClient.invalidateQueries({ queryKey: ["paper-stars", paperId] });
      // Also invalidate any related queries
      queryClient.invalidateQueries({ queryKey: ["papers"] });
      queryClient.invalidateQueries({ queryKey: ["feed"] });
    },
  });
}
