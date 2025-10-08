"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toggleStarAction } from "@/posts/actions/toggleStarAction";
import { queryKeys } from "@/lib/hooks/useServerMutation";

interface StarData {
  totalStars: number;
  isStarredByCurrentUser: boolean;
}

interface ToggleStarVariables {
  paperId: string;
}

/**
 * Hook for toggling paper stars with optimistic updates
 *
 * This provides optimistic UI updates for better UX, with automatic
 * rollback on error and cache synchronization on success.
 */
export function useStarMutation() {
  const queryClient = useQueryClient();

  return useMutation<
    unknown,
    Error,
    ToggleStarVariables,
    { previousStars?: StarData }
  >({
    mutationFn: async ({ paperId }: ToggleStarVariables) => {
      const formData = new FormData();
      formData.append("paperId", paperId);
      const result = await toggleStarAction(formData);

      if (result.serverError) {
        throw new Error(result.serverError);
      }

      return result.data;
    },

    onMutate: async ({ paperId }) => {
      // Cancel any outgoing refetches for this paper
      const starQueryKey = queryKeys.papers.stars(paperId);
      await queryClient.cancelQueries({ queryKey: starQueryKey });

      // Snapshot the previous value
      const previousStars = queryClient.getQueryData<StarData>(starQueryKey);

      // Optimistically update the star state
      if (previousStars) {
        const newData: StarData = {
          totalStars: previousStars.isStarredByCurrentUser
            ? Math.max(previousStars.totalStars - 1, 0)
            : previousStars.totalStars + 1,
          isStarredByCurrentUser: !previousStars.isStarredByCurrentUser,
        };

        queryClient.setQueryData<StarData>(starQueryKey, newData);
      }

      // Return context for rollback
      return { previousStars };
    },

    onError: (_err, { paperId }, context) => {
      // Revert optimistic update on error
      if (context?.previousStars) {
        queryClient.setQueryData(
          queryKeys.papers.stars(paperId),
          context.previousStars
        );
      }
    },

    onSettled: (_data, _error, { paperId }) => {
      // Always refetch to ensure we have the latest data
      queryClient.invalidateQueries({
        queryKey: queryKeys.papers.stars(paperId),
      });
      // Also invalidate any related queries
      queryClient.invalidateQueries({ queryKey: queryKeys.papers.all });
      queryClient.invalidateQueries({ queryKey: queryKeys.feed.all });
    },
  });
}
