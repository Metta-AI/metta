"use client";

import { createServerMutation, queryKeys } from "@/lib/hooks/useServerMutation";
import { toggleQueueAction } from "@/posts/actions/toggleQueueAction";

interface ToggleQueueVariables {
  paperId: string;
  postId?: string;
}

/**
 * Hook for toggling paper queue status
 *
 * Adds or removes a paper from the user's reading queue.
 */
export const useQueuePaper = createServerMutation<
  unknown,
  ToggleQueueVariables
>({
  mutationFn: toggleQueueAction,
  toFormData: ({ paperId, postId }) => {
    const formData = new FormData();
    formData.append("paperId", paperId);
    if (postId) {
      formData.append("postId", postId);
    }
    return formData;
  },
  invalidateQueries: [
    queryKeys.papers.all,
    queryKeys.feed.all,
    queryKeys.user.papers,
  ],
});
