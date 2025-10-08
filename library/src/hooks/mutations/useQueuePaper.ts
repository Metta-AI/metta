"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toggleQueueAction } from "@/posts/actions/toggleQueueAction";

export function useQueuePaper() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      paperId,
      postId,
    }: {
      paperId: string;
      postId?: string;
    }) => {
      const formData = new FormData();
      formData.append("paperId", paperId);
      if (postId) {
        formData.append("postId", postId);
      }
      return await toggleQueueAction(formData);
    },
    onSuccess: () => {
      // Invalidate relevant queries to refetch data
      queryClient.invalidateQueries({ queryKey: ["papers"] });
      queryClient.invalidateQueries({ queryKey: ["feed"] });
      queryClient.invalidateQueries({ queryKey: ["user-papers"] });
    },
  });
}
