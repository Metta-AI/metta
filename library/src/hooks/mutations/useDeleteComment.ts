"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { deleteCommentAction } from "@/posts/actions/deleteCommentAction";

export function useDeleteComment() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (commentId: string) => {
      const formData = new FormData();
      formData.append("commentId", commentId);
      return await deleteCommentAction(formData);
    },
    onSuccess: (_data, _variables, context) => {
      // Invalidate comments queries to refetch
      queryClient.invalidateQueries({ queryKey: ["comments"] });
      // Also invalidate feed to update reply counts
      queryClient.invalidateQueries({ queryKey: ["feed"] });
    },
  });
}
