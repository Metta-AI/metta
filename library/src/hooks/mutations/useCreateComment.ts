"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { createCommentAction } from "@/posts/actions/createCommentAction";

interface CreateCommentInput {
  postId: string;
  content: string;
  parentCommentId?: string;
  mentions?: string[];
}

export function useCreateComment() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      postId,
      content,
      parentCommentId,
      mentions = [],
    }: CreateCommentInput) => {
      const formData = new FormData();
      formData.append("postId", postId);
      formData.append("content", content);
      if (parentCommentId) {
        formData.append("parentCommentId", parentCommentId);
      }
      // Add mentions
      mentions.forEach((mention) => {
        formData.append("mentions", mention);
      });
      return await createCommentAction(formData);
    },
    onSuccess: (_data, variables) => {
      // Invalidate comments for this post
      queryClient.invalidateQueries({
        queryKey: ["comments", variables.postId],
      });
      // Invalidate feed to update reply counts
      queryClient.invalidateQueries({ queryKey: ["feed"] });
    },
  });
}
