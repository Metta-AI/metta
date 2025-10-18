"use client";

import { createServerMutation, queryKeys } from "@/lib/hooks/useServerMutation";
import { createCommentAction } from "@/posts/actions/createCommentAction";

interface CreateCommentInput {
  postId: string;
  content: string;
  parentCommentId?: string;
  mentions?: string[];
}

/**
 * Hook for creating a comment on a post
 *
 * Adds a new comment and invalidates related queries to update the UI.
 */
export const useCreateComment = createServerMutation<
  unknown,
  CreateCommentInput
>({
  mutationFn: createCommentAction,
  toFormData: ({ postId, content, parentCommentId, mentions = [] }) => {
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
    return formData;
  },
  invalidateQueries: (_, variables) => [
    queryKeys.comments.byPost(variables.postId),
    queryKeys.feed.all,
  ],
});
