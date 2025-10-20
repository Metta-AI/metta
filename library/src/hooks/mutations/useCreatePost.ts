"use client";

import { useRouter } from "next/navigation";
import { createServerMutation, queryKeys } from "@/lib/hooks/useServerMutation";
import { createPostAction } from "@/posts/actions/createPostAction";

interface CreatePostInput {
  title: string;
  content: string;
  images?: string[];
  mentions?: string[];
  quotedPostIds?: string[];
}

/**
 * Hook for creating a new post
 *
 * Creates a post and invalidates feed/posts queries to update the UI.
 */
export function useCreatePost() {
  const router = useRouter();

  const mutation = createServerMutation<unknown, CreatePostInput>({
    mutationFn: createPostAction,
    toFormData: ({ title, content, images, mentions, quotedPostIds }) => {
      const formData = new FormData();
      formData.append("title", title);
      formData.append("content", content);
      if (images && images.length > 0) {
        formData.append("images", JSON.stringify(images));
      }
      if (mentions && mentions.length > 0) {
        formData.append("mentions", JSON.stringify(mentions));
      }
      if (quotedPostIds && quotedPostIds.length > 0) {
        formData.append("quotedPostIds", JSON.stringify(quotedPostIds));
      }
      return formData;
    },
    invalidateQueries: [queryKeys.feed.all, queryKeys.posts.all],
  })({
    onSuccess: () => {
      // Refresh to ensure server components update with the new post
      router.refresh();
    },
  });

  return mutation;
}
