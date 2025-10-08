"use client";

import { useRouter } from "next/navigation";
import { createServerMutation, queryKeys } from "@/lib/hooks/useServerMutation";
import { deletePostAction } from "@/posts/actions/deletePostAction";

interface DeletePostVariables {
  postId: string;
}

/**
 * Hook for deleting a post
 *
 * Removes a post and invalidates related queries to update the UI.
 * Note: This hook also triggers a router refresh to ensure server components update.
 */
export function useDeletePost() {
  const router = useRouter();

  const mutation = createServerMutation<unknown, DeletePostVariables>({
    mutationFn: deletePostAction,
    invalidateQueries: [queryKeys.feed.all, queryKeys.posts.all],
  })({
    onSuccess: () => {
      // Refresh the page to ensure server components update
      router.refresh();
    },
  });

  return mutation;
}
