"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { deletePostAction } from "@/posts/actions/deletePostAction";

export function useDeletePost() {
  const queryClient = useQueryClient();
  const router = useRouter();

  return useMutation({
    mutationFn: async (postId: string) => {
      const formData = new FormData();
      formData.append("postId", postId);
      return await deletePostAction(formData);
    },
    onSuccess: () => {
      // Invalidate feed to remove the deleted post
      queryClient.invalidateQueries({ queryKey: ["feed"] });
      queryClient.invalidateQueries({ queryKey: ["posts"] });
      // Optionally refresh the page
      router.refresh();
    },
  });
}
