"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";

import { deleteGroupAction } from "@/groups/actions/deleteGroupAction";
import { createServerMutation } from "@/lib/hooks/useServerMutation";

interface DeleteGroupVariables {
  groupId: string;
}

/**
 * Hook for deleting a group
 *
 * Removes a group and invalidates related queries to update the UI.
 */
export function useDeleteGroup() {
  const router = useRouter();
  const queryClient = useQueryClient();

  const mutation = createServerMutation<unknown, DeleteGroupVariables>({
    mutationFn: deleteGroupAction,
    invalidateQueries: [["groups"]],
  })({
    onSuccess: () => {
      // Invalidate groups queries
      queryClient.invalidateQueries({ queryKey: ["groups"] });
      // Refresh the page to ensure server components update
      router.refresh();
    },
  });

  return mutation;
}
