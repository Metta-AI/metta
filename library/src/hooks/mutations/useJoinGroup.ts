"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { joinGroupAction } from "@/groups/actions/joinGroupAction";
import { queryKeys } from "@/lib/hooks/useServerMutation";

interface JoinGroupInput {
  groupId: string;
}

/**
 * Hook for joining a group
 * Invalidates relevant queries to refresh group membership data
 */
export function useJoinGroup() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (input: JoinGroupInput) => {
      const formData = new FormData();
      formData.append("groupId", input.groupId);

      const result = await joinGroupAction(formData);

      // Handle next-safe-action error structure
      if (result.serverError) {
        throw new Error(result.serverError);
      }

      return result.data;
    },
    onSuccess: () => {
      // Invalidate groups queries to refresh the directory and memberships
      // Note: groups don't have a dedicated queryKeys entry yet, using generic approach
      queryClient.invalidateQueries({ queryKey: ["groups"] });
    },
  });
}
