"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { manageGroupMembershipAction } from "@/groups/actions/manageGroupMembershipAction";

interface ManageGroupMemberInput {
  groupId: string;
  userEmail: string;
  action: "add" | "remove";
  role?: string;
}

export function useManageGroupMember() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (input: ManageGroupMemberInput) => {
      const formData = new FormData();
      formData.append("groupId", input.groupId);
      formData.append("userEmail", input.userEmail);
      formData.append("action", input.action);
      if (input.role) formData.append("role", input.role);

      return await manageGroupMembershipAction(formData);
    },
    onSuccess: () => {
      // Invalidate groups queries
      queryClient.invalidateQueries({ queryKey: ["groups"] });
      queryClient.invalidateQueries({ queryKey: ["group-members"] });
    },
  });
}
