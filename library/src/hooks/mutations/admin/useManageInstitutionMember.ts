"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { manageUserMembershipAction } from "@/institutions/actions/manageUserMembershipAction";

interface ManageInstitutionMemberInput {
  institutionId: string;
  userEmail: string;
  action: "add" | "remove" | "update";
  role?: string;
  department?: string;
  title?: string;
}

export function useManageInstitutionMember() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (input: ManageInstitutionMemberInput) => {
      const formData = new FormData();
      formData.append("institutionId", input.institutionId);
      formData.append("userEmail", input.userEmail);
      formData.append("action", input.action);
      if (input.role) formData.append("role", input.role);
      if (input.department) formData.append("department", input.department);
      if (input.title) formData.append("title", input.title);

      return await manageUserMembershipAction(formData);
    },
    onSuccess: () => {
      // Invalidate institutions queries
      queryClient.invalidateQueries({ queryKey: ["institutions"] });
      queryClient.invalidateQueries({ queryKey: ["institution-members"] });
    },
  });
}
