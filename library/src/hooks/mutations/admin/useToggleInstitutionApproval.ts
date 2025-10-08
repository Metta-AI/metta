"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toggleApprovalRequirementAction } from "@/institutions/actions/toggleApprovalRequirementAction";

interface ToggleApprovalInput {
  institutionId: string;
  requiresApproval: boolean;
}

export function useToggleInstitutionApproval() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (input: ToggleApprovalInput) => {
      const formData = new FormData();
      formData.append("institutionId", input.institutionId);
      formData.append("requiresApproval", input.requiresApproval.toString());

      return await toggleApprovalRequirementAction(formData);
    },
    onSuccess: () => {
      // Invalidate institutions queries
      queryClient.invalidateQueries({ queryKey: ["institutions"] });
    },
  });
}
