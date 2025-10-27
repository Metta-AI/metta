"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { joinInstitutionAction } from "@/institutions/actions/joinInstitutionAction";
import { queryKeys } from "@/lib/hooks/useServerMutation";

interface JoinInstitutionInput {
  institutionId: string;
}

/**
 * Hook for joining an institution
 * Invalidates institutions queries to refresh membership data
 */
export function useJoinInstitution() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (input: JoinInstitutionInput) => {
      const formData = new FormData();
      formData.append("institutionId", input.institutionId);

      const result = await joinInstitutionAction(formData);

      // Handle next-safe-action error structure
      if (result.serverError) {
        throw new Error(result.serverError);
      }

      return result.data;
    },
    onSuccess: () => {
      // Invalidate institutions queries to refresh the directory and memberships
      queryClient.invalidateQueries({ queryKey: queryKeys.institutions.all });
    },
  });
}
