"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { createInstitutionAction } from "@/institutions/actions/createInstitutionAction";

interface CreateInstitutionInput {
  name: string;
  domain: string;
  description?: string;
}

export function useCreateInstitution() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (input: CreateInstitutionInput) => {
      const formData = new FormData();
      formData.append("name", input.name);
      formData.append("domain", input.domain);
      if (input.description) {
        formData.append("description", input.description);
      }

      return await createInstitutionAction(formData);
    },
    onSuccess: () => {
      // Invalidate institutions list
      queryClient.invalidateQueries({ queryKey: ["institutions"] });
    },
  });
}
