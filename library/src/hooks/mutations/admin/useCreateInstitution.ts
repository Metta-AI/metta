"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { createInstitutionAction } from "@/institutions/actions/createInstitutionAction";

interface CreateInstitutionInput {
  name: string;
  domain?: string;
  description?: string;
  website?: string;
  location?: string;
  type:
    | "UNIVERSITY"
    | "COMPANY"
    | "RESEARCH_LAB"
    | "NONPROFIT"
    | "GOVERNMENT"
    | "OTHER";
}

export function useCreateInstitution() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (input: CreateInstitutionInput) => {
      const formData = new FormData();
      formData.append("name", input.name);
      if (input.domain) formData.append("domain", input.domain);
      if (input.description) formData.append("description", input.description);
      if (input.website) formData.append("website", input.website);
      if (input.location) formData.append("location", input.location);
      formData.append("type", input.type);

      return await createInstitutionAction(formData);
    },
    onSuccess: () => {
      // Invalidate institutions list
      queryClient.invalidateQueries({ queryKey: ["institutions"] });
    },
  });
}
