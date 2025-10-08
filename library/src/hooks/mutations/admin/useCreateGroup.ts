"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { createGroupAction } from "@/groups/actions/createGroupAction";

interface CreateGroupInput {
  name: string;
  description?: string;
  isPrivate: boolean;
}

export function useCreateGroup() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (input: CreateGroupInput) => {
      const formData = new FormData();
      formData.append("name", input.name);
      if (input.description) {
        formData.append("description", input.description);
      }
      formData.append("isPrivate", input.isPrivate.toString());

      return await createGroupAction(formData);
    },
    onSuccess: () => {
      // Invalidate groups list
      queryClient.invalidateQueries({ queryKey: ["groups"] });
    },
  });
}
