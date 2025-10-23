"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { banUserAction } from "@/users/actions/banUserAction";

interface BanUserInput {
  userId: string;
  reason: string;
}

export function useBanUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (input: BanUserInput) => {
      const formData = new FormData();
      formData.append("userId", input.userId);
      formData.append("reason", input.reason);

      return await banUserAction(formData);
    },
    onSuccess: () => {
      // Invalidate users queries
      queryClient.invalidateQueries({ queryKey: ["users"] });
      queryClient.invalidateQueries({ queryKey: ["admin-users"] });
    },
  });
}
