"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { unbanUserAction } from "@/users/actions/unbanUserAction";

export function useUnbanUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (userId: string) => {
      const formData = new FormData();
      formData.append("userId", userId);

      return await unbanUserAction(formData);
    },
    onSuccess: () => {
      // Invalidate users queries
      queryClient.invalidateQueries({ queryKey: ["users"] });
      queryClient.invalidateQueries({ queryKey: ["admin-users"] });
    },
  });
}
