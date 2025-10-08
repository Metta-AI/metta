"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { markNotificationsReadAction } from "@/notifications/actions/markNotificationsReadAction";

interface MarkNotificationsReadInput {
  notificationIds?: string[];
  markAllRead?: boolean;
}

export function useMarkNotificationsRead() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (input: MarkNotificationsReadInput) => {
      const formData = new FormData();
      if (input.notificationIds) {
        formData.append(
          "notificationIds",
          JSON.stringify(input.notificationIds)
        );
      }
      if (input.markAllRead !== undefined) {
        formData.append("markAllRead", input.markAllRead.toString());
      }
      return await markNotificationsReadAction(formData);
    },
    onSuccess: () => {
      // Invalidate notifications queries to refetch
      queryClient.invalidateQueries({ queryKey: ["notifications"] });
      queryClient.invalidateQueries({ queryKey: ["notification-count"] });
    },
  });
}
