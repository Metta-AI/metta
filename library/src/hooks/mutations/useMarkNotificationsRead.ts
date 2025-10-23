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
      return await markNotificationsReadAction(input);
    },
    onSuccess: () => {
      // Invalidate notifications queries to refetch
      queryClient.invalidateQueries({ queryKey: ["notifications"] });
      queryClient.invalidateQueries({ queryKey: ["notification-count"] });
    },
  });
}
