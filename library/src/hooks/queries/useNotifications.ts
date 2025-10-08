"use client";

import { useQuery } from "@tanstack/react-query";
import * as notificationsApi from "@/lib/api/resources/notifications";

export function useNotifications(params?: {
  limit?: number;
  includeRead?: boolean;
}) {
  return useQuery({
    queryKey: ["notifications", params],
    queryFn: () => notificationsApi.listNotifications(params),
  });
}

export function useNotificationCounts() {
  return useQuery({
    queryKey: ["notification-count"],
    queryFn: () => notificationsApi.getNotificationCounts(),
    refetchInterval: 30000, // Refetch every 30 seconds
  });
}
