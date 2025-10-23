import { fetchJson } from "@/lib/api/client";

export interface NotificationDTO {
  id: string;
  type: string;
  title: string;
  message?: string | null;
  isRead: boolean;
  createdAt: string;
  actionUrl?: string | null;
  mentionText?: string | null;
  actor?: {
    id: string;
    name?: string | null;
    email?: string | null;
  } | null;
  post?: {
    id: string;
    title: string;
  } | null;
  comment?: {
    id: string;
    content: string;
    post: {
      id: string;
      title: string;
    };
  } | null;
}

export interface NotificationCounts {
  total: number;
  unread: number;
}

export interface NotificationsResponse {
  notifications: NotificationDTO[];
  counts: NotificationCounts;
}

export async function listNotifications(params?: {
  limit?: number;
  includeRead?: boolean;
}): Promise<NotificationsResponse> {
  const queryParams = new URLSearchParams();
  if (params?.limit) queryParams.set("limit", String(params.limit));
  if (params?.includeRead !== undefined) {
    queryParams.set("includeRead", String(params.includeRead));
  }

  const url =
    queryParams.size > 0
      ? `/api/notifications?${queryParams.toString()}`
      : "/api/notifications";
  return fetchJson<NotificationsResponse>(url);
}

export async function getNotificationCounts(): Promise<NotificationCounts> {
  return fetchJson<NotificationCounts>("/api/notifications/count");
}

export async function markNotificationRead(id: string): Promise<void> {
  await fetchJson(`/api/notifications/${id}/read`, {
    method: "POST",
    skipJsonParse: true,
  });
}
