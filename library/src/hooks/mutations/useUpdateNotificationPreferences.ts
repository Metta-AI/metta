"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { updateNotificationPreferencesAction } from "@/settings/actions/updateNotificationPreferencesAction";

interface NotificationPreferenceSettings {
  emailEnabled?: boolean;
  discordEnabled?: boolean;
}

interface NotificationPreferences {
  [key: string]: NotificationPreferenceSettings;
}

export function useUpdateNotificationPreferences() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (preferences: NotificationPreferences) => {
      const formData = new FormData();
      formData.append("preferences", JSON.stringify(preferences));
      return await updateNotificationPreferencesAction(formData);
    },
    onSuccess: () => {
      // Invalidate preferences query
      queryClient.invalidateQueries({ queryKey: ["notification-preferences"] });
    },
  });
}
