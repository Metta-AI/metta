"use client";

import { useMutation } from "@tanstack/react-query";
import { sendTestDiscordNotification } from "@/lib/api/resources/settings";

export function useTestDiscordNotification() {
  return useMutation({
    mutationFn: async () => {
      return await sendTestDiscordNotification();
    },
  });
}
