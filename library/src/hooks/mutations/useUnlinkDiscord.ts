"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { unlinkDiscordAction } from "@/settings/actions/unlinkDiscordAction";

export function useUnlinkDiscord() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async () => {
      return await unlinkDiscordAction();
    },
    onSuccess: () => {
      // Invalidate Discord status query
      queryClient.invalidateQueries({ queryKey: ["discord-status"] });
    },
  });
}
