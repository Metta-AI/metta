"use server";

import { zfd } from "zod-form-data";
import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { updateNotificationPreferences } from "@/lib/notification-preferences";

const inputSchema = zfd.formData({
  preferences: zfd.text(),
});

export const updateNotificationPreferencesAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    const session = await getSessionOrRedirect();
    const userId = session.user.id;

    const preferences = JSON.parse(input.preferences);

    await updateNotificationPreferences(userId, preferences);

    return { success: true, preferences };
  });
