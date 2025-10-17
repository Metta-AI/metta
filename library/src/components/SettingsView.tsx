"use client";

import { useState, useEffect } from "react";
import { User } from "next-auth";
import { toast } from "sonner";
import { useDiscordStatus } from "@/hooks/queries/useSettings";
import { useBotStatus } from "@/hooks/queries/useSettings";
import { useNotificationPreferences } from "@/hooks/queries/useSettings";
import { useUnlinkDiscord } from "@/hooks/mutations/useUnlinkDiscord";
import { useUpdateNotificationPreferences } from "@/hooks/mutations/useUpdateNotificationPreferences";
import { useTestDiscordNotification } from "@/hooks/mutations/useTestDiscordNotification";

interface SettingsViewProps {
  user: User;
}

export function SettingsView({ user }: SettingsViewProps) {
  // TanStack Query hooks
  const {
    data: discordStatusData,
    isLoading: isLoadingDiscord,
    refetch: refetchDiscordStatus,
  } = useDiscordStatus();
  const { data: botStatusData, isLoading: isLoadingBot } = useBotStatus();
  const {
    data: preferencesData,
    isLoading: isLoadingPreferences,
    refetch: refetchPreferences,
  } = useNotificationPreferences();

  // Mutations
  const unlinkDiscordMutation = useUnlinkDiscord();
  const updatePreferencesMutation = useUpdateNotificationPreferences();
  const testDiscordMutation = useTestDiscordNotification();

  // UI-only state
  const [testMessage, setTestMessage] = useState("");
  const [urlMessage, setUrlMessage] = useState<{
    type: "success" | "error";
    message: string;
  } | null>(null);

  // Extract data from queries with defaults
  const discordStatus = discordStatusData || { isLinked: false };
  const botStatus = botStatusData?.configuration ?? null;
  const preferences = preferencesData?.preferences || {};
  const loading = isLoadingDiscord || isLoadingBot || isLoadingPreferences;

  const handleDiscordOAuth = () => {
    const clientId = process.env.NEXT_PUBLIC_DISCORD_CLIENT_ID;
    const redirectUri = encodeURIComponent(
      `${window.location.origin}/api/discord/auth`
    );
    const scope = encodeURIComponent("identify");

    const authUrl = `https://discord.com/oauth2/authorize?client_id=${clientId}&redirect_uri=${redirectUri}&response_type=code&scope=${scope}`;
    window.location.href = authUrl;
  };

  const handleDiscordUnlink = async () => {
    unlinkDiscordMutation.mutate(undefined, {
      onSuccess: () => {
        // Refetch preferences to update Discord settings
        refetchPreferences();
        toast.success("Discord account unlinked successfully");
      },
      onError: (error) => {
        console.error("Failed to unlink Discord:", error);
        toast.error("Failed to unlink Discord account");
      },
    });
  };

  const handlePreferenceChange = async (
    notificationType: string,
    channel: "emailEnabled" | "discordEnabled",
    enabled: boolean
  ) => {
    const newPreferences = {
      ...preferences,
      [notificationType]: {
        ...preferences[notificationType],
        [channel]: enabled,
      },
    };

    updatePreferencesMutation.mutate(newPreferences, {
      onError: (error) => {
        console.error("Failed to update preferences:", error);
        toast.error("Failed to update preferences");
      },
    });
  };

  const handleTestDiscord = async () => {
    if (!discordStatus.isLinked) {
      toast.error("Please link your Discord account first");
      return;
    }

    testDiscordMutation.mutate(undefined, {
      onSuccess: (data) => {
        if (data.success) {
          toast.success("Test Discord DM sent successfully");
          setTestMessage("");
        } else {
          toast.error(`Failed to send test Discord DM: ${data.message}`);
        }
      },
      onError: (error) => {
        console.error("Failed to test Discord:", error);
        toast.error("Failed to send test Discord DM");
      },
    });
  };

  const notificationTypes = [
    {
      key: "MENTION",
      label: "Mentions",
      description: "When someone mentions you",
    },
    {
      key: "COMMENT",
      label: "Comments",
      description: "Comments on your posts",
    },
    { key: "REPLY", label: "Replies", description: "Replies to your comments" },
    { key: "SYSTEM", label: "System", description: "System notifications" },
  ];

  // Check URL parameters for success/error messages
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const success = urlParams.get("success");
    const error = urlParams.get("error");
    const message = urlParams.get("message");

    if (success && message) {
      toast.success(message);
      // Clean up URL
      window.history.replaceState({}, "", "/settings");
      // Reload data if Discord was linked
      if (success === "discord_linked") {
        refetchDiscordStatus();
        refetchPreferences();
      }
    } else if (error && message) {
      toast.error(message);
      // Clean up URL
      window.history.replaceState({}, "", "/settings");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (loading) {
    return (
      <div className="flex h-full w-full flex-col">
        {/* Header Section - matches NewPostForm styling */}
        <div className="border-b border-gray-200 bg-white p-4 md:p-6">
          <div className="flex items-center justify-between">
            <h1 className="text-xl font-semibold text-gray-900">Settings</h1>
          </div>
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-4xl p-6">
            <div className="animate-pulse">
              <div className="mb-6 h-8 w-1/4 rounded bg-gray-200"></div>
              <div className="space-y-4">
                <div className="h-32 rounded bg-gray-200"></div>
                <div className="h-48 rounded bg-gray-200"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full w-full flex-col">
      {/* Header Section - matches NewPostForm styling */}
      <div className="border-b border-gray-200 bg-white p-4 md:p-6">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-semibold text-gray-900">Settings</h1>
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-4xl p-6">
          {/* Discord Integration Section */}
          <div className="mb-8">
            <div className="rounded-lg border border-gray-200 bg-white p-6">
              <h2 className="mb-4 text-2xl font-semibold text-gray-900">
                Discord Integration
              </h2>

              {/* Success/Error Message */}
              {urlMessage && (
                <div
                  className={`mt-4 rounded-lg border px-4 py-3 text-sm ${
                    urlMessage.type === "success"
                      ? "border-green-200 bg-green-50 text-green-700"
                      : "border-red-200 bg-red-50 text-red-700"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    {urlMessage.type === "success" ? (
                      <span className="text-lg">✅</span>
                    ) : (
                      <span className="text-lg">⚠️</span>
                    )}
                    <span>{urlMessage.message}</span>
                  </div>
                </div>
              )}

              {/* Bot Status */}
              <div className="mb-6 rounded-lg bg-gray-50 p-4">
                <h3 className="mb-2 font-medium text-gray-900">Bot Status</h3>
                {botStatus ? (
                  <div className="text-sm">
                    <div className="mb-1 flex items-center gap-2">
                      <div
                        className={`h-2 w-2 rounded-full ${
                          botStatus.ready ? "bg-green-500" : "bg-red-500"
                        }`}
                      ></div>
                      <span
                        className={
                          botStatus.ready ? "text-green-700" : "text-red-700"
                        }
                      >
                        {botStatus.ready ? "Ready" : "Not Ready"}
                      </span>
                    </div>
                    {botStatus.botUser && (
                      <p className="text-gray-600">Bot: {botStatus.botUser}</p>
                    )}
                    {botStatus.error && (
                      <p className="text-red-600">Error: {botStatus.error}</p>
                    )}
                  </div>
                ) : (
                  <p className="text-gray-500">Loading bot status...</p>
                )}
              </div>

              {/* Discord Account Linking */}
              <div className="mb-6">
                <h3 className="mb-3 font-medium text-gray-900">
                  Your Discord Account
                </h3>

                {discordStatus.isLinked ? (
                  <div className="flex items-center justify-between rounded-lg bg-green-50 p-4">
                    <div>
                      <div className="mb-1 flex items-center gap-2">
                        <div className="h-2 w-2 rounded-full bg-green-500"></div>
                        <span className="font-medium text-green-900">
                          Connected
                        </span>
                      </div>
                      <p className="text-green-700">
                        Linked to: {discordStatus.discordUsername}
                      </p>
                      <p className="text-sm text-green-600">
                        You'll receive Discord DMs for enabled notification
                        types.
                      </p>
                    </div>
                    <button
                      onClick={handleDiscordUnlink}
                      className="rounded-lg bg-red-600 px-4 py-2 text-white transition-colors hover:bg-red-700"
                    >
                      Unlink
                    </button>
                  </div>
                ) : (
                  <div className="flex items-center justify-between rounded-lg bg-yellow-50 p-4">
                    <div>
                      <div className="mb-1 flex items-center gap-2">
                        <div className="h-2 w-2 rounded-full bg-yellow-500"></div>
                        <span className="font-medium text-yellow-900">
                          Not Connected
                        </span>
                      </div>
                      <p className="mb-1 text-yellow-700">
                        Link your Discord account to receive DM notifications
                      </p>
                      <p className="text-sm text-yellow-600">
                        You'll be redirected to Discord to authorize the
                        connection.
                      </p>
                    </div>
                    <button
                      onClick={handleDiscordOAuth}
                      className="rounded-lg bg-blue-600 px-4 py-2 text-white transition-colors hover:bg-blue-700"
                    >
                      Link Discord
                    </button>
                  </div>
                )}
              </div>

              {/* Test Discord */}
              {discordStatus.isLinked && botStatus?.ready && (
                <div className="rounded-lg bg-blue-50 p-4">
                  <h3 className="mb-3 font-medium text-blue-900">
                    Test Discord DM
                  </h3>
                  <div className="space-y-3">
                    <textarea
                      value={testMessage}
                      onChange={(e) => setTestMessage(e.target.value)}
                      placeholder="Enter a custom test message (optional)"
                      className="w-full resize-none rounded-lg border border-gray-300 p-3"
                      rows={2}
                    />
                    <button
                      onClick={handleTestDiscord}
                      disabled={testDiscordMutation.isPending}
                      className="rounded-lg bg-blue-600 px-4 py-2 text-white transition-colors hover:bg-blue-700 disabled:bg-blue-400"
                    >
                      {testDiscordMutation.isPending
                        ? "Sending..."
                        : "Send Test DM"}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Notification Preferences Section */}
          <div>
            <div className="rounded-lg border border-gray-200 bg-white p-6">
              <h2 className="mb-4 text-2xl font-semibold text-gray-900">
                Notification Preferences
              </h2>
              <p className="mb-6 text-gray-600">
                Choose how you want to receive different types of notifications.
              </p>

              <div className="space-y-4">
                {notificationTypes.map((type) => {
                  const pref = preferences[type.key] || {
                    emailEnabled: true,
                    discordEnabled: false,
                  };

                  return (
                    <div
                      key={type.key}
                      className="flex items-center justify-between rounded-lg border border-gray-200 p-4"
                    >
                      <div>
                        <h3 className="font-medium text-gray-900">
                          {type.label}
                        </h3>
                        <p className="text-sm text-gray-600">
                          {type.description}
                        </p>
                      </div>

                      <div className="flex gap-4">
                        {/* Email Toggle */}
                        <label className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={pref.emailEnabled}
                            onChange={(e) =>
                              handlePreferenceChange(
                                type.key,
                                "emailEnabled",
                                e.target.checked
                              )
                            }
                            className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                          />
                          <span className="text-sm font-medium text-gray-700">
                            Email
                          </span>
                        </label>

                        {/* Discord Toggle */}
                        <label className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={
                              pref.discordEnabled && discordStatus.isLinked
                            }
                            onChange={(e) =>
                              handlePreferenceChange(
                                type.key,
                                "discordEnabled",
                                e.target.checked
                              )
                            }
                            disabled={!discordStatus.isLinked}
                            className="rounded border-gray-300 text-blue-600 focus:ring-blue-500 disabled:opacity-50"
                          />
                          <span
                            className={`text-sm font-medium ${
                              discordStatus.isLinked
                                ? "text-gray-700"
                                : "text-gray-400"
                            }`}
                          >
                            Discord
                          </span>
                        </label>
                      </div>
                    </div>
                  );
                })}
              </div>

              {!discordStatus.isLinked && (
                <div className="mt-4 rounded-lg bg-yellow-50 p-3">
                  <p className="text-sm text-yellow-700">
                    💡 Link your Discord account above to enable Discord
                    notifications.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
