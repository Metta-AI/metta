"use client";

import { useState, useEffect } from "react";
import { User } from "next-auth";

interface DiscordLinkStatus {
  isLinked: boolean;
  discordUsername?: string;
  discordUserId?: string;
}

interface DiscordBotStatus {
  configured: boolean;
  botUser: string | null;
  ready: boolean;
  error?: string;
}

interface NotificationPreferences {
  [key: string]: {
    emailEnabled: boolean;
    discordEnabled: boolean;
  };
}

interface SettingsViewProps {
  user: User;
}

export function SettingsView({ user }: SettingsViewProps) {
  const [discordStatus, setDiscordStatus] = useState<DiscordLinkStatus>({
    isLinked: false,
  });
  const [botStatus, setBotStatus] = useState<DiscordBotStatus | null>(null);
  const [preferences, setPreferences] = useState<NotificationPreferences>({});
  const [loading, setLoading] = useState(true);
  const [testingDiscord, setTestingDiscord] = useState(false);
  const [testMessage, setTestMessage] = useState("");
  const [urlMessage, setUrlMessage] = useState<{
    type: "success" | "error";
    message: string;
  } | null>(null);

  const loadDiscordStatus = async () => {
    try {
      const response = await fetch("/api/discord/link");
      if (response.ok) {
        const data = await response.json();
        setDiscordStatus(data);
      }
    } catch (error) {
      console.error("Failed to load Discord status:", error);
    }
  };

  const loadBotStatus = async () => {
    try {
      const response = await fetch("/api/discord/test");
      if (response.ok) {
        const data = await response.json();
        setBotStatus(data.configuration);
      }
    } catch (error) {
      console.error("Failed to load bot status:", error);
    }
  };

  const loadNotificationPreferences = async () => {
    try {
      const response = await fetch("/api/notification-preferences");
      if (response.ok) {
        const data = await response.json();
        setPreferences(data.preferences || {});
      }
    } catch (error) {
      console.error("Failed to load notification preferences:", error);
    }
  };

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
    try {
      const response = await fetch("/api/discord/link", {
        method: "DELETE",
      });

      if (response.ok) {
        setDiscordStatus({ isLinked: false });
        // Reload preferences to update Discord settings
        await loadNotificationPreferences();
        alert("Discord account unlinked successfully!");
      } else {
        alert("Failed to unlink Discord account");
      }
    } catch (error) {
      console.error("Failed to unlink Discord:", error);
      alert("Failed to unlink Discord account");
    }
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

    setPreferences(newPreferences);

    try {
      const response = await fetch("/api/notification-preferences", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          preferences: newPreferences,
        }),
      });

      if (!response.ok) {
        // Revert on failure
        await loadNotificationPreferences();
        alert("Failed to update preferences");
      }
    } catch (error) {
      console.error("Failed to update preferences:", error);
      // Revert on failure
      await loadNotificationPreferences();
      alert("Failed to update preferences");
    }
  };

  const handleTestDiscord = async () => {
    if (!discordStatus.isLinked) {
      alert("Please link your Discord account first");
      return;
    }

    setTestingDiscord(true);
    try {
      const response = await fetch("/api/discord/test", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          action: "send",
          testMessage: testMessage || undefined,
        }),
      });

      const data = await response.json();
      if (response.ok && data.success) {
        alert(`✅ Test Discord DM sent successfully to ${data.discordUser}!`);
        setTestMessage("");
      } else {
        alert(
          `❌ Failed to send test Discord DM: ${data.message || data.error}`
        );
      }
    } catch (error) {
      console.error("Failed to test Discord:", error);
      alert("❌ Failed to send test Discord DM");
    } finally {
      setTestingDiscord(false);
    }
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
      setUrlMessage({ type: "success", message });
      // Clean up URL
      window.history.replaceState({}, "", "/settings");
      // Reload data if Discord was linked
      if (success === "discord_linked") {
        loadDiscordStatus();
        loadNotificationPreferences();
      }
    } else if (error && message) {
      setUrlMessage({ type: "error", message });
      // Clean up URL
      window.history.replaceState({}, "", "/settings");
    }
  }, []);

  // Load initial data
  useEffect(() => {
    Promise.all([
      loadDiscordStatus(),
      loadBotStatus(),
      loadNotificationPreferences(),
    ]).finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="mx-auto max-w-4xl p-6">
        <div className="animate-pulse">
          <div className="mb-6 h-8 w-1/4 rounded bg-gray-200"></div>
          <div className="space-y-4">
            <div className="h-32 rounded bg-gray-200"></div>
            <div className="h-48 rounded bg-gray-200"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-4xl p-6">
      <h1 className="mb-8 text-3xl font-bold text-gray-900">Settings</h1>

      {/* Discord Integration Section */}
      <div className="mb-8">
        <div className="rounded-lg border border-gray-200 bg-white p-6">
          <h2 className="mb-4 text-2xl font-semibold text-gray-900">
            Discord Integration
          </h2>

          {/* Success/Error Message */}
          {urlMessage && (
            <div
              className={`mb-4 rounded-lg p-4 ${
                urlMessage.type === "success"
                  ? "border border-green-200 bg-green-50"
                  : "border border-red-200 bg-red-50"
              }`}
            >
              <div className="flex">
                <div className="flex-shrink-0">
                  {urlMessage.type === "success" ? (
                    <svg
                      className="h-5 w-5 text-green-400"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                    >
                      <path
                        fillRule="evenodd"
                        d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                        clipRule="evenodd"
                      />
                    </svg>
                  ) : (
                    <svg
                      className="h-5 w-5 text-red-400"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                    >
                      <path
                        fillRule="evenodd"
                        d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                        clipRule="evenodd"
                      />
                    </svg>
                  )}
                </div>
                <div className="ml-3">
                  <p
                    className={`text-sm ${
                      urlMessage.type === "success"
                        ? "text-green-800"
                        : "text-red-800"
                    }`}
                  >
                    {urlMessage.message}
                  </p>
                </div>
                <div className="ml-auto pl-3">
                  <button
                    onClick={() => setUrlMessage(null)}
                    className={`inline-flex rounded-md p-1.5 ${
                      urlMessage.type === "success"
                        ? "bg-green-50 text-green-500 hover:bg-green-100"
                        : "bg-red-50 text-red-500 hover:bg-red-100"
                    } focus:ring-2 focus:ring-offset-2 focus:outline-none ${
                      urlMessage.type === "success"
                        ? "focus:ring-green-600"
                        : "focus:ring-red-600"
                    }`}
                  >
                    <svg
                      className="h-4 w-4"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                    >
                      <path
                        fillRule="evenodd"
                        d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                        clipRule="evenodd"
                      />
                    </svg>
                  </button>
                </div>
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
                    You'll receive Discord DMs for enabled notification types.
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
                    You'll be redirected to Discord to authorize the connection.
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
                  disabled={testingDiscord}
                  className="rounded-lg bg-blue-600 px-4 py-2 text-white transition-colors hover:bg-blue-700 disabled:bg-blue-400"
                >
                  {testingDiscord ? "Sending..." : "Send Test DM"}
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
                    <h3 className="font-medium text-gray-900">{type.label}</h3>
                    <p className="text-sm text-gray-600">{type.description}</p>
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
                        checked={pref.discordEnabled && discordStatus.isLinked}
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
  );
}
