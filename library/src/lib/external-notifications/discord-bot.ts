/**
 * Discord Bot Service (REST API)
 *
 * Handles Discord bot operations using REST API for DM sending,
 * user interaction, and account linking verification.
 * No native dependencies - just HTTP requests!
 */

import { prisma } from "@/lib/db/prisma";
import type { NotificationWithDetails } from "./email";

export interface DiscordNotificationEmbed {
  title: string;
  description: string;
  color: number;
  url?: string;
  fields?: Array<{
    name: string;
    value: string;
    inline?: boolean;
  }>;
  footer?: {
    text: string;
  };
  timestamp?: string;
}

export class DiscordBotService {
  private botToken: string | null;
  private baseUrl: string;
  private discordApiUrl: string = "https://discord.com/api/v10";
  private isConfigured: boolean = false;

  // Rate limiting
  private rateLimitResetTimes: Map<string, number> = new Map();
  private rateLimitRemaining: Map<string, number> = new Map();

  constructor() {
    this.baseUrl = process.env.NEXTAUTH_URL || "http://localhost:3001";
    this.botToken = process.env.DISCORD_BOT_TOKEN || null;
    this.isConfigured = !!this.botToken;

    if (this.isConfigured) {
      console.log("🤖 Discord bot configured with REST API");
    } else {
      console.warn(
        "⚠️ Discord bot token not configured - Discord notifications disabled"
      );
    }
  }

  /**
   * Make a Discord API request with proper headers and rate limiting
   */
  private async discordRequest(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<Response> {
    if (!this.botToken) {
      throw new Error("Discord bot token not configured");
    }

    const url = `${this.discordApiUrl}${endpoint}`;

    // Check rate limits
    await this.checkRateLimit(endpoint);

    const response = await fetch(url, {
      ...options,
      headers: {
        Authorization: `Bot ${this.botToken}`,
        "Content-Type": "application/json",
        ...options.headers,
      },
    });

    // Handle rate limiting
    await this.handleRateLimit(endpoint, response);

    return response;
  }

  /**
   * Check if we're rate limited for this endpoint
   */
  private async checkRateLimit(endpoint: string): Promise<void> {
    const resetTime = this.rateLimitResetTimes.get(endpoint);
    const remaining = this.rateLimitRemaining.get(endpoint);

    if (remaining === 0 && resetTime && Date.now() < resetTime * 1000) {
      const waitTime = resetTime * 1000 - Date.now();
      console.log(`⏳ Rate limited on ${endpoint}, waiting ${waitTime}ms`);
      await new Promise((resolve) => setTimeout(resolve, waitTime));
    }
  }

  /**
   * Handle rate limit headers from Discord response
   */
  private async handleRateLimit(
    endpoint: string,
    response: Response
  ): Promise<void> {
    const resetTime = response.headers.get("x-ratelimit-reset");
    const remaining = response.headers.get("x-ratelimit-remaining");

    if (resetTime) this.rateLimitResetTimes.set(endpoint, parseInt(resetTime));
    if (remaining) this.rateLimitRemaining.set(endpoint, parseInt(remaining));

    // Handle 429 Too Many Requests
    if (response.status === 429) {
      const retryAfter = response.headers.get("retry-after");
      const waitTime = retryAfter ? parseInt(retryAfter) * 1000 : 1000;

      console.warn(`⚠️ Rate limit hit on ${endpoint}, waiting ${waitTime}ms`);
      await new Promise((resolve) => setTimeout(resolve, waitTime));
    }
  }

  /**
   * Send a DM notification to a Discord user
   */
  async sendNotification(
    notification: NotificationWithDetails,
    discordUserId: string,
    deliveryId?: string
  ): Promise<boolean> {
    try {
      if (!this.isConfigured) {
        console.warn("Discord bot not configured");
        return false;
      }

      // Create DM channel with user
      const dmChannel = await this.createDMChannel(discordUserId);
      if (!dmChannel) {
        console.warn(`Failed to create DM channel with user ${discordUserId}`);
        return false;
      }

      // Generate Discord embed
      const embed = this.generateNotificationEmbed(notification);

      // Send message to DM channel
      const success = await this.sendMessageToChannel(dmChannel.id, {
        embeds: [embed],
      });

      if (success) {
        console.log(`🤖 Discord DM sent to user ${discordUserId}`);
        return true;
      } else {
        return false;
      }
    } catch (error: any) {
      console.error(`❌ Failed to send Discord DM to ${discordUserId}:`, error);

      // Handle specific Discord errors
      if (
        error?.code === 50007 ||
        error?.message?.includes("Cannot send messages")
      ) {
        console.warn(
          `User ${discordUserId} has DMs disabled or blocked the bot`
        );
        // Mark user as having DMs disabled
        await this.markUserDMsDisabled(discordUserId);
      }

      return false;
    }
  }

  /**
   * Create a DM channel with a Discord user
   */
  private async createDMChannel(
    discordUserId: string
  ): Promise<{ id: string } | null> {
    try {
      const response = await this.discordRequest("/users/@me/channels", {
        method: "POST",
        body: JSON.stringify({
          recipient_id: discordUserId,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        console.error(
          `Failed to create DM channel with ${discordUserId}:`,
          error
        );
        return null;
      }

      const channel = await response.json();
      return { id: channel.id };
    } catch (error) {
      console.error(`Error creating DM channel with ${discordUserId}:`, error);
      return null;
    }
  }

  /**
   * Send a message to a Discord channel
   */
  private async sendMessageToChannel(
    channelId: string,
    message: any
  ): Promise<boolean> {
    try {
      const response = await this.discordRequest(
        `/channels/${channelId}/messages`,
        {
          method: "POST",
          body: JSON.stringify(message),
        }
      );

      if (!response.ok) {
        const error = await response.json();
        console.error(`Failed to send message to channel ${channelId}:`, error);
        return false;
      }

      return true;
    } catch (error) {
      console.error(`Error sending message to channel ${channelId}:`, error);
      return false;
    }
  }

  /**
   * Get Discord user info by ID (REST API)
   */
  async getDiscordUser(discordUserId: string): Promise<any | null> {
    try {
      const response = await this.discordRequest(`/users/${discordUserId}`);

      if (!response.ok) {
        const error = await response.json();
        console.error(`Failed to fetch Discord user ${discordUserId}:`, error);
        return null;
      }

      return await response.json();
    } catch (error) {
      console.error(`Error fetching Discord user ${discordUserId}:`, error);
      return null;
    }
  }

  /**
   * Generate Discord embed for notification
   */
  private generateNotificationEmbed(
    notification: NotificationWithDetails
  ): any {
    const actorName = this.getActorDisplayName(notification.actor);

    // Base embed structure
    const embed = {
      timestamp: notification.createdAt.toISOString(),
      footer: {
        text: "Library Notifications",
      },
      fields: [] as any[],
    };

    switch (notification.type) {
      case "MENTION":
        return this.generateMentionEmbed(embed, notification, actorName);
      case "COMMENT":
        return this.generateCommentEmbed(embed, notification, actorName);
      case "REPLY":
        return this.generateReplyEmbed(embed, notification, actorName);
      case "SYSTEM":
        return this.generateSystemEmbed(embed, notification);
      default:
        return this.generateGenericEmbed(embed, notification, actorName);
    }
  }

  /**
   * Generate mention embed
   */
  private generateMentionEmbed(
    embed: any,
    notification: NotificationWithDetails,
    actorName: string
  ): any {
    const contextType = notification.post ? "post" : "comment";
    const contextTitle = notification.post?.title || "a comment";

    return {
      ...embed,
      title: "📧 You were mentioned!",
      description: `**${actorName}** mentioned you in a ${contextType}`,
      color: 0x3498db,
      url: this.getActionUrl(notification),
      fields: [
        {
          name: "In",
          value:
            contextType === "post" ? `Post: "${contextTitle}"` : contextTitle,
          inline: true,
        },
        ...(notification.mentionText
          ? [
              {
                name: "Mention",
                value: `"${notification.mentionText}"`,
                inline: false,
              },
            ]
          : []),
      ],
    };
  }

  /**
   * Generate comment embed
   */
  private generateCommentEmbed(
    embed: any,
    notification: NotificationWithDetails,
    actorName: string
  ): any {
    const postTitle = notification.post?.title || "your post";

    return {
      ...embed,
      title: "💬 New Comment",
      description: `**${actorName}** commented on your post`,
      color: 0x2ecc71,
      url: this.getActionUrl(notification),
      fields: [
        {
          name: "Post",
          value: `"${postTitle}"`,
          inline: false,
        },
        ...(notification.message
          ? [
              {
                name: "Comment Preview",
                value:
                  notification.message.substring(0, 200) +
                  (notification.message.length > 200 ? "..." : ""),
                inline: false,
              },
            ]
          : []),
      ],
    };
  }

  /**
   * Generate reply embed
   */
  private generateReplyEmbed(
    embed: any,
    notification: NotificationWithDetails,
    actorName: string
  ): any {
    return {
      ...embed,
      title: "↩️ New Reply",
      description: `**${actorName}** replied to your comment`,
      color: 0x9b59b6,
      url: this.getActionUrl(notification),
      fields: [
        ...(notification.message
          ? [
              {
                name: "Reply Preview",
                value:
                  notification.message.substring(0, 200) +
                  (notification.message.length > 200 ? "..." : ""),
                inline: false,
              },
            ]
          : []),
      ],
    };
  }

  /**
   * Generate system embed
   */
  private generateSystemEmbed(
    embed: any,
    notification: NotificationWithDetails
  ): any {
    return {
      ...embed,
      title: "🔔 System Notification",
      description: notification.title,
      color: 0xe67e22,
      url: this.getActionUrl(notification),
      fields: [
        ...(notification.message
          ? [
              {
                name: "Details",
                value: notification.message,
                inline: false,
              },
            ]
          : []),
      ],
    };
  }

  /**
   * Generate generic embed
   */
  private generateGenericEmbed(
    embed: any,
    notification: NotificationWithDetails,
    actorName: string
  ): any {
    return {
      ...embed,
      title: "🔔 New Notification",
      description: notification.title,
      color: 0x95a5a6,
      url: this.getActionUrl(notification),
      fields: [
        ...(notification.message
          ? [
              {
                name: "Details",
                value: notification.message,
                inline: false,
              },
            ]
          : []),
      ],
    };
  }

  /**
   * Get action URL for notification
   */
  private getActionUrl(notification: NotificationWithDetails): string {
    if (notification.actionUrl) {
      return notification.actionUrl.startsWith("http")
        ? notification.actionUrl
        : `${this.baseUrl}${notification.actionUrl}`;
    }

    if (notification.post) {
      return `${this.baseUrl}/posts/${notification.post.id}${
        notification.commentId ? `#comment-${notification.commentId}` : ""
      }`;
    }

    if (notification.comment) {
      return `${this.baseUrl}/posts/${notification.comment.post.id}#comment-${notification.comment.id}`;
    }

    return this.baseUrl;
  }

  /**
   * Get actor display name
   */
  private getActorDisplayName(
    actor: { name: string | null; email: string | null } | null | undefined
  ): string {
    if (!actor) return "Someone";
    return actor.name || actor.email?.split("@")[0] || "Someone";
  }

  /**
   * Mark user as having DMs disabled
   */
  private async markUserDMsDisabled(discordUserId: string): Promise<void> {
    try {
      await prisma.notificationPreference.updateMany({
        where: {
          discordUserId: discordUserId,
        },
        data: {
          discordEnabled: false,
        },
      });

      console.log(
        `Disabled Discord notifications for user ${discordUserId} (DMs blocked)`
      );
    } catch (error) {
      console.error(
        `Failed to disable Discord for user ${discordUserId}:`,
        error
      );
    }
  }

  /**
   * Send a welcome message when user links their Discord account
   */
  async sendWelcomeMessage(
    discordUserId: string,
    userName: string
  ): Promise<boolean> {
    try {
      if (!this.isConfigured) {
        console.warn("Discord bot not configured");
        return false;
      }

      // Create DM channel with user
      const dmChannel = await this.createDMChannel(discordUserId);
      if (!dmChannel) {
        console.warn(`Failed to create DM channel with user ${discordUserId}`);
        return false;
      }

      const embed = {
        title: "🎉 Discord Account Linked!",
        description: `Hi ${userName}! Your Discord account has been successfully linked to the Library.`,
        color: 0x00ff00,
        fields: [
          {
            name: "What's Next?",
            value:
              "You'll now receive notification DMs here for mentions, comments, and replies.",
            inline: false,
          },
          {
            name: "Manage Preferences",
            value: `Visit ${this.baseUrl}/settings/notifications to customize which notifications you receive.`,
            inline: false,
          },
          {
            name: "Need Help?",
            value:
              "Use `/help` command or visit our support page if you have questions.",
            inline: false,
          },
        ],
        footer: { text: "Library Notifications" },
        timestamp: new Date().toISOString(),
      };

      const success = await this.sendMessageToChannel(dmChannel.id, {
        embeds: [embed],
      });
      if (success) {
        console.log(`🤖 Welcome message sent to user ${discordUserId}`);
        return true;
      } else {
        return false;
      }
    } catch (error) {
      console.error(
        `Failed to send welcome message to ${discordUserId}:`,
        error
      );
      return false;
    }
  }

  /**
   * Test Discord bot configuration
   */
  async testConfiguration(): Promise<boolean> {
    try {
      if (!this.isConfigured) {
        return false;
      }

      // Test bot token by making a simple API request
      const response = await this.discordRequest("/users/@me");
      return response.ok;
    } catch (error) {
      console.error("Discord bot configuration test failed:", error);
      return false;
    }
  }

  /**
   * Get bot configuration info
   */
  async getConfigurationInfo(): Promise<{
    configured: boolean;
    botUser: string | null;
    ready: boolean;
    error?: string;
  }> {
    let botUser = null;
    let error = undefined;

    if (this.isConfigured) {
      try {
        const response = await this.discordRequest("/users/@me");
        if (response.ok) {
          const user = await response.json();
          botUser = `${user.username}#${user.discriminator}`;
        }
      } catch (err: any) {
        error = err.message;
      }
    }

    return {
      configured: this.isConfigured,
      botUser,
      ready: this.isConfigured && !error,
      ...(error && { error }),
    };
  }

  /**
   * Graceful shutdown
   */
  async shutdown(): Promise<void> {
    // Clear rate limit tracking
    this.rateLimitResetTimes.clear();
    this.rateLimitRemaining.clear();
    console.log("🤖 Discord bot service shut down");
  }
}

// Export singleton instance
export const discordBot = new DiscordBotService();
