/**
 * Discord OAuth Integration
 *
 * Handles Discord OAuth flow for account linking and user authentication
 */

export interface DiscordUser {
  id: string;
  username: string;
  discriminator: string;
  global_name?: string | null;
  avatar?: string | null;
  email?: string | null;
}

export interface DiscordTokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  refresh_token: string;
  scope: string;
}

export class DiscordAuthService {
  private clientId: string;
  private clientSecret: string;
  private redirectUri: string;

  constructor() {
    this.clientId = process.env.DISCORD_CLIENT_ID!;
    this.clientSecret = process.env.DISCORD_CLIENT_SECRET!;
    this.redirectUri = process.env.DISCORD_REDIRECT_URI!;

    if (!this.clientId || !this.clientSecret || !this.redirectUri) {
      throw new Error(
        "Discord OAuth not configured. Missing DISCORD_CLIENT_ID, DISCORD_CLIENT_SECRET, or DISCORD_REDIRECT_URI"
      );
    }
  }

  /**
   * Generate Discord OAuth authorization URL
   */
  getAuthorizationUrl(state?: string): string {
    const params = new URLSearchParams({
      client_id: this.clientId,
      redirect_uri: this.redirectUri,
      response_type: "code",
      scope: "identify", // We only need basic user info
      ...(state && { state }),
    });

    return `https://discord.com/api/oauth2/authorize?${params.toString()}`;
  }

  /**
   * Exchange authorization code for access token
   */
  async exchangeCodeForToken(code: string): Promise<DiscordTokenResponse> {
    const params = new URLSearchParams({
      client_id: this.clientId,
      client_secret: this.clientSecret,
      grant_type: "authorization_code",
      code,
      redirect_uri: this.redirectUri,
    });

    const response = await fetch("https://discord.com/api/oauth2/token", {
      method: "POST",
      body: params,
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Failed to exchange code for token: ${error}`);
    }

    return response.json();
  }

  /**
   * Get Discord user info using access token
   */
  async getDiscordUser(accessToken: string): Promise<DiscordUser> {
    const response = await fetch("https://discord.com/api/v10/users/@me", {
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Failed to get Discord user info: ${error}`);
    }

    return response.json();
  }

  /**
   * Revoke Discord access token (for unlinking)
   */
  async revokeToken(accessToken: string): Promise<boolean> {
    try {
      const params = new URLSearchParams({
        client_id: this.clientId,
        client_secret: this.clientSecret,
        token: accessToken,
      });

      const response = await fetch(
        "https://discord.com/api/oauth2/token/revoke",
        {
          method: "POST",
          body: params,
          headers: {
            "Content-Type": "application/x-www-form-urlencoded",
          },
        }
      );

      return response.ok;
    } catch (error) {
      console.error("Failed to revoke Discord token:", error);
      return false;
    }
  }

  /**
   * Get Discord username display format
   */
  static formatDiscordUsername(user: DiscordUser): string {
    // New Discord usernames don't have discriminators
    if (user.discriminator === "0") {
      return `@${user.username}`;
    }
    // Legacy format with discriminator
    return `${user.username}#${user.discriminator}`;
  }

  /**
   * Get Discord user display name
   */
  static getDisplayName(user: DiscordUser): string {
    return user.global_name || user.username;
  }

  /**
   * Check if Discord OAuth is properly configured
   */
  isConfigured(): boolean {
    return !!(this.clientId && this.clientSecret && this.redirectUri);
  }

  /**
   * Get configuration info for debugging
   */
  getConfigurationInfo(): {
    configured: boolean;
    clientId: string | null;
    redirectUri: string | null;
  } {
    return {
      configured: this.isConfigured(),
      clientId: this.clientId || null,
      redirectUri: this.redirectUri || null,
    };
  }
}

// Export singleton instance
export const discordAuth = new DiscordAuthService();




