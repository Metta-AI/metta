"use client";

// Client-side utilities for other services to validate sessions with the login service

export interface AuthUser {
  id: string;
  email: string;
  name?: string | null;
}

export interface ValidationResponse {
  valid: boolean;
  user?: AuthUser;
  error?: string;
}

export class AuthClient {
  private baseUrl: string;

  constructor(loginServiceUrl = "http://localhost:3002") {
    this.baseUrl = loginServiceUrl;
  }

  async validateCurrentSession(): Promise<ValidationResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/validate`, {
        credentials: "include", // Include cookies for session
      });

      if (!response.ok) {
        return { valid: false, error: "Validation failed" };
      }

      return await response.json();
    } catch (error) {
      console.error("Session validation error:", error);
      return { valid: false, error: "Network error" };
    }
  }

  async validateSessionToken(sessionToken: string): Promise<ValidationResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/validate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ sessionToken }),
      });

      if (!response.ok) {
        return { valid: false, error: "Validation failed" };
      }

      return await response.json();
    } catch (error) {
      console.error("Token validation error:", error);
      return { valid: false, error: "Network error" };
    }
  }

  async getCurrentUser(): Promise<AuthUser | null> {
    try {
      const response = await fetch(`${this.baseUrl}/api/user`, {
        credentials: "include",
      });

      if (!response.ok) {
        return null;
      }

      const data = await response.json();
      return data.user;
    } catch (error) {
      console.error("Get user error:", error);
      return null;
    }
  }

  getSignInUrl(callbackUrl?: string): string {
    const url = new URL(`${this.baseUrl}/api/auth/signin`);
    if (callbackUrl) {
      url.searchParams.set("callbackUrl", callbackUrl);
    }
    return url.toString();
  }

  getSignOutUrl(callbackUrl?: string): string {
    const url = new URL(`${this.baseUrl}/api/auth/signout`);
    if (callbackUrl) {
      url.searchParams.set("callbackUrl", callbackUrl);
    }
    return url.toString();
  }
}

// Default instance
export const authClient = new AuthClient();