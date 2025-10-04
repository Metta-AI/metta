import "server-only";

// Server-side utilities for other services to validate sessions with the login service

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

export class ServerAuthClient {
  private baseUrl: string;

  constructor(loginServiceUrl = "http://localhost:3002") {
    this.baseUrl = loginServiceUrl;
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

  async validateSessionFromCookies(cookies: string): Promise<ValidationResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/validate`, {
        headers: {
          Cookie: cookies,
        },
      });

      if (!response.ok) {
        return { valid: false, error: "Validation failed" };
      }

      return await response.json();
    } catch (error) {
      console.error("Cookie validation error:", error);
      return { valid: false, error: "Network error" };
    }
  }

  async healthCheck(): Promise<{ healthy: boolean; error?: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/health`);
      const data = await response.json();

      return {
        healthy: response.ok && data.status === "healthy",
        error: data.error,
      };
    } catch (error) {
      return {
        healthy: false,
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  }
}

// Default instance
export const serverAuthClient = new ServerAuthClient();

// Middleware helper for Next.js apps
export async function withAuth(
  request: Request,
  handler: (user: AuthUser) => Promise<Response> | Response
): Promise<Response> {
  const cookies = request.headers.get("cookie") || "";
  const validation = await serverAuthClient.validateSessionFromCookies(cookies);

  if (!validation.valid || !validation.user) {
    return new Response(JSON.stringify({ error: "Unauthorized" }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    });
  }

  return handler(validation.user);
}