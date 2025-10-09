import "server-only";

import { PrismaAdapter } from "@auth/prisma-adapter";
import NextAuth, { NextAuthConfig, NextAuthResult, Session } from "next-auth";
import { Provider } from "next-auth/providers";
import Google from "next-auth/providers/google";
import { redirect } from "next/navigation";
import type { Adapter, AdapterSession } from "next-auth/adapters";

import { prisma } from "./db/prisma";
import { config } from "./config";
import { Logger } from "./logging/logger";

/**
 * Wrap Prisma adapter to handle missing session deletion gracefully.
 * This prevents errors when trying to delete a non-existent session during magic link login.
 */
function createSafeAdapter(): Adapter {
  const baseAdapter = PrismaAdapter(prisma);

  return {
    ...baseAdapter,
    async deleteSession(
      sessionToken: string
    ): Promise<AdapterSession | null | undefined> {
      try {
        const result = await baseAdapter.deleteSession!(sessionToken);
        return result === undefined ? null : result;
      } catch (error: any) {
        // Ignore "record not found" errors when deleting sessions
        if (error?.code === "P2025") {
          Logger.debug("Session not found for deletion (safe to ignore)", {
            sessionToken,
          });
          return null;
        }
        throw error;
      }
    },
  };
}

function buildAuthConfig(): NextAuthConfig {
  const providers: Provider[] = [];

  if (config.features.devMode) {
    // Fake email provider. For local development only!
    providers.push({
      id: "fake-email",
      type: "email",
      name: "Log magic link to console (dev)",
      async sendVerificationRequest(params) {
        const { url } = params;
        Logger.info(`Magic login link: ${url}`);
      },
    });
  } else {
    // Google OAuth provider for production
    if (config.google.clientId && config.google.clientSecret) {
      providers.push(
        Google({
          clientId: config.google.clientId,
          clientSecret: config.google.clientSecret,
        })
      );
    }
  }

  const allowedEmailDomains =
    config.auth.allowedDomains.length > 0
      ? config.auth.allowedDomains
      : ["stem.ai", "softmax.com"];

  const authConfig: NextAuthConfig = {
    adapter: createSafeAdapter(),
    providers,
    callbacks: {
      async signIn({ account, profile, user }) {
        // Check domain restrictions for Google OAuth
        if (config.features.devMode && account?.provider === "google") {
          const domainCheck = Boolean(
            profile?.email_verified &&
              allowedEmailDomains.some((domain) =>
                profile?.email?.endsWith(`@${domain}`)
              )
          );
          if (!domainCheck) return false;
        }

        // Check if user is banned (prevent banned users from logging in)
        if (user?.id) {
          const userData = await prisma.user.findUnique({
            where: { id: user.id },
            select: {
              isBanned: true,
              banReason: true,
            },
          });

          if (userData?.isBanned) {
            Logger.info(`Banned user attempted login: ${user.email}`);
            // Prevent login by returning false
            return false;
          }
        }

        return true;
      },
      async session({ session, user }) {
        if (session?.user && user?.id) {
          // Add user ID to session for convenience
          session.user.id = user.id;
        }
        return session;
      },
    },
    session: {
      strategy: "database",
    },
    secret: config.auth.secret,
  };

  return authConfig;
}

function makeAuth(): NextAuthResult {
  return NextAuth(buildAuthConfig());
}

export const { handlers, signIn, signOut, auth } = makeAuth();

// Helper functions.

export type SignedInSession = Session & {
  user: NonNullable<Session["user"]> & {
    id: NonNullable<NonNullable<Session["user"]>["id"]>;
    email: NonNullable<NonNullable<Session["user"]>["email"]>;
  };
};

export function isSignedIn(
  session: Session | null
): session is SignedInSession {
  return Boolean(session?.user?.email);
}

export async function getSessionOrRedirect() {
  const session = await auth();
  if (isSignedIn(session)) {
    return session;
  }
  redirect("/api/auth/signin"); // TODO - callbackUrl
}
