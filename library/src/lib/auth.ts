import "server-only";

import { PrismaAdapter } from "@auth/prisma-adapter";
import NextAuth, { NextAuthConfig, NextAuthResult, Session } from "next-auth";
import { Provider } from "next-auth/providers";
import Google from "next-auth/providers/google";
import { redirect } from "next/navigation";

import { prisma } from "./db/prisma";
import { config } from "./config";
import { Logger } from "./logging/logger";

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
        Logger.info({ url });
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
    adapter: PrismaAdapter(prisma),
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
