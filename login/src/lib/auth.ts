import "server-only";

import { PrismaAdapter } from "@auth/prisma-adapter";
import NextAuth, { NextAuthConfig, NextAuthResult, Session } from "next-auth";
import { Provider } from "next-auth/providers";
import Google from "next-auth/providers/google";
import { redirect } from "next/navigation";

import { prisma } from "./db/prisma";

function buildAuthConfig(): NextAuthConfig {
  const providers: Provider[] = [];

  if (process.env.DEV_MODE === "true") {
    // Fake email provider. For local development only!
    providers.push({
      id: "fake-email",
      type: "email",
      name: "Log magic link to console (dev)",
      async sendVerificationRequest(params) {
        const { url } = params;
        console.log({ url });
      },
    });
  } else {
    // Google OAuth provider for production
    if (process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET) {
      providers.push(
        Google({
          clientId: process.env.GOOGLE_CLIENT_ID,
          clientSecret: process.env.GOOGLE_CLIENT_SECRET,
        })
      );
    }
  }

  const allowedEmailDomains = process.env.ALLOWED_EMAIL_DOMAINS
    ? process.env.ALLOWED_EMAIL_DOMAINS.split(",")
    : ["stem.ai", "softmax.com"];

  const config: NextAuthConfig = {
    adapter: PrismaAdapter(prisma),
    providers,
    callbacks: {
      async signIn({ account, profile }) {
        // adapted from https://authjs.dev/getting-started/providers/google#email-verified
        if (
          process.env.DEV_MODE !== "false" &&
          account?.provider === "google"
        ) {
          return Boolean(
            profile?.email_verified &&
              allowedEmailDomains.some((domain) =>
                profile?.email?.endsWith(`@${domain}`)
              )
          );
        }
        return true;
      },
    },
    session: {
      strategy: "database",
    },
    secret: process.env.NEXTAUTH_SECRET,
  };

  return config;
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