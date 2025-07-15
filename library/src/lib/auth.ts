import "server-only";

import NextAuth, { NextAuthConfig, NextAuthResult, Session } from "next-auth";
import { Provider } from "next-auth/providers";
import { DrizzleAdapter } from "@auth/drizzle-adapter";
import { db } from "./db";
import { redirect } from "next/navigation";

function buildAuthConfig(): NextAuthConfig {
  const providers: Provider[] = [];

  if (process.env.DEV_MODE === "true") {
    // Fake email provider. For local development only!
    providers.push({
      id: "fake-email",
      type: "email",
      name: "Log magic link to console (dev)",
      async sendVerificationRequest(params) {
        const { identifier: to, provider, url, theme } = params;
        const { host } = new URL(url);
        console.log({ url });
      },
    });
  }

  // TODO: configure Google provider for production.

  const config: NextAuthConfig = {
    adapter: DrizzleAdapter(db),
    providers,
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
