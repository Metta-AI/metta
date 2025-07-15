import "server-only";

import NextAuth, { NextAuthConfig, NextAuthResult } from "next-auth";
import { Provider } from "next-auth/providers";
import { DrizzleAdapter } from "@auth/drizzle-adapter";
import Credentials from "next-auth/providers/credentials";
import Resend from "next-auth/providers/resend";
import { db } from "./db";
import { usersTable } from "./db/schema/auth";

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
