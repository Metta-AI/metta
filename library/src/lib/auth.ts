import "server-only";

import NextAuth, { NextAuthConfig, NextAuthResult } from "next-auth";
import { Provider } from "next-auth/providers";
import { DrizzleAdapter } from "@auth/drizzle-adapter";
import Credentials from "next-auth/providers/credentials";
import { db } from "./db";
import { usersTable } from "./db/schema/auth";

function buildAuthConfig(): NextAuthConfig {
	const providers: Provider[] = [];

	if (process.env.DEV_MODE === "true") {
		providers.push(
			// Fake credentials provider, registers any email without doing any checks.
			// For local development only!
			Credentials({
				credentials: {
					email: {
						type: "email",
						label: "Email",
						placeholder: "somebody@softmax.com",
					},
				},
				async authorize(credentials) {
					if (!credentials?.email || typeof credentials.email !== "string") {
						return null;
					}

					// Create user with drizzle
					const [user] = await db
						.insert(usersTable)
						.values({
							email: credentials.email,
						})
						.returning();

					return {
						id: user.id,
						email: user.email,
						name: user.name,
						image: user.image,
					};
				},
			}),
		);
	}

	const config: NextAuthConfig = {
		adapter: DrizzleAdapter(db),
		providers,
	};

	return config;
}

function makeAuth(): NextAuthResult {
	return NextAuth(buildAuthConfig());
}

export const { handlers, signIn, signOut, auth } = makeAuth();
