import { pgTable, text, timestamp, unique, foreignKey, integer, boolean, primaryKey } from "drizzle-orm/pg-core"
import { sql } from "drizzle-orm"



export const verificationToken = pgTable("verificationToken", {
	identifier: text().notNull(),
	token: text().notNull(),
	expires: timestamp({ mode: 'string' }).notNull(),
});

export const user = pgTable("user", {
	id: text().primaryKey().notNull(),
	name: text(),
	email: text(),
	emailVerified: timestamp({ mode: 'string' }),
	image: text(),
}, (table) => [
	unique("user_email_unique").on(table.email),
]);

export const account = pgTable("account", {
	userId: text().notNull(),
	type: text().notNull(),
	provider: text().notNull(),
	providerAccountId: text().notNull(),
	refreshToken: text("refresh_token"),
	accessToken: text("access_token"),
	expiresAt: integer("expires_at"),
	tokenType: text("token_type"),
	scope: text(),
	idToken: text("id_token"),
	sessionState: text("session_state"),
}, (table) => [
	foreignKey({
			columns: [table.userId],
			foreignColumns: [user.id],
			name: "account_userId_user_id_fk"
		}).onDelete("cascade"),
]);

export const authenticator = pgTable("authenticator", {
	credentialId: text().notNull(),
	userId: text().notNull(),
	providerAccountId: text().notNull(),
	credentialPublicKey: text().notNull(),
	counter: integer().notNull(),
	credentialDeviceType: text().notNull(),
	credentialBackedUp: boolean().notNull(),
	transports: text(),
}, (table) => [
	foreignKey({
			columns: [table.userId],
			foreignColumns: [user.id],
			name: "authenticator_userId_user_id_fk"
		}).onDelete("cascade"),
	unique("authenticator_credentialID_unique").on(table.credentialId),
]);

export const session = pgTable("session", {
	sessionToken: text().primaryKey().notNull(),
	userId: text().notNull(),
	expires: timestamp({ mode: 'string' }).notNull(),
}, (table) => [
	foreignKey({
			columns: [table.userId],
			foreignColumns: [user.id],
			name: "session_userId_user_id_fk"
		}).onDelete("cascade"),
]);

export const post = pgTable("post", {
	id: text().primaryKey().notNull(),
	authorId: text().notNull(),
	title: text().notNull(),
	createdAt: timestamp({ mode: 'string' }).defaultNow().notNull(),
	updatedAt: timestamp({ mode: 'string' }).defaultNow().notNull(),
}, (table) => [
	foreignKey({
			columns: [table.authorId],
			foreignColumns: [user.id],
			name: "post_authorId_user_id_fk"
		}),
]);

export const paper = pgTable("paper", {
	id: text().primaryKey().notNull(),
	title: text().notNull(),
	abstract: text(),
	authors: text().array(),
	institutions: text().array(),
	tags: text().array(),
	link: text(),
	source: text(),
	externalId: text(),
	stars: integer().default(0),
	starred: boolean().default(false),
	pdfS3Url: text("pdf_s3_url"),
	createdAt: timestamp({ mode: 'string' }).defaultNow().notNull(),
	updatedAt: timestamp({ mode: 'string' }).defaultNow().notNull(),
});

export const userPaperInteraction = pgTable("user_paper_interaction", {
	userId: text().notNull(),
	paperId: text().notNull(),
	starred: boolean().default(false),
	readAt: timestamp({ mode: 'string' }),
	queued: boolean().default(false),
	notes: text(),
}, (table) => [
	foreignKey({
			columns: [table.userId],
			foreignColumns: [user.id],
			name: "user_paper_interaction_userId_user_id_fk"
		}),
	foreignKey({
			columns: [table.paperId],
			foreignColumns: [paper.id],
			name: "user_paper_interaction_paperId_paper_id_fk"
		}),
	primaryKey({ columns: [table.userId, table.paperId], name: "user_paper_interaction_userId_paperId_pk"}),
]);
