import { relations } from "drizzle-orm/relations";
import { user, account, authenticator, session, post, userPaperInteraction, paper } from "./schema";

export const accountRelations = relations(account, ({one}) => ({
	user: one(user, {
		fields: [account.userId],
		references: [user.id]
	}),
}));

export const userRelations = relations(user, ({many}) => ({
	accounts: many(account),
	authenticators: many(authenticator),
	sessions: many(session),
	posts: many(post),
	userPaperInteractions: many(userPaperInteraction),
}));

export const authenticatorRelations = relations(authenticator, ({one}) => ({
	user: one(user, {
		fields: [authenticator.userId],
		references: [user.id]
	}),
}));

export const sessionRelations = relations(session, ({one}) => ({
	user: one(user, {
		fields: [session.userId],
		references: [user.id]
	}),
}));

export const postRelations = relations(post, ({one}) => ({
	user: one(user, {
		fields: [post.authorId],
		references: [user.id]
	}),
	paper: one(paper, {
		fields: [post.paperId],
		references: [paper.id]
	}),
}));

export const userPaperInteractionRelations = relations(userPaperInteraction, ({one}) => ({
	user: one(user, {
		fields: [userPaperInteraction.userId],
		references: [user.id]
	}),
	paper: one(paper, {
		fields: [userPaperInteraction.paperId],
		references: [paper.id]
	}),
}));

export const paperRelations = relations(paper, ({many}) => ({
	userPaperInteractions: many(userPaperInteraction),
	posts: many(post),
}));