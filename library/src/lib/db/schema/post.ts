import { pgTable, text, timestamp, integer } from "drizzle-orm/pg-core";
import { usersTable } from "./auth";
import { papersTable } from "./paper";

export const postsTable = pgTable("post", {
  id: text("id")
    .primaryKey()
    .$defaultFn(() => crypto.randomUUID()),
  authorId: text("authorId")
    .references(() => usersTable.id)
    .notNull(),
  title: text("title").notNull(),
  content: text("content"),
  postType: text("postType").default('user-post').notNull(),
  likes: integer("likes").default(0),
  retweets: integer("retweets").default(0),
  replies: integer("replies").default(0),
  paperId: text("paperId")
    .references(() => papersTable.id),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().notNull(),
});
