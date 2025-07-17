import { pgTable, text, timestamp } from "drizzle-orm/pg-core";
import { usersTable } from "./auth";

export const postsTable = pgTable("post", {
  id: text("id")
    .primaryKey()
    .$defaultFn(() => crypto.randomUUID()),
  authorId: text("authorId")
    .references(() => usersTable.id)
    .notNull(),
  title: text("title").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().notNull(),
});
