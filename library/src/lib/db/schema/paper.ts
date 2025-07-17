import { pgTable, text, timestamp, integer, boolean, primaryKey } from "drizzle-orm/pg-core";
import { usersTable } from "./auth";

/**
 * Papers table - stores research papers and their metadata
 * 
 * This is a simplified schema that can be extended later with more normalized
 * tables for authors, institutions, categories, etc.
 */
export const papersTable = pgTable("paper", {
  id: text("id")
    .primaryKey()
    .$defaultFn(() => crypto.randomUUID()),
  title: text("title").notNull(),
  abstract: text("abstract"),
  authors: text("authors").array(), // Simple array for now, can normalize later
  institutions: text("institutions").array(), // Simple array for now
  tags: text("tags").array(), // For categories/topics
  link: text("link"), // URL to paper (arXiv, bioRxiv, etc.)
  source: text("source"), // 'arxiv', 'biorxiv', 'manual'
  externalId: text("externalId"), // arXiv ID, DOI, etc.
  stars: integer("stars").default(0),
  starred: boolean("starred").default(false),
  pdfS3Url: text("pdf_s3_url"), // S3 URL for the raw PDF (future use)
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().notNull(),
});

/**
 * User paper interactions - tracks user-specific actions on papers
 * 
 * This table stores user interactions like starring, reading status,
 * queuing, and personal notes for each paper.
 */
export const userPaperInteractionsTable = pgTable("user_paper_interaction", {
  userId: text("userId")
    .references(() => usersTable.id)
    .notNull(),
  paperId: text("paperId")
    .references(() => papersTable.id)
    .notNull(),
  starred: boolean("starred").default(false),
  readAt: timestamp("readAt"),
  queued: boolean("queued").default(false),
  notes: text("notes"),
}, (table) => ({
  // Create a composite primary key for user-paper interactions
  pk: primaryKey(table.userId, table.paperId),
})); 