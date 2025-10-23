/**
 * Type definitions for Prisma models used in data layer
 * These types represent the structure of data returned from Prisma queries
 */

import type { LLMAbstract } from "@/lib/llm-abstract-generator-clean";

/**
 * User model from Prisma with minimal fields
 */
export interface PrismaUser {
  id: string;
  name: string | null;
  email: string | null;
  image: string | null;
}

/**
 * Author model from Prisma
 */
export interface PrismaAuthor {
  id: string;
  name: string;
  orcid: string | null;
  institution: string | null;
}

/**
 * Institution model from Prisma
 */
export interface PrismaInstitution {
  name: string;
}

/**
 * Paper-Author join table
 */
export interface PrismaPaperAuthor {
  author: PrismaAuthor;
}

/**
 * Paper-Institution join table
 */
export interface PrismaPaperInstitution {
  institution: PrismaInstitution;
}

/**
 * User-Paper interaction
 */
export interface PrismaUserPaperInteraction {
  userId: string;
  paperId: string;
  starred: boolean;
  queued: boolean;
}

/**
 * Paper model from Prisma with relations
 */
export interface PrismaPaper {
  id: string;
  title: string;
  abstract: string | null;
  tags: string[] | null;
  link: string | null;
  source: string | null;
  externalId: string | null;
  createdAt: Date;
  updatedAt: Date;
  llmAbstract: LLMAbstract | null;
  llmAbstractGeneratedAt: Date | null;
  paperAuthors?: PrismaPaperAuthor[];
  paperInstitutions?: PrismaPaperInstitution[];
  userPaperInteractions?: PrismaUserPaperInteraction[];
}

/**
 * Comment model from Prisma
 */
export interface PrismaComment {
  id: string;
  content: string;
  createdAt: Date;
  authorId: string;
}

/**
 * Quoted post reference
 */
export interface PrismaQuotedPost {
  id: string;
  title: string;
  content: string | null;
  authorId: string;
  createdAt: Date;
  author?: PrismaUser;
}

/**
 * Post model from Prisma with relations
 */
export interface PrismaPost {
  id: string;
  title: string;
  content: string | null;
  images: string[] | null;
  postType: "user-post" | "paper-post" | "pure-paper" | "quote-post";
  authorId: string;
  paperId: string | null;
  createdAt: Date;
  updatedAt: Date;
  queues?: number;
  replies?: number;
  quotedPostIds?: string[];
  quotedPosts?: PrismaQuotedPost[];
  comments?: PrismaComment[];
}

/**
 * Paper with institution data (for author queries)
 */
export interface PrismaPaperWithInstitutions {
  id: string;
  title: string;
  link: string | null;
  createdAt: Date;
  stars: number;
  paperInstitutions: PrismaPaperInstitution[];
}

/**
 * Paper-Author join for author queries
 */
export interface PrismaPaperAuthorForAuthor {
  paper: PrismaPaperWithInstitutions;
}

/**
 * Full Author model from Prisma with relations
 */
export interface PrismaAuthorWithRelations {
  id: string;
  name: string;
  username: string | null;
  email: string | null;
  avatar: string | null;
  institution: string | null;
  department: string | null;
  title: string | null;
  expertise: string[];
  hIndex: number | null;
  totalCitations: number | null;
  claimed: boolean;
  recentActivity: Date | null;
  orcid: string | null;
  googleScholarId: string | null;
  arxivId: string | null;
  createdAt: Date;
  updatedAt: Date;
  paperAuthors: PrismaPaperAuthorForAuthor[];
}
