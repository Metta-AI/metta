import { desc, eq, inArray, lt } from "drizzle-orm";

import { db } from "@/lib/db";
import { postsTable } from "@/lib/db/schema/post";
import { usersTable } from "@/lib/db/schema/auth";
import { papersTable } from "@/lib/db/schema/paper";
import { makePaginated, Paginated } from "@/lib/paginated";

type FeedPostRow = typeof postsTable.$inferSelect;

export type FeedPostDTO = {
  id: string;
  title: string;
  content: string | null;
  postType: 'user-post' | 'paper-post' | 'pure-paper';
  likes: number;
  retweets: number;
  replies: number;
  author: {
    id: string;
    name: string | null;
    email: string | null;
    image: string | null;
  };
  paper?: {
    id: string;
    title: string;
    abstract: string | null;
    authors: string[] | null;
    institutions: string[] | null;
    tags: string[] | null;
    link: string | null;
    source: string | null;
    externalId: string | null;
    stars: number;
    starred: boolean;
    pdfS3Url: string | null;
    createdAt: Date;
    updatedAt: Date;
  };
  createdAt: Date;
  updatedAt: Date;
};

export function toFeedPostDTO(dbModel: FeedPostRow, usersMap: Map<string, any>, papersMap: Map<string, any>): FeedPostDTO {
  const author = usersMap.get(dbModel.authorId);
  const paper = dbModel.paperId ? papersMap.get(dbModel.paperId) : null;
  
  return {
    id: dbModel.id,
    title: dbModel.title,
    content: dbModel.content,
    postType: dbModel.postType as 'user-post' | 'paper-post' | 'pure-paper',
    likes: dbModel.likes ?? 0,
    retweets: dbModel.retweets ?? 0,
    replies: dbModel.replies ?? 0,
    author: {
      id: dbModel.authorId,
      name: author?.name ?? null,
      email: author?.email ?? null,
      image: author?.image ?? null,
    },
    paper: paper ? {
      id: paper.id,
      title: paper.title,
      abstract: paper.abstract,
      authors: paper.authors,
      institutions: paper.institutions,
      tags: paper.tags,
      link: paper.link,
      source: paper.source,
      externalId: paper.externalId,
      stars: paper.stars ?? 0,
      starred: false, // TODO: Get from user interactions
      pdfS3Url: paper.pdfS3Url,
      createdAt: paper.createdAt,
      updatedAt: paper.updatedAt,
    } : undefined,
    createdAt: dbModel.createdAt,
    updatedAt: dbModel.updatedAt,
  };
}

export async function loadFeedPosts({
  limit = 10,
  cursor,
}: {
  limit?: number;
  cursor?: Date;
} = {}): Promise<Paginated<FeedPostDTO>> {
  // Build the query with proper cursor-based pagination
  const rows = await db
    .select()
    .from(postsTable)
    .where(cursor ? lt(postsTable.createdAt, cursor) : undefined)
    .limit(limit + 1)
    .orderBy(desc(postsTable.createdAt));

  // Get unique author IDs
  const authorIds = [...new Set(rows.map(row => row.authorId))];
  
  // Get unique paper IDs
  const paperIds = [...new Set(rows.map(row => row.paperId).filter((id): id is string => id !== null))];
  
  // Fetch all users who authored these posts
  const users = authorIds.length > 0 ? await db
    .select()
    .from(usersTable)
    .where(inArray(usersTable.id, authorIds)) : [];
  
  // Fetch all papers referenced in these posts
  const papers = paperIds.length > 0 ? await db
    .select()
    .from(papersTable)
    .where(inArray(papersTable.id, paperIds)) : [];
  
  // Create maps for quick lookup
  const usersMap = new Map(users.map(user => [user.id, user]));
  const papersMap = new Map(papers.map(paper => [paper.id, paper]));

  const posts = rows.map(row => toFeedPostDTO(row, usersMap, papersMap));

  // Check if there are more posts to load
  const hasMore = posts.length > limit;
  const nextCursor = hasMore ? posts[limit - 1]?.createdAt : undefined;

  async function loadMore(limit: number) {
    "use server";
    return loadFeedPosts({ cursor: nextCursor, limit });
  }

  return makePaginated(posts, limit, loadMore);
}
