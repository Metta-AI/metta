import { desc, eq, inArray } from "drizzle-orm";

import { db } from "@/lib/db";
import { postsTable } from "@/lib/db/schema/post";
import { usersTable } from "@/lib/db/schema/auth";
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

export function toFeedPostDTO(dbModel: FeedPostRow, usersMap: Map<string, any>): FeedPostDTO {
  const author = usersMap.get(dbModel.authorId);
  
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
    createdAt: dbModel.createdAt,
    updatedAt: dbModel.updatedAt,
  };
}

export async function loadFeedPosts({
  limit = 5,
  cursor,
}: {
  limit?: number;
  cursor?: Date;
} = {}): Promise<Paginated<FeedPostDTO>> {
  const rows = await db
    .select()
    .from(postsTable)
    .where(cursor ? desc(postsTable.createdAt) : undefined)
    .limit(limit + 1)
    .orderBy(desc(postsTable.createdAt));

  // Get unique author IDs
  const authorIds = [...new Set(rows.map(row => row.authorId))];
  
  // Fetch all users who authored these posts
  const users = authorIds.length > 0 ? await db
    .select()
    .from(usersTable)
    .where(inArray(usersTable.id, authorIds)) : [];
  
  // Create a map for quick user lookup
  const usersMap = new Map(users.map(user => [user.id, user]));

  const posts = rows.map(row => toFeedPostDTO(row, usersMap));

  const nextCursor = posts[posts.length - 1]?.createdAt;
  async function loadMore(limit: number) {
    "use server";
    return loadFeedPosts({ cursor: nextCursor, limit });
  }

  return makePaginated(posts, limit, loadMore);
}
