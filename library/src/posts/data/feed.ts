import { desc } from "drizzle-orm";

import { db } from "@/lib/db";
import { postsTable } from "@/lib/db/schema/post";
import { makePaginated, Paginated } from "@/lib/paginated";

type FeedPostRow = typeof postsTable.$inferSelect;

export type FeedPostDTO = {
  id: string;
  title: string;
  createdAt: Date;
  updatedAt: Date;
};

export function toFeedPostDTO(dbModel: FeedPostRow): FeedPostDTO {
  return {
    id: dbModel.id,
    title: dbModel.title,
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
  const rows = await db.query.postsTable.findMany({
    where: (postsTable, { lt }) =>
      cursor ? lt(postsTable.createdAt, cursor) : undefined,
    limit: limit + 1,
    orderBy: [desc(postsTable.createdAt)],
  });

  const posts = rows.map(toFeedPostDTO);

  const nextCursor = posts[posts.length - 1]?.createdAt;
  async function loadMore(limit: number) {
    "use server";
    return loadFeedPosts({ cursor: nextCursor, limit });
  }

  return makePaginated(posts, limit, loadMore);
}
