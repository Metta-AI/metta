import { db } from "@/lib/db";
import { Paginated, findPaginated, makePaginated } from "@/lib/paginated";
import { postsTable } from "@/lib/db/schema/post";
import { desc } from "drizzle-orm";

type FeedPostRow = typeof postsTable.$inferSelect;

export type FeedPostDTO = {
  id: string;
  title: string;
  createdAt: Date;
  updatedAt: Date;
};

function toDTO(dbModel: FeedPostRow): FeedPostDTO {
  return {
    id: dbModel.id,
    title: dbModel.title,
    createdAt: dbModel.createdAt,
    updatedAt: dbModel.updatedAt,
  };
}

export async function loadFeedPosts({
  limit = 20,
  cursor,
}: {
  limit?: number;
  cursor?: string;
} = {}): Promise<Paginated<FeedPostDTO>> {
  const rows = await db.query.postsTable.findMany({
    where: (postsTable, { gt }) =>
      cursor ? gt(postsTable.id, cursor) : undefined,
    limit,
    orderBy: [desc(postsTable.createdAt)],
    // ...findPaginated(cursor, limit),
  });

  const models = rows.map(toDTO);

  const nextCursor = models[models.length - 1]?.id;
  async function loadMore(limit: number) {
    "use server";
    return loadPosts({ cursor: nextCursor, limit });
  }

  return makePaginated(models, limit, loadMore);
}
