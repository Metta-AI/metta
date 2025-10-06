import { db } from "@/lib/db";

interface ListPostsOptions {
  limit?: number;
  offset?: number;
}

export async function listPosts(options: ListPostsOptions = {}) {
  const { limit = 20, offset = 0 } = options;

  const [posts, total] = await Promise.all([
    db.post.findMany({
      skip: offset,
      take: limit,
      orderBy: { createdAt: "desc" },
      select: {
        id: true,
        title: true,
        createdAt: true,
        content: true,
        images: true,
        quotedPostIds: true,
        paperId: true,
        author: {
          select: {
            id: true,
            name: true,
          },
        },
        paper: {
          select: {
            stars: true,
          },
        },
      },
    }),
    db.post.count(),
  ]);

  const summaries = posts.map((post) => ({
    id: post.id,
    title: post.title,
    createdAt: post.createdAt.toISOString(),
    author: {
      id: post.author.id,
      name: post.author.name,
    },
    paperId: post.paperId,
    stars: post.paper?.stars ?? 0,
  }));

  return {
    posts: summaries,
    pagination: {
      limit,
      offset,
      total,
    },
  };
}
