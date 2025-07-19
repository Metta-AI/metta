import { prisma } from "@/lib/db/prisma";

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
    createdAt: Date;
    updatedAt: Date;
  };
  createdAt: Date;
  updatedAt: Date;
};

export function toFeedPostDTOPrisma(dbModel: any, usersMap: Map<string, any>, papersMap: Map<string, any>): FeedPostDTO {
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
      createdAt: paper.createdAt,
      updatedAt: paper.updatedAt,
    } : undefined,
    createdAt: dbModel.createdAt,
    updatedAt: dbModel.updatedAt,
  };
}

export async function loadFeedPostsPrisma({
  limit = 10,
  cursor,
}: {
  limit?: number;
  cursor?: Date;
} = {}): Promise<FeedPostDTO[]> {
  // Build the query with proper cursor-based pagination
  const rows = await prisma.post.findMany({
    where: cursor ? {
      createdAt: {
        lt: cursor,
      },
    } : undefined,
    take: limit + 1,
    orderBy: {
      createdAt: 'desc',
    },
    include: {
      author: true,
      paper: true,
    },
  });

  // Transform the data to match the expected format
  const posts = rows.map(row => {
    // Create maps for the lookup (maintaining compatibility with existing toFeedPostDTO function)
    const usersMap = new Map();
    if (row.author) {
      usersMap.set(row.author.id, row.author);
    }

    const papersMap = new Map();
    if (row.paper) {
      papersMap.set(row.paper.id, row.paper);
    }

    return toFeedPostDTOPrisma(row, usersMap, papersMap);
  });

  return posts;
} 