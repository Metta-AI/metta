import { prisma } from "@/lib/db/prisma";
import { makePaginated, Paginated } from "@/lib/paginated";
import { auth } from "@/lib/auth";

export type FeedPostDTO = {
  id: string;
  title: string;
  content: string | null;
  postType: "user-post" | "paper-post" | "pure-paper";
  queues: number;
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
    authors?: {
      id: string;
      name: string;
      orcid?: string | null;
      institution?: string | null;
    }[];
    institutions: string[] | null;
    tags: string[] | null;
    link: string | null;
    source: string | null;
    externalId: string | null;
    stars: number;
    starred: boolean;
    queued: boolean;
    createdAt: Date;
    updatedAt: Date;
    llmAbstract?: any; // LLM-generated enhanced abstract
    llmAbstractGeneratedAt?: Date | null;
  };
  createdAt: Date;
  updatedAt: Date;
  lastActivityAt: Date; // Most recent activity (post creation or last comment)
};

export function toFeedPostDTO(
  dbModel: any,
  usersMap: Map<string, any>,
  papersMap: Map<string, any>,
  userPaperInteractionsMap: Map<string, any> = new Map()
): FeedPostDTO {
  const author = usersMap.get(dbModel.authorId);
  const paper = dbModel.paperId ? papersMap.get(dbModel.paperId) : null;

  // Calculate the most recent activity timestamp
  // This is the max of post creation time and the most recent comment time
  let lastActivityAt = dbModel.createdAt;
  if (dbModel.comments && dbModel.comments.length > 0) {
    const mostRecentCommentTime = dbModel.comments.reduce(
      (latest: Date, comment: any) => {
        return comment.createdAt > latest ? comment.createdAt : latest;
      },
      new Date(0)
    );
    lastActivityAt =
      mostRecentCommentTime > dbModel.createdAt
        ? mostRecentCommentTime
        : dbModel.createdAt;
  }

  return {
    id: dbModel.id,
    title: dbModel.title,
    content: dbModel.content,
    postType: dbModel.postType as "user-post" | "paper-post" | "pure-paper",
    queues: dbModel.queues ?? 0,
    replies: dbModel.replies ?? 0,
    author: {
      id: dbModel.authorId,
      name: author?.name ?? null,
      email: author?.email ?? null,
      image: author?.image ?? null,
    },
    paper: paper
      ? {
          id: paper.id,
          title: paper.title,
          abstract: paper.abstract,
          authors:
            paper.paperAuthors?.map((pa: any) => ({
              id: pa.author.id,
              name: pa.author.name,
              orcid: pa.author.orcid,
              institution: pa.author.institution,
            })) || [],
          institutions: paper.institutions,
          tags: paper.tags,
          link: paper.link,
          source: paper.source,
          externalId: paper.externalId,
          stars: paper.userPaperInteractions?.length ?? 0,
          starred: userPaperInteractionsMap.get(paper.id)?.starred ?? false,
          queued: userPaperInteractionsMap.get(paper.id)?.queued ?? false,
          createdAt: paper.createdAt,
          updatedAt: paper.updatedAt,
          llmAbstract: paper.llmAbstract,
          llmAbstractGeneratedAt: paper.llmAbstractGeneratedAt,
        }
      : undefined,
    createdAt: dbModel.createdAt,
    updatedAt: dbModel.updatedAt,
    lastActivityAt: lastActivityAt,
  };
}

export async function loadFeedPosts({
  limit = 10,
  cursor,
}: {
  limit?: number;
  cursor?: Date;
} = {}): Promise<Paginated<FeedPostDTO>> {
  // Get current user session
  const session = await auth();

  // Fetch all posts with comments for activity-based sorting
  // We can't easily sort by calculated field in Prisma, so we'll sort in memory
  const allRows = await prisma.post.findMany({
    orderBy: [
      {
        createdAt: "desc",
      },
      {
        id: "desc", // Secondary sort by ID to ensure consistent ordering
      },
    ],
    include: {
      author: true,
      comments: {
        select: {
          createdAt: true,
        },
        orderBy: {
          createdAt: "desc",
        },
        take: 1, // Only need the most recent comment for each post
      },
      paper: {
        select: {
          id: true,
          title: true,
          abstract: true,
          institutions: true,
          tags: true,
          link: true,
          source: true,
          externalId: true,
          createdAt: true,
          updatedAt: true,
          llmAbstract: true,
          llmAbstractGeneratedAt: true,
          paperAuthors: {
            include: {
              author: {
                select: {
                  id: true,
                  name: true,
                  orcid: true,
                  institution: true,
                },
              },
            },
          },
          userPaperInteractions: {
            where: {
              starred: true,
            },
            select: {
              userId: true,
            },
          },
        },
      },
    },
  });

  // Calculate lastActivityAt for each post and sort by it
  const postsWithActivity = allRows
    .map((row) => ({
      ...row,
      lastActivityAt:
        row.comments.length > 0
          ? row.comments[0].createdAt > row.createdAt
            ? row.comments[0].createdAt
            : row.createdAt
          : row.createdAt,
    }))
    .sort((a, b) => b.lastActivityAt.getTime() - a.lastActivityAt.getTime());

  // Apply cursor-based pagination
  let filteredRows = postsWithActivity;
  if (cursor) {
    filteredRows = postsWithActivity.filter(
      (row) => row.lastActivityAt < cursor
    );
  }

  // Take the requested number + 1 for pagination
  const rows = filteredRows.slice(0, limit + 1);

  // Fetch user paper interactions if user is authenticated
  let userPaperInteractionsMap = new Map<string, any>();

  if (session?.user?.id) {
    // Fetch user paper interactions (for starred/queued status)
    const paperIds = rows
      .filter((row) => row.paperId)
      .map((row) => row.paperId!)
      .filter((id, index, self) => self.indexOf(id) === index); // Remove duplicates

    if (paperIds.length > 0) {
      const userPaperInteractions = await prisma.userPaperInteraction.findMany({
        where: {
          userId: session.user.id,
          paperId: {
            in: paperIds,
          },
        },
      });

      userPaperInteractionsMap = new Map(
        userPaperInteractions.map((interaction) => [
          interaction.paperId,
          interaction,
        ])
      );
    }
  }

  // Transform the data to match the expected format
  const posts = rows.map((row) => {
    // Create maps for the lookup (maintaining compatibility with existing toFeedPostDTO function)
    const usersMap = new Map();
    if (row.author) {
      usersMap.set(row.author.id, row.author);
    }

    const papersMap = new Map();
    if (row.paper) {
      papersMap.set(row.paper.id, row.paper);
    }

    return toFeedPostDTO(row, usersMap, papersMap, userPaperInteractionsMap);
  });

  // Check if there are more posts to load
  const hasMore = posts.length > limit;
  const nextCursor = hasMore ? posts[limit - 1]?.lastActivityAt : undefined;

  async function loadMore(limit: number) {
    "use server";
    return loadFeedPosts({ cursor: nextCursor, limit });
  }

  return makePaginated(posts, limit, loadMore);
}
