import { prisma } from "@/lib/db/prisma";
import { makePaginated, Paginated } from "@/lib/paginated";
import { auth } from "@/lib/auth";
import type { LLMAbstract } from "@/lib/llm-abstract-generator-clean";
import type {
  PrismaPost,
  PrismaPaper,
  PrismaUser,
  PrismaUserPaperInteraction,
} from "@/types/prisma-models";

export type FeedPostDTO = {
  id: string;
  title: string;
  content: string | null;
  images: string[];
  postType: "user-post" | "paper-post" | "pure-paper" | "quote-post";
  replies: number;
  quotedPostIds: string[];
  quotedPosts?: {
    id: string;
    title: string;
    content: string | null;
    author: {
      id: string;
      name: string | null;
      email: string | null;
      image: string | null;
    };
    createdAt: Date;
  }[];
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
    institutions: string[];
    tags: string[] | null;
    link: string | null;
    source: string | null;
    externalId: string | null;
    stars: number;
    starred: boolean;
    createdAt: Date;
    updatedAt: Date;
    llmAbstract?: LLMAbstract | null;
    llmAbstractGeneratedAt?: Date | null;
  };
  createdAt: Date;
  updatedAt: Date;
  lastActivityAt: Date; // Most recent activity (post creation or last comment)
};

export function toFeedPostDTO(
  dbModel: PrismaPost,
  usersMap: Map<string, PrismaUser>,
  papersMap: Map<string, PrismaPaper>,
  userPaperInteractionsMap: Map<string, PrismaUserPaperInteraction> = new Map()
): FeedPostDTO {
  const author = usersMap.get(dbModel.authorId);
  const paper = dbModel.paperId ? papersMap.get(dbModel.paperId) : null;

  // Calculate the most recent activity timestamp
  // This is the max of post creation time and the most recent comment time
  let lastActivityAt = dbModel.createdAt;
  if (dbModel.comments && dbModel.comments.length > 0) {
    const mostRecentCommentTime = dbModel.comments.reduce(
      (latest: Date, comment) => {
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
    images: dbModel.images ?? [],
    postType: dbModel.postType as
      | "user-post"
      | "paper-post"
      | "pure-paper"
      | "quote-post",
    replies: dbModel.replies ?? 0,
    quotedPostIds: dbModel.quotedPostIds ?? [],
    quotedPosts: dbModel.quotedPosts?.map((qp) => ({
      id: qp.id,
      title: qp.title,
      content: qp.content,
      author: {
        id: qp.authorId,
        name: qp.author?.name ?? null,
        email: qp.author?.email ?? null,
        image: qp.author?.image ?? null,
      },
      createdAt: qp.createdAt,
    })),
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
            paper.paperAuthors?.map((pa) => ({
              id: pa.author.id,
              name: pa.author.name,
              institution: pa.author.institution,
            })) || [],
          institutions:
            paper.paperInstitutions?.map((pi) => pi.institution.name) || [],
          tags: paper.tags,
          link: paper.link,
          source: paper.source,
          externalId: paper.externalId,
          stars: paper.userPaperInteractions?.length ?? 0,
          starred: userPaperInteractionsMap.get(paper.id)?.starred ?? false,
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
          tags: true,
          link: true,
          source: true,
          externalId: true,
          createdAt: true,
          updatedAt: true,
          llmAbstract: true,
          llmAbstractGeneratedAt: true,
          paperInstitutions: {
            select: {
              institution: {
                select: {
                  id: true,
                  name: true,
                },
              },
            },
          },
          paperAuthors: {
            include: {
              author: {
                select: {
                  id: true,
                  name: true,
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

  // Separate query to load quoted posts for posts that have quotedPostIds
  const allQuotedPostIds = [
    ...new Set(allRows.flatMap((row) => row.quotedPostIds || [])),
  ];
  const quotedPostsMap = new Map();

  if (allQuotedPostIds.length > 0) {
    const quotedPosts = await prisma.post.findMany({
      where: {
        id: {
          in: allQuotedPostIds,
        },
      },
      select: {
        id: true,
        title: true,
        content: true,
        authorId: true,
        createdAt: true,
        author: {
          select: {
            id: true,
            name: true,
            email: true,
            image: true,
          },
        },
      },
    });

    quotedPosts.forEach((qp) => quotedPostsMap.set(qp.id, qp));
  }

  // Add quoted posts to each row
  const rowsWithQuotedPosts = allRows.map((row) => ({
    ...row,
    quotedPosts: (row.quotedPostIds || [])
      .map((id) => quotedPostsMap.get(id))
      .filter(Boolean),
  }));

  // Calculate lastActivityAt for each post and sort by it
  const postsWithActivity = rowsWithQuotedPosts
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
    // Fetch user paper interactions (for starred status)
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

    // Cast row to PrismaPost (Prisma query result has different shape)
    return toFeedPostDTO(
      row as unknown as PrismaPost,
      usersMap,
      papersMap,
      userPaperInteractionsMap
    );
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
