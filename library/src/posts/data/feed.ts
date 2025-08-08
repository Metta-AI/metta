import { prisma } from "@/lib/db/prisma";
import { makePaginated, Paginated } from "@/lib/paginated";
import { auth } from "@/lib/auth";

export type FeedPostDTO = {
  id: string;
  title: string;
  content: string | null;
  postType: 'user-post' | 'paper-post' | 'pure-paper';
  likes: number;
  liked: boolean;
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
    authors?: { id: string; name: string; orcid?: string | null; institution?: string | null }[];
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

export function toFeedPostDTO(dbModel: any, usersMap: Map<string, any>, papersMap: Map<string, any>, userLikesMap: Map<string, boolean> = new Map()): FeedPostDTO {
  const author = usersMap.get(dbModel.authorId);
  const paper = dbModel.paperId ? papersMap.get(dbModel.paperId) : null;
  
  return {
    id: dbModel.id,
    title: dbModel.title,
    content: dbModel.content,
    postType: dbModel.postType as 'user-post' | 'paper-post' | 'pure-paper',
    likes: dbModel.likes ?? 0,
    liked: userLikesMap.get(dbModel.id) ?? false,
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
      authors: paper.paperAuthors?.map((pa: any) => ({
        id: pa.author.id,
        name: pa.author.name,
        orcid: pa.author.orcid,
        institution: pa.author.institution
      })) || [],
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

export async function loadFeedPosts({
  limit = 10,
  cursor,
}: {
  limit?: number;
  cursor?: Date;
} = {}): Promise<Paginated<FeedPostDTO>> {
  // Get current user session
  const session = await auth();
  
  // Build the query with proper cursor-based pagination
  const rows = await prisma.post.findMany({
    where: cursor ? {
      createdAt: {
        lt: cursor,
      },
    } : undefined,
    take: limit + 1,
    orderBy: [
      {
        createdAt: 'desc',
      },
      {
        id: 'desc', // Secondary sort by ID to ensure consistent ordering
      },
    ],
    include: {
      author: true,
      paper: {
        include: {
          paperAuthors: {
            include: {
              author: {
                select: {
                  id: true,
                  name: true,
                  orcid: true,
                  institution: true
                }
              }
            }
          }
        }
      },
    },
  });

  // Fetch user likes for these posts if user is authenticated
  let userLikesMap = new Map<string, boolean>();
  if (session?.user?.id) {
    const postIds = rows.map(row => row.id);
    const userLikes = await prisma.userPostLike.findMany({
      where: {
        userId: session.user.id,
        postId: {
          in: postIds,
        },
      },
    });
    
    userLikesMap = new Map(userLikes.map(like => [like.postId, true]));
  }

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

    return toFeedPostDTO(row, usersMap, papersMap, userLikesMap);
  });

  // Check if there are more posts to load
  const hasMore = posts.length > limit;
  const nextCursor = hasMore ? posts[limit - 1]?.createdAt : undefined;

  async function loadMore(limit: number) {
    "use server";
    return loadFeedPosts({ cursor: nextCursor, limit });
  }

  return makePaginated(posts, limit, loadMore);
} 