import { prisma } from "@/lib/db/prisma";
import { auth } from "@/lib/auth";
import { FeedPostDTO, toFeedPostDTO } from "./feed";

export type PostDTO = FeedPostDTO;

export default async function loadPost(postId: string): Promise<PostDTO> {
  const post = await prisma.post.findUnique({
    where: { id: postId },
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

  if (!post) {
    throw new Error("Post not found");
  }

  // Get current user session to fetch their interactions
  const session = await auth();

  // Create maps for the lookup (maintaining compatibility with existing toFeedPostDTO function)
  const usersMap = new Map();
  if (post.author) {
    usersMap.set(post.author.id, post.author);
  }

  const papersMap = new Map();
  if (post.paper) {
    papersMap.set(post.paper.id, post.paper);
  }

  // Fetch user interactions for this paper if user is authenticated
  let userPaperInteractionsMap = new Map<string, any>();
  if (session?.user?.id && post.paper) {
    const userPaperInteraction = await prisma.userPaperInteraction.findUnique({
      where: {
        userId_paperId: {
          userId: session.user.id,
          paperId: post.paper.id,
        },
      },
    });

    if (userPaperInteraction) {
      userPaperInteractionsMap.set(post.paper.id, userPaperInteraction);
    }
  }

  const postDTO = toFeedPostDTO(
    post as any,
    usersMap,
    papersMap,
    userPaperInteractionsMap
  );
  return postDTO;
}
