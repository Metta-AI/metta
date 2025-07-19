import { prisma } from "@/lib/db/prisma";
import { FeedPostDTO, toFeedPostDTO } from "./feed";

export type PostDTO = FeedPostDTO;

export default async function loadPost(postId: string): Promise<PostDTO> {
  const post = await prisma.post.findUnique({
    where: { id: postId },
    include: {
      author: true,
      paper: true,
    },
  });

  if (!post) {
    throw new Error("Post not found");
  }

  // Create maps for the lookup (maintaining compatibility with existing toFeedPostDTO function)
  const usersMap = new Map();
  if (post.author) {
    usersMap.set(post.author.id, post.author);
  }

  const papersMap = new Map();
  if (post.paper) {
    papersMap.set(post.paper.id, post.paper);
  }

  const postDTO = toFeedPostDTO(post, usersMap, papersMap);
  return postDTO;
} 