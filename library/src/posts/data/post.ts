import { eq } from "drizzle-orm";

import { db } from "@/lib/db";
import { postsTable } from "@/lib/db/schema/post";

import { FeedPostDTO, toFeedPostDTO } from "./feed";

export type PostDTO = FeedPostDTO;

export default async function loadPost(postId: string): Promise<PostDTO> {
  const row = await db.query.postsTable.findFirst({
    where: eq(postsTable.id, postId),
  });
  if (!row) {
    throw new Error("Post not found");
  }

  // If you need to have a different DTO for the post page, create a new toPostDTO function.
  const post = toFeedPostDTO(row);
  return post;
}
