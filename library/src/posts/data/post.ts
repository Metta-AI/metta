import { eq } from "drizzle-orm";

import { db } from "@/lib/db";
import { postsTable } from "@/lib/db/schema/post";
import { usersTable } from "@/lib/db/schema/auth";

import { FeedPostDTO, toFeedPostDTO } from "./feed";

export type PostDTO = FeedPostDTO;

export default async function loadPost(postId: string): Promise<PostDTO> {
  const row = await db
    .select()
    .from(postsTable)
    .where(eq(postsTable.id, postId))
    .limit(1);

  if (!row[0]) {
    throw new Error("Post not found");
  }

  // Fetch the author information
  const author = await db
    .select()
    .from(usersTable)
    .where(eq(usersTable.id, row[0].authorId))
    .limit(1);

  // Create a map for the user lookup
  const usersMap = new Map();
  if (author[0]) {
    usersMap.set(author[0].id, author[0]);
  }

  const post = toFeedPostDTO(row[0], usersMap);
  return post;
}
