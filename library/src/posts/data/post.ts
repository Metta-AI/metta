import { eq } from "drizzle-orm";

import { db } from "@/lib/db";
import { postsTable } from "@/lib/db/schema/post";
import { usersTable } from "@/lib/db/schema/auth";
import { papersTable } from "@/lib/db/schema/paper";

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

  // Fetch the paper information if paperId exists
  let paper = null;
  if (row[0].paperId) {
    const paperResult = await db
      .select()
      .from(papersTable)
      .where(eq(papersTable.id, row[0].paperId))
      .limit(1);
    paper = paperResult[0] || null;
  }

  // Create maps for the lookup
  const usersMap = new Map();
  if (author[0]) {
    usersMap.set(author[0].id, author[0]);
  }

  const papersMap = new Map();
  if (paper) {
    papersMap.set(paper.id, paper);
  }

  const post = toFeedPostDTO(row[0], usersMap, papersMap);
  return post;
}
