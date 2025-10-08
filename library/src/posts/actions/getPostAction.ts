"use server";

import loadPost, { type PostDTO } from "../data/post";

/**
 * Server action to fetch a single post
 * Wraps the loadPost data function for use in client components via React Query
 */
export async function getPostAction(postId: string): Promise<PostDTO> {
  return loadPost(postId);
}
