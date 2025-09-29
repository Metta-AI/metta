/**
 * Utility functions for parsing post links and extracting quoted posts
 */

/**
 * Extracts post IDs from post content by looking for post URLs
 * @param content The content to parse
 * @returns Array of post IDs (max 2)
 */
export function extractPostIdsFromContent(content: string): string[] {
  if (!content) return [];

  // Pattern to match post URLs like /posts/[id] or domain.com/posts/[id]
  const postLinkPattern =
    /(?:https?:\/\/[^\/\s]+)?\/posts\/([a-zA-Z0-9_-]+)(?:[^\w\-]|$)/g;

  const postIds: string[] = [];
  let match;

  while (
    (match = postLinkPattern.exec(content)) !== null &&
    postIds.length < 2
  ) {
    const postId = match[1];
    // Avoid duplicates
    if (!postIds.includes(postId)) {
      postIds.push(postId);
    }
  }

  return postIds;
}

/**
 * Validates that post IDs exist in the database
 * @param postIds Array of post IDs to validate
 * @returns Array of valid post IDs
 */
export async function validatePostIds(postIds: string[]): Promise<string[]> {
  if (postIds.length === 0) return [];

  const { prisma } = await import("@/lib/db/prisma");

  const existingPosts = await prisma.post.findMany({
    where: {
      id: {
        in: postIds,
      },
    },
    select: {
      id: true,
    },
  });

  return existingPosts.map((post) => post.id);
}

/**
 * Determines if a post should be treated as a quote post
 * @param content Post content
 * @param quotedPostIds Array of quoted post IDs (already validated)
 * @returns true if post should be a quote post
 */
export function shouldBeQuotePost(
  content: string,
  quotedPostIds: string[]
): boolean {
  return quotedPostIds.length > 0;
}
