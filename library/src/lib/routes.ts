/*
 * This file contains functions that generate URLs for the app.
 *
 * For the ease of refactoring, you should use these functions instead of
 * hardcoding URLs in `<Link>` components and other routing logic.
 */

export function postRoute(postId: string) {
  return `/posts/${postId}`;
}
