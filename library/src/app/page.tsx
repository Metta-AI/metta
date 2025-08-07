import { loadFeedPosts } from "@/posts/data/feed";
import { loadPapersWithUserContext } from "@/posts/data/papers";
import { auth } from "@/lib/auth";

import { FeedPostsPage } from "./FeedPostsPage";

export default async function FrontPage() {
  const posts = await loadFeedPosts();
  const papersData = await loadPapersWithUserContext();
  const session = await auth();

  // Only pass user if they have a valid ID
  const currentUser = session?.user?.id
    ? {
        id: session.user.id,
        name: session.user.name,
        email: session.user.email,
      }
    : null;

  return (
    <FeedPostsPage
      posts={posts}
      papersData={papersData}
      currentUser={currentUser}
    />
  );
}
