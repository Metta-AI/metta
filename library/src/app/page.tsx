import { loadFeedPosts } from "@/posts/data/feed";
import { loadPapersWithUserContext } from "@/posts/data/papers";

import { FeedPostsPage } from "./FeedPostsPage";

export default async function FrontPage() {
  const posts = await loadFeedPosts();
  const papersData = await loadPapersWithUserContext();

  return <FeedPostsPage posts={posts} papersData={papersData} />;
}
