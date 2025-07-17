import { loadFeedPosts } from "@/posts/data/feed";

import { FeedPostsPage } from "./FeedPostsPage";

export default async function FrontPage() {
  const posts = await loadFeedPosts();

  return <FeedPostsPage posts={posts} />;
}
