import { auth } from "@/lib/auth";
import { loadFeedPosts } from "@/posts/data/feed";

export default async function Home() {
  const session = await auth();
  console.log({ session });

  const posts = await loadFeedPosts();

  return (
    <div>
      <div>
        {posts.items.map((post) => (
          <div key={post.id}>{post.title}</div>
        ))}
      </div>
    </div>
  );
}
