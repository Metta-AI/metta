import loadPost from "@/posts/data/post";

export default async function PostPage({
  params,
}: {
  params: Promise<{ postId: string }>;
}) {
  const postId = (await params).postId;
  const post = await loadPost(postId);

  return <div>POST: {post.title}</div>;
}
