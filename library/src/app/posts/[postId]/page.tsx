import { auth } from "@/lib/auth";
import loadPost from "@/posts/data/post";
import { PostDiscussion } from "@/components/PostDiscussion";
import { PaperSidebar } from "@/components/PaperSidebar";
import {
  OverlayStackProvider,
  OverlayStackRenderer,
} from "@/components/OverlayStack";

export default async function PostPage({
  params,
}: {
  params: Promise<{ postId: string }>;
}) {
  const postId = (await params).postId;
  const post = await loadPost(postId);
  const session = await auth();

  const currentUser = session?.user
    ? {
        id: session.user.id!,
        name: session.user.name,
        email: session.user.email,
      }
    : null;

  return (
    <OverlayStackProvider>
      <div className="flex h-screen">
        {/* Main content area - Post Discussion (Twitter-style narrow) */}
        <div className="w-[600px] overflow-y-auto p-6">
          <PostDiscussion post={post} currentUser={currentUser} />
        </div>

        {/* Right sidebar - Paper Overview (takes remaining space) */}
        <div className="flex-1">
          <PaperSidebar paper={post.paper} />
        </div>
      </div>
      <OverlayStackRenderer />
    </OverlayStackProvider>
  );
}
