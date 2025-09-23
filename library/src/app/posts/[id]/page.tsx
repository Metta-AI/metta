import { notFound } from "next/navigation";

import { loadFeedPosts } from "@/posts/data/feed";
import { loadPapersWithUserContext } from "@/posts/data/papers";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";
import {
  OverlayStackProvider,
  OverlayStackRenderer,
} from "@/components/OverlayStack";

import { PostPage } from "./PostPage";

interface Props {
  params: Promise<{
    id: string;
  }>;
}

export default async function SinglePostPage({ params }: Props) {
  const { id } = await params;
  
  // Verify the post exists first
  const post = await prisma.post.findUnique({
    where: { id },
    select: { id: true, title: true },
  });

  if (!post) {
    notFound();
  }

  // Load the same data as the main feed
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
    <OverlayStackProvider>
      <PostPage
        postId={id}
        posts={posts}
        papersData={papersData}
        currentUser={currentUser}
      />
      <OverlayStackRenderer />
    </OverlayStackProvider>
  );
}

// Generate metadata for the post page
export async function generateMetadata({ params }: Props) {
  const { id } = await params;
  const post = await prisma.post.findUnique({
    where: { id },
    select: { 
      title: true, 
      content: true,
      author: { select: { name: true, email: true } }
    },
  });

  if (!post) {
    return {
      title: "Post Not Found",
    };
  }

  const authorName = post.author.name || post.author.email?.split("@")[0] || "Unknown";
  const preview = post.content 
    ? post.content.slice(0, 150) + (post.content.length > 150 ? "..." : "")
    : post.title;

  return {
    title: `${post.title} - ${authorName} | Softmax Library`,
    description: preview,
  };
}
