"use client";

import { FC, useEffect, useRef, useState } from "react";

import { InfiniteScroll } from "@/components/InfiniteScroll";
import { useMathJax } from "@/components/MathJaxProvider";
import { usePaginator } from "@/lib/hooks/usePaginator";
import { Paginated } from "@/lib/paginated";
import { FeedPostDTO } from "@/posts/data/feed";
import {
  PaperWithUserContext,
  User,
  UserInteraction,
} from "@/posts/data/papers";
import { useOverlayNavigation } from "@/components/OverlayStack";
import { toggleStarAction } from "@/posts/actions/toggleStarAction";
import { toggleQueueAction } from "@/posts/actions/toggleQueueAction";
import UserCard from "@/components/UserCard";

import { FeedPost } from "./FeedPost";
import { NewPostForm } from "./NewPostForm";

/**
 * FeedPostsPage Component
 *
 * Displays the main social feed with:
 * - Post composition form at the top
 * - Chronological list of posts
 * - Infinite scrolling for seamless content loading
 * - Rich post display with author info and social metrics
 * - MathJax rendering for mathematical content
 * - Paper overlay for viewing paper details
 */
export const FeedPostsPage: FC<{
  posts: Paginated<FeedPostDTO>;
  papersData: {
    papers: PaperWithUserContext[];
    users: User[];
    interactions: UserInteraction[];
  };
  currentUser: {
    id: string;
    name?: string | null;
    email?: string | null;
  } | null;
}> = ({ posts: initialPosts, papersData, currentUser }) => {
  const page = usePaginator(initialPosts);
  const { mathJaxLoaded, renderMath } = useMathJax();
  const feedRef = useRef<HTMLDivElement>(null);
  const { openPaper } = useOverlayNavigation();

  // User card state
  const [selectedUser, setSelectedUser] = useState<User | null>(null);

  // Handle paper click using overlay navigation
  const handlePaperClick = (paperId: string) => {
    const paper = papersData.papers.find((p) => p.id === paperId);
    if (paper) {
      openPaper(
        paper,
        papersData.users,
        papersData.interactions,
        handleToggleStar,
        handleToggleQueue
      );
    }
  };

  // Handle user click
  const handleUserClick = (userId: string) => {
    const user = papersData.users.find((u) => u.id === userId);
    if (user) {
      setSelectedUser(user);
    }
  };

  // Handle user card close
  const handleUserCardClose = () => {
    setSelectedUser(null);
  };

  // Handle toggle star
  const handleToggleStar = async (paperId: string) => {
    try {
      const formData = new FormData();
      formData.append("paperId", paperId);
      await toggleStarAction(formData);

      // The overlay stack handles its own state updates
    } catch (error) {
      console.error("Error toggling star:", error);
    }
  };

  // Handle toggle queue
  const handleToggleQueue = async (paperId: string) => {
    try {
      const formData = new FormData();
      formData.append("paperId", paperId);
      await toggleQueueAction(formData);

      // The overlay stack handles its own state updates
    } catch (error) {
      console.error("Error toggling queue:", error);
    }
  };

  // MathJax rendering effect - similar to mockup approach
  useEffect(() => {
    if (mathJaxLoaded && feedRef.current) {
      const renderMathContent = async () => {
        try {
          await renderMath(feedRef.current!);
        } catch (error) {
          console.error("MathJax rendering failed for feed:", error);
        }
      };

      // Try immediately and after delays to ensure DOM is ready
      renderMathContent();
      setTimeout(renderMathContent, 100);
      setTimeout(renderMathContent, 500);
      setTimeout(renderMathContent, 1000);
    }
  }, [mathJaxLoaded, page.items, renderMath]); // Re-render when posts change

  return (
    <>
      {" "}
      {/* Simplified root div */}
      {/* Post Composition */}
      <NewPostForm />
      {/* Feed with Infinite Scroll */}
      <div ref={feedRef} className="mx-auto max-w-2xl">
        {page.items.length > 0 ? (
          <InfiniteScroll
            loadNext={page.loadNext!}
            hasMore={!!page.loadNext}
            loading={page.loading}
          >
            {page.items.map((post) => (
              <FeedPost
                key={post.id}
                post={post}
                onPaperClick={handlePaperClick}
                onUserClick={handleUserClick}
                currentUser={currentUser}
              />
            ))}
          </InfiniteScroll>
        ) : (
          <div className="rounded-lg border border-gray-200 bg-white p-8 text-center">
            <div className="mb-4 text-gray-400">
              <svg
                className="mx-auto h-12 w-12"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                />
              </svg>
            </div>
            <h3 className="mb-2 text-lg font-medium text-gray-900">
              No posts yet
            </h3>
            <p className="text-gray-500">
              Be the first to share something interesting!
            </p>
          </div>
        )}
      </div>
      {/* Paper overlays are now handled by the OverlayStack */}
      {/* User Card */}
      {selectedUser && (
        <UserCard
          user={selectedUser}
          allPapers={papersData.papers}
          users={papersData.users}
          interactions={papersData.interactions}
          onClose={handleUserCardClose}
        />
      )}
    </>
  );
};
