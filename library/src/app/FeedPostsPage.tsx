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
import { useStarMutation } from "@/hooks/useStarMutation";
import { toggleQueueAction } from "@/posts/actions/toggleQueueAction";
import UserCard from "@/components/UserCard";
import { PaperSidebar } from "@/components/PaperSidebar";

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

  // Star mutation
  const starMutation = useStarMutation();

  // User card state
  const [selectedUser, setSelectedUser] = useState<User | null>(null);

  // Global comment expansion state - only one post can have expanded comments at a time
  const [expandedPostId, setExpandedPostId] = useState<string | null>(null);

  // Selected post for paper sidebar
  const [selectedPostForPaper, setSelectedPostForPaper] =
    useState<FeedPostDTO | null>(null);

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

  // Handle comment toggle - only one post can have expanded comments
  const handleCommentToggle = (postId: string) => {
    setExpandedPostId((current) => (current === postId ? null : postId));
  };

  // Handle post selection for paper sidebar
  const handlePostSelect = (post: FeedPostDTO) => {
    setSelectedPostForPaper(post);
    // Also expand comments for the selected post
    setExpandedPostId(post.id);
  };

  // Handle paper sidebar close
  const handlePaperSidebarClose = () => {
    setSelectedPostForPaper(null);
    // Also close expanded comments when closing paper sidebar
    setExpandedPostId(null);
  };

  // Handle toggle star
  const handleToggleStar = (paperId: string) => {
    starMutation.mutate(paperId);
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

  // MathJax rendering effect - single debounced call
  useEffect(() => {
    if (mathJaxLoaded && feedRef.current) {
      const renderMathContent = async () => {
        try {
          await renderMath(feedRef.current!);
        } catch (error) {
          console.error("MathJax rendering failed for feed:", error);
        }
      };

      // Single delayed call to avoid race conditions
      const timeoutId = setTimeout(renderMathContent, 200);

      return () => {
        clearTimeout(timeoutId);
      };
    }
  }, [mathJaxLoaded, page.items, renderMath]); // Re-render when posts change

  return (
    <div className="flex h-screen">
      {/* Main feed area */}
      <div className="flex-1 overflow-y-auto">
        {/* Post Composition */}
        <NewPostForm />
        {/* Feed with Infinite Scroll */}
        <div ref={feedRef} className="mt-6 ml-6 max-w-2xl">
          {page.items.length > 0 ? (
            <InfiniteScroll
              loadNext={page.loadNext!}
              hasMore={!!page.loadNext}
              loading={page.loading}
            >
              <div className="flex flex-col gap-4">
                {page.items.map((post) => (
                  <FeedPost
                    key={post.id}
                    post={post}
                    onPaperClick={handlePaperClick}
                    onUserClick={handleUserClick}
                    currentUser={currentUser}
                    isCommentsExpanded={expandedPostId === post.id}
                    onCommentToggle={() => handleCommentToggle(post.id)}
                    onPostSelect={() => handlePostSelect(post)}
                    isSelected={selectedPostForPaper?.id === post.id}
                  />
                ))}
              </div>
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
      </div>

      {/* Right sidebar - Paper Overview */}
      {selectedPostForPaper?.paper && (
        <div className="flex-1">
          <PaperSidebar
            paper={selectedPostForPaper.paper}
            onClose={handlePaperSidebarClose}
          />
        </div>
      )}
    </div>
  );
};
