"use client";

import { FC, useEffect, useRef, useState } from "react";
import { MessageCircle } from "lucide-react";

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
  const { openPaper } = useOverlayNavigation();

  // Star mutation
  const starMutation = useStarMutation();

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
        handleToggleStar
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
  const handleToggleStar = (paperId: string) => {
    starMutation.mutate({ paperId });
  };

  // Handle post deletion - optimistically remove from list
  const handlePostDeleted = (postId: string) => {
    page.remove((post) => post.id === postId);
  };

  const feedScrollRef = useRef<HTMLDivElement>(null);
  const loadMoreRef = useRef<HTMLDivElement>(null);

  // Handle new post creation - prepend to feed
  const handlePostCreated = (newPost: FeedPostDTO) => {
    page.prepend(newPost);
  };

  // Infinite scroll with IntersectionObserver
  useEffect(() => {
    if (!page.loadNext || page.loading) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && page.loadNext) {
          page.loadNext(10);
        }
      },
      { threshold: 0.1 }
    );

    if (loadMoreRef.current) {
      observer.observe(loadMoreRef.current);
    }

    return () => observer.disconnect();
  }, [page.loadNext, page.loading]);

  // MathJax rendering
  useEffect(() => {
    if (mathJaxLoaded && feedScrollRef.current) {
      const renderMathContent = async () => {
        try {
          await renderMath(feedScrollRef.current!);
        } catch (error) {
          console.error("MathJax rendering failed for feed:", error);
        }
      };
      const timeoutId = setTimeout(renderMathContent, 200);
      return () => {
        clearTimeout(timeoutId);
      };
    }
  }, [mathJaxLoaded, page.items, renderMath]);

  return (
    <div className="flex h-auto w-full flex-col md:flex-row">
      {/* Main feed area */}
      <div ref={feedScrollRef} className="h-full flex-1 overflow-y-auto">
        {/* Post Composition */}
        <NewPostForm onPostCreated={handlePostCreated} />
        {/* Feed */}
        <div className="mx-4 mt-6 max-w-2xl md:mr-4 md:ml-6">
          {page.items.length > 0 ? (
            <div className="space-y-3">
              {page.items.map((post) => (
                <FeedPost
                  key={post.id}
                  post={post}
                  onPaperClick={handlePaperClick}
                  onUserClick={handleUserClick}
                  currentUser={currentUser}
                  isCommentsExpanded={false}
                  onCommentToggle={() => {}}
                  highlightedCommentId={null}
                  onPostDeleted={handlePostDeleted}
                />
              ))}

              {/* Load more trigger */}
              {page.loadNext && (
                <div ref={loadMoreRef} className="py-6">
                  {page.loading && (
                    <div className="flex items-center justify-center gap-2 text-gray-500">
                      <div className="h-4 w-4 animate-spin rounded-full border-2 border-gray-300 border-t-blue-600" />
                      <span>Loading more posts...</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : (
            <div className="rounded-lg border border-gray-200 bg-white p-8 text-center">
              <div className="mb-4 text-gray-400">
                <MessageCircle className="mx-auto h-12 w-12" />
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
    </div>
  );
};
