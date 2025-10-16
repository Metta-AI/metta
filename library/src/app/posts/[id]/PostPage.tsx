"use client";

import { FC, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { MessageCircle, AlertTriangle } from "lucide-react";

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
import { PaperSidebar } from "@/components/PaperSidebar";

import { FeedPost } from "../../FeedPost";
import { NewPostForm } from "../../NewPostForm";

/**
 * PostPage Component
 *
 * Shows the same feed as the main page, but with a specific post highlighted
 * and expanded to show comments. This provides permalinking for posts while
 * maintaining the familiar feed UI.
 */
export const PostPage: FC<{
  postId: string;
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
}> = ({ postId, posts: initialPosts, papersData, currentUser }) => {
  const router = useRouter();
  const page = usePaginator(initialPosts);
  const { mathJaxLoaded, renderMath } = useMathJax();
  const feedRef = useRef<HTMLDivElement>(null);
  const targetPostRef = useRef<HTMLDivElement>(null);
  const { openPaper } = useOverlayNavigation();

  // Star mutation
  const starMutation = useStarMutation();

  // User card state
  const [selectedUser, setSelectedUser] = useState<User | null>(null);

  // Auto-expand the target post and allow expanding others
  const [expandedPostId, setExpandedPostId] = useState<string | null>(postId);

  // Selected post for paper sidebar - will be set automatically for target post
  const [selectedPostForPaper, setSelectedPostForPaper] =
    useState<FeedPostDTO | null>(null);

  // Track if we've scrolled to the target post yet
  const [hasScrolledToPost, setHasScrolledToPost] = useState(false);

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

  // Handle user card close - navigate back to feed
  const handleUserCardClose = () => {
    router.push("/");
  };

  // Handle comment toggle - allow expanding/collapsing any post
  const handleCommentToggle = (clickedPostId: string) => {
    setExpandedPostId((current) =>
      current === clickedPostId ? null : clickedPostId
    );
  };

  // Handle post selection for paper sidebar
  const handlePostSelect = (post: FeedPostDTO) => {
    setSelectedPostForPaper(post);
    // Also expand comments for the selected post
    setExpandedPostId(post.id);
  };

  // Handle paper sidebar close - navigate back to feed
  const handlePaperSidebarClose = () => {
    router.push("/");
  };

  // Handle toggle star
  const handleToggleStar = (paperId: string) => {
    starMutation.mutate({ paperId });
  };

  // Handle post deletion - optimistically remove from list
  const handlePostDeleted = (deletedPostId: string) => {
    page.remove((post) => post.id === deletedPostId);
    // If we're deleting the target post, navigate home
    if (deletedPostId === postId) {
      router.push("/");
    }
  };

  // Scroll to the target post when the page loads
  useEffect(() => {
    if (!hasScrolledToPost && targetPostRef.current) {
      const timer = setTimeout(() => {
        targetPostRef.current?.scrollIntoView({
          behavior: "smooth",
          block: "center",
        });
        setHasScrolledToPost(true);
      }, 500); // Give time for the page to render

      return () => clearTimeout(timer);
    }
  }, [hasScrolledToPost, page.items]);

  // MathJax rendering effect
  useEffect(() => {
    if (mathJaxLoaded && feedRef.current) {
      const renderMathContent = async () => {
        try {
          await renderMath(feedRef.current!);
        } catch (error) {
          console.error("MathJax rendering failed for feed:", error);
        }
      };

      const timeoutId = setTimeout(renderMathContent, 200);
      return () => clearTimeout(timeoutId);
    }
  }, [mathJaxLoaded, page.items, renderMath]);

  // Check if the target post exists in the current page
  const targetPost = page.items.find((post) => post.id === postId);
  const targetPostExists = !!targetPost;

  // Auto-select target post for paper sidebar if it has a paper
  useEffect(() => {
    if (targetPost && targetPost.paper && !selectedPostForPaper) {
      setSelectedPostForPaper(targetPost);
    }
  }, [targetPost, selectedPostForPaper]);

  // Extract highlighted comment ID from URL hash
  const [highlightedCommentId, setHighlightedCommentId] = useState<
    string | null
  >(null);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const hash = window.location.hash;
      if (hash.startsWith("#comment-")) {
        const commentId = hash.replace("#comment-", "");
        setHighlightedCommentId(commentId);

        // Scroll to the comment after a short delay to let the page render
        setTimeout(() => {
          const commentElement = document.getElementById(
            `comment-${commentId}`
          );
          if (commentElement) {
            commentElement.scrollIntoView({
              behavior: "smooth",
              block: "center",
            });
          }
        }, 1000);
      }
    }
  }, [postId]);

  const feedScrollRef = useRef<HTMLDivElement>(null);
  const loadMoreRef = useRef<HTMLDivElement>(null);

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

  return (
    <div className="flex h-[calc(97vh-53px)] w-full flex-col md:flex-row">
      {/* Mobile: Paper details on top (when selected) */}
      {selectedPostForPaper?.paper && (
        <div className="h-1/2 border-b border-gray-300 pt-14 md:hidden">
          <PaperSidebar
            paper={selectedPostForPaper.paper}
            onClose={handlePaperSidebarClose}
          />
        </div>
      )}

      {/* Main feed area */}
      <div
        ref={feedScrollRef}
        className={`flex-1 overflow-y-auto ${
          selectedPostForPaper?.paper ? "md:w-[45%] md:flex-none" : ""
        }`}
      >
        {/* Post Composition */}
        <NewPostForm />

        {/* Feed with Infinite Scroll */}
        <div className="mx-4 mt-6 max-w-2xl md:mr-4 md:ml-6">
          {page.items.length > 0 ? (
            <div className="space-y-3">
              {page.items.map((post) => {
                const isTargetPost = post.id === postId;

                return (
                  <div
                    key={post.id}
                    ref={(node) => {
                      if (isTargetPost) {
                        targetPostRef.current = node as HTMLDivElement | null;
                      }
                    }}
                  >
                    <FeedPost
                      post={post}
                      onPaperClick={handlePaperClick}
                      onUserClick={handleUserClick}
                      currentUser={currentUser}
                      isCommentsExpanded={expandedPostId === post.id}
                      onCommentToggle={() => handleCommentToggle(post.id)}
                      highlightedCommentId={
                        isTargetPost ? highlightedCommentId : null
                      }
                      onPostDeleted={handlePostDeleted}
                    />
                  </div>
                );
              })}

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
                Be the first to start a discussion about a paper!
              </p>
            </div>
          )}

          {/* Warning if target post not found */}
          {!targetPostExists && page.items.length > 0 && (
            <div className="mt-4 rounded-lg border border-yellow-200 bg-yellow-50 p-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <AlertTriangle className="h-5 w-5 text-yellow-400" />
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-yellow-800">
                    Post not found in current feed
                  </h3>
                  <div className="mt-2 text-sm text-yellow-700">
                    <p>
                      The post you're looking for might be older or not visible
                      in the current feed view.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* User Card Modal */}
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

      {/* Desktop: Paper sidebar on right */}
      {selectedPostForPaper?.paper && (
        <div className="hidden w-full md:flex md:h-screen md:flex-shrink-0">
          <PaperSidebar
            paper={selectedPostForPaper.paper}
            onClose={handlePaperSidebarClose}
          />
        </div>
      )}
    </div>
  );
};
