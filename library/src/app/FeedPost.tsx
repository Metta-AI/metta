"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { FC, useState, useEffect } from "react";
import { useAction } from "next-safe-action/hooks";

import { FeedPostDTO } from "@/posts/data/feed";
import { PaperCard } from "@/components/PaperCard";
import { DeleteConfirmationModal } from "@/components/DeleteConfirmationModal";
import { linkifyText } from "@/lib/utils/linkify";
import { deletePostAction } from "@/posts/actions/deletePostAction";
import { toggleQueueAction } from "@/posts/actions/toggleQueueAction";
import { SilentArxivRefresh } from "@/components/SilentArxivRefresh";
import { ChevronRight } from "lucide-react";

/**
 * FeedPost Component
 *
 * Displays a single post in the social feed with rich formatting including:
 * - Author information with avatar
 * - Post content with LaTeX support (rendered by parent component)
 * - Social metrics (likes, queues, replies)
 * - Paper references when applicable using PaperCard
 * - Interactive elements
 */
export const FeedPost: FC<{
  post: FeedPostDTO;
  onPaperClick?: (paperId: string) => void;
  onUserClick?: (userId: string) => void;
  currentUser: {
    id: string;
    name?: string | null;
    email?: string | null;
  } | null;
}> = ({ post, onPaperClick, onUserClick, currentUser }) => {
  const router = useRouter();

  // Local state for paper data that can be updated when institutions are added
  const [paperData, setPaperData] = useState(post.paper);

  // Local state for optimistic queue updates
  const [optimisticQueues, setOptimisticQueues] = useState(post.queues);
  const [optimisticQueued, setOptimisticQueued] = useState(
    post.paper?.queued ?? false
  );

  // Delete modal state
  const [showDeleteModal, setShowDeleteModal] = useState(false);

  // Sync local state when post prop changes
  useEffect(() => {
    setPaperData(post.paper);
    setOptimisticQueues(post.queues);
    setOptimisticQueued(post.paper?.queued ?? false);
  }, [post.paper, post.queues]);

  // Callback to update paper data when institutions are added
  const handleInstitutionsAdded = (institutions: string[]) => {
    if (paperData) {
      setPaperData({
        ...paperData,
        institutions: institutions,
      });
    }
  };

  // Generate user initials for avatar
  const getUserInitials = (name: string | null, email: string | null) => {
    if (name) {
      return name
        .split(" ")
        .map((n) => n[0])
        .join("")
        .toUpperCase()
        .slice(0, 2);
    }
    if (email) {
      return email.charAt(0).toUpperCase();
    }
    return "?";
  };

  // Format relative time
  const formatRelativeTime = (date: Date) => {
    const now = new Date();
    const diffInHours = Math.floor(
      (now.getTime() - date.getTime()) / (1000 * 60 * 60)
    );

    if (diffInHours < 1) return "now";
    if (diffInHours < 24) return `${diffInHours}h`;

    const diffInDays = Math.floor(diffInHours / 24);
    if (diffInDays < 7) return `${diffInDays}d`;

    const diffInWeeks = Math.floor(diffInDays / 7);
    if (diffInWeeks < 4) return `${diffInWeeks}w`;

    const diffInMonths = Math.floor(diffInDays / 30);
    return `${diffInMonths}m`;
  };

  // Remove arxiv URLs and trim trailing whitespace
  const cleanPostContent = (content: string) => {
    // Remove arxiv URLs (various formats)
    const arxivPattern =
      /https?:\/\/(www\.)?(arxiv\.org\/abs\/|arxiv\.org\/pdf\/)[^\s]+/gi;
    return content.replace(arxivPattern, "").replace(/\s+$/, "");
  };

  // Delete post action
  const { execute: executeDelete, isExecuting: isDeleting } = useAction(
    deletePostAction,
    {
      onSuccess: () => {
        window.location.reload(); // Refresh to show updated feed
      },
      onError: (error) => {
        console.error("Error deleting post:", error);
      },
    }
  );

  // Queue paper action
  const { execute: executeQueue, isExecuting: isQueueing } = useAction(
    toggleQueueAction,
    {
      onSuccess: () => {
        // Success is handled by optimistic updates
      },
      onError: (error) => {
        // Revert optimistic updates on error
        setOptimisticQueued(!optimisticQueued);
        setOptimisticQueues(post.queues);
        console.error("Error toggling queue:", error);
      },
    }
  );

  // Check if current user can delete this post
  const canDelete = currentUser && currentUser.id === post.author.id;

  const handleQueue = () => {
    if (!currentUser) {
      console.log("User must be logged in to queue papers");
      return;
    }

    if (!paperData) {
      console.log("Cannot queue post without associated paper");
      return;
    }

    // Optimistic updates
    const newQueued = !optimisticQueued;
    const newQueues = newQueued ? optimisticQueues + 1 : optimisticQueues - 1;

    setOptimisticQueued(newQueued);
    setOptimisticQueues(newQueues);

    // Execute the action
    const formData = new FormData();
    formData.append("paperId", paperData.id);
    formData.append("postId", post.id); // Need to pass post ID to update count
    executeQueue(formData);
  };

  const handleReply = () => {
    router.push(`/posts/${post.id}`);
  };

  const handleDelete = () => {
    setShowDeleteModal(true);
  };

  const handleConfirmDelete = () => {
    const formData = new FormData();
    formData.append("postId", post.id);
    executeDelete(formData);
    setShowDeleteModal(false);
  };

  const handleCancelDelete = () => {
    setShowDeleteModal(false);
  };

  // Handle pure paper posts (papers without user commentary)
  if (post.postType === "pure-paper" && post.paper) {
    return (
      <div
        className="group relative cursor-pointer overflow-hidden rounded-2xl border border-neutral-200 bg-white px-6 pt-6 shadow-sm transition-all duration-200 before:absolute before:top-0 before:bottom-0 before:left-0 before:w-1 before:bg-transparent before:content-[''] hover:border-neutral-300 hover:bg-neutral-50 hover:shadow-md hover:before:bg-neutral-900/70"
        style={{ paddingBottom: "2px" }}
      >
        {/* Header with user info */}
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={() => onUserClick?.(post.author.id)}
              className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-full text-sm font-semibold transition-colors hover:bg-gray-200"
              style={{ backgroundColor: "#EFF3F9", color: "#131720" }}
            >
              {getUserInitials(post.author.name, post.author.email)}
            </button>
            <div className="flex items-center gap-2 text-[12.5px] leading-5 text-neutral-600">
              <button
                onClick={() => onUserClick?.(post.author.id)}
                className="cursor-pointer font-medium text-neutral-900 transition-colors hover:text-blue-600"
              >
                {post.author.name ||
                  post.author.email?.split("@")[0] ||
                  "Unknown User"}
              </button>
              <span>•</span>
              <span>{formatRelativeTime(post.createdAt)}</span>
            </div>
          </div>

          {/* Action buttons group */}
          <div className="flex items-center gap-2">
            {/* Queue button - only show for posts with papers */}
            {paperData && (
              <button
                onClick={handleQueue}
                disabled={isQueueing}
                className={`rounded-full p-1 transition-colors disabled:opacity-50 ${
                  optimisticQueued
                    ? "text-blue-500 hover:bg-neutral-100 hover:text-blue-600"
                    : "text-neutral-400 hover:bg-neutral-100 hover:text-blue-500"
                }`}
                title={`${optimisticQueues} queued`}
              >
                <svg
                  className="h-4 w-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  suppressHydrationWarning
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4"
                  />
                </svg>
              </button>
            )}

            {/* Delete button - only show for post author */}
            {canDelete && (
              <button
                onClick={handleDelete}
                disabled={isDeleting}
                className="rounded-full p-1 text-neutral-400 transition-colors hover:bg-neutral-100 hover:text-red-500 disabled:opacity-50"
                title="Delete post"
              >
                <svg
                  className="h-4 w-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  suppressHydrationWarning
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                  />
                </svg>
              </button>
            )}

            {/* Reply button - hidden on hover, rightmost when visible */}
            <button
              onClick={handleReply}
              className="rounded-full p-1 text-neutral-400 transition-colors group-hover:hidden hover:bg-neutral-100 hover:text-blue-500"
              title={`${post.replies} comments`}
            >
              <svg
                className="h-4 w-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                suppressHydrationWarning
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                />
              </svg>
            </button>

            {/* Open button - shown on hover to replace reply button */}
            <button
              onClick={handleReply}
              className="hidden cursor-pointer items-center gap-1 rounded-full border bg-white/80 px-2 py-1 text-[11px] text-neutral-700 backdrop-blur transition-colors group-hover:flex hover:bg-white"
              title="Open post"
            >
              Open <ChevronRight className="h-3 w-3" />
            </button>
          </div>
        </div>

        {/* Paper card */}
        <div className="px-4 pb-4">
          <PaperCard paper={post.paper} onPaperClick={onPaperClick} />
        </div>

        {/* Delete Post Modal */}
        <DeleteConfirmationModal
          isOpen={showDeleteModal}
          onClose={handleCancelDelete}
          onConfirm={handleConfirmDelete}
          title="Delete Post"
          message="Are you sure you want to delete this post? This action cannot be undone and will permanently remove the post and all its comments from the feed."
          isDeleting={isDeleting}
        />
      </div>
    );
  }

  // Handle user posts and paper posts with commentary
  return (
    <div className="group relative cursor-pointer overflow-hidden rounded-2xl border border-neutral-200 bg-white shadow-sm transition-all duration-200 before:absolute before:top-0 before:bottom-0 before:left-0 before:w-1 before:bg-transparent before:content-[''] hover:border-neutral-300 hover:bg-neutral-50 hover:shadow-md hover:before:bg-neutral-900/70">
      {/* Post header with user info */}
      <div className="flex items-center justify-between px-4 pt-4 pb-2">
        <div className="flex items-center gap-3">
          <button
            onClick={() => onUserClick?.(post.author.id)}
            className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-full text-sm font-medium transition-colors hover:bg-neutral-100"
            style={{ backgroundColor: "#EFF3F9", color: "#131720" }}
          >
            {getUserInitials(post.author.name, post.author.email)}
          </button>
          <div className="flex items-center gap-2 text-sm">
            <button
              onClick={() => onUserClick?.(post.author.id)}
              className="cursor-pointer font-semibold text-gray-900 transition-colors hover:text-blue-600"
            >
              {post.author.name ||
                post.author.email?.split("@")[0] ||
                "Unknown User"}
            </button>
            <span className="text-gray-500">·</span>
            <span className="text-gray-500">
              {formatRelativeTime(post.createdAt)}
            </span>
          </div>
        </div>

        {/* Action buttons group */}
        <div className="flex items-center gap-2">
          {/* Queue button - only show for posts with papers */}
          {paperData && (
            <button
              onClick={handleQueue}
              disabled={isQueueing}
              className={`rounded-full p-1 transition-colors disabled:opacity-50 ${
                optimisticQueued
                  ? "text-blue-500 hover:bg-neutral-100 hover:text-blue-600"
                  : "text-neutral-400 hover:bg-neutral-100 hover:text-blue-500"
              }`}
              title={`${optimisticQueues} queued`}
            >
              <svg
                className="h-4 w-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                suppressHydrationWarning
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4"
                />
              </svg>
            </button>
          )}

          {/* Delete button - only show for post author */}
          {canDelete && (
            <button
              onClick={handleDelete}
              disabled={isDeleting}
              className="rounded-full p-1 text-neutral-400 transition-colors hover:bg-neutral-100 hover:text-red-500 disabled:opacity-50"
              title="Delete post"
            >
              <svg
                className="h-4 w-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                suppressHydrationWarning
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                />
              </svg>
            </button>
          )}

          {/* Reply button - hidden on hover, rightmost when visible */}
          <button
            onClick={handleReply}
            className="rounded-full p-1 text-neutral-400 transition-colors group-hover:hidden hover:bg-neutral-100 hover:text-blue-500"
            title={`${post.replies} comments`}
          >
            <svg
              className="h-4 w-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              suppressHydrationWarning
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
              />
            </svg>
          </button>

          {/* Open button - shown on hover to replace reply button */}
          <button
            onClick={handleReply}
            className="hidden cursor-pointer items-center gap-1 rounded-full border bg-white/80 px-2 py-1 text-[11px] text-neutral-700 backdrop-blur transition-colors group-hover:flex hover:bg-white"
            title="Open post"
          >
            Open <ChevronRight className="h-3 w-3" />
          </button>
        </div>
      </div>

      {/* Post content with LaTeX support */}
      {post.content && cleanPostContent(post.content).trim() && (
        <div className="px-4 pb-2 text-[14px] leading-[1.55] whitespace-pre-wrap text-neutral-900">
          {linkifyText(cleanPostContent(post.content))}
        </div>
      )}

      {/* Embedded paper card for paper posts */}
      {post.postType === "paper-post" && paperData && (
        <div className="px-4 pb-4">
          <PaperCard paper={paperData} onPaperClick={onPaperClick} />
        </div>
      )}

      {/* Silent refresh for arXiv institution processing */}
      {post.content && (
        <SilentArxivRefresh
          postId={post.id}
          content={post.content}
          onInstitutionsAdded={handleInstitutionsAdded}
        />
      )}

      {/* Delete Post Modal */}
      <DeleteConfirmationModal
        isOpen={showDeleteModal}
        onClose={handleCancelDelete}
        onConfirm={handleConfirmDelete}
        title="Delete Post"
        message="Are you sure you want to delete this post? This action cannot be undone and will permanently remove the post and all its comments from the feed."
        isDeleting={isDeleting}
      />
    </div>
  );
};
