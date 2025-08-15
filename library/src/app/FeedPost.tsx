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
      <div className="border-b border-gray-200 bg-white p-6">
        <PaperCard paper={post.paper} onPaperClick={onPaperClick} />
      </div>
    );
  }

  // Handle user posts and paper posts with commentary
  return (
    <div className="border-b border-gray-200 bg-white p-6">
      {/* Post header with user info */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            onClick={() => onUserClick?.(post.author.id)}
            className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-full bg-blue-600 text-sm font-semibold text-white transition-colors hover:bg-blue-700"
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

        {/* Delete button - only show for post author */}
        {canDelete && (
          <button
            onClick={handleDelete}
            disabled={isDeleting}
            className="rounded-full p-1 text-gray-400 transition-colors hover:bg-gray-100 hover:text-red-500 disabled:opacity-50"
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
      </div>

      {/* Post content with LaTeX support */}
      {post.content && (
        <div className="mb-4 leading-relaxed whitespace-pre-wrap text-gray-900">
          {linkifyText(post.content)}
        </div>
      )}

      {/* Embedded paper card for paper posts */}
      {post.postType === "paper-post" && paperData && (
        <div className="mb-4">
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

      {/* Post interaction buttons */}
      <div className="flex items-center gap-6 text-gray-500">
        {/* Reply button */}
        <button
          onClick={handleReply}
          className="flex items-center gap-2 transition-colors hover:text-gray-700"
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
          <span className="text-sm">{post.replies}</span>
        </button>

        {/* Queue button - only show for posts with papers */}
        {paperData ? (
          <button
            onClick={handleQueue}
            disabled={isQueueing}
            className={`flex items-center gap-2 transition-colors ${
              optimisticQueued
                ? "text-blue-500 hover:text-blue-600"
                : "text-gray-500 hover:text-blue-500"
            } disabled:opacity-50`}
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
            <span className="text-sm">{optimisticQueues}</span>
          </button>
        ) : (
          <div className="flex items-center gap-2 text-gray-300">
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
            <span className="text-sm">{optimisticQueues}</span>
          </div>
        )}

        {/* Share button */}
        <button className="flex items-center gap-2 transition-colors hover:text-gray-700">
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
              d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.367 2.684 3 3 0 00-5.367-2.684z"
            />
          </svg>
        </button>
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
};
