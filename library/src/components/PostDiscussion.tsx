"use client";

import { FC, useState, useEffect } from "react";
import { useAction } from "next-safe-action/hooks";
import Link from "next/link";

import { FeedPostDTO } from "@/posts/data/feed";
import { CommentDTO } from "@/posts/data/comments";
import { createCommentAction } from "@/posts/actions/createCommentAction";
import { loadCommentsAction } from "@/posts/actions/loadCommentsAction";
import { deleteCommentAction } from "@/posts/actions/deleteCommentAction";
import { DeleteConfirmationModal } from "@/components/DeleteConfirmationModal";
import { linkifyText } from "@/lib/utils/linkify";

interface PostDiscussionProps {
  post: FeedPostDTO;
  currentUser: {
    id: string;
    name?: string | null;
    email?: string | null;
  } | null;
}

/**
 * PostDiscussion Component
 *
 * Shows the main post at the top followed by all comments below in a discussion format.
 * Similar to Twitter's post discussion layout.
 */
export const PostDiscussion: FC<PostDiscussionProps> = ({
  post,
  currentUser,
}) => {
  const [comments, setComments] = useState<CommentDTO[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [newComment, setNewComment] = useState("");
  const [commentToDelete, setCommentToDelete] = useState<string | null>(null);

  // Load comments action
  const { execute: executeLoadComments, isExecuting: isLoadingComments } =
    useAction(loadCommentsAction, {
      onSuccess: (result) => {
        if (result && result.data) {
          setComments(result.data);
        }
        setIsLoading(false);
      },
      onError: (error) => {
        console.error("Error loading comments:", error);
        setComments([]);
        setIsLoading(false);
      },
    });

  // Create comment action
  const { execute: executeCreateComment, isExecuting: isSubmitting } =
    useAction(createCommentAction, {
      onSuccess: () => {
        setNewComment("");
        // Refresh comments
        const formData = new FormData();
        formData.append("postId", post.id);
        executeLoadComments(formData);
      },
      onError: (error) => {
        console.error("Error creating comment:", error);
      },
    });

  // Delete comment action
  const { execute: executeDeleteComment, isExecuting: isDeletingComment } =
    useAction(deleteCommentAction, {
      onSuccess: () => {
        // Refresh comments
        const formData = new FormData();
        formData.append("postId", post.id);
        executeLoadComments(formData);
      },
      onError: (error) => {
        console.error("Error deleting comment:", error);
      },
    });

  // Load comments on component mount
  useEffect(() => {
    setIsLoading(true);
    const formData = new FormData();
    formData.append("postId", post.id);
    executeLoadComments(formData);
  }, [post.id, executeLoadComments]);

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

  // Handle comment submission
  const handleSubmitComment = async () => {
    if (!newComment.trim() || !currentUser) return;

    const formData = new FormData();
    formData.append("postId", post.id);
    formData.append("content", newComment.trim());

    executeCreateComment(formData);
  };

  // Handle comment deletion
  const handleDeleteComment = (commentId: string) => {
    setCommentToDelete(commentId);
  };

  const handleConfirmDeleteComment = () => {
    if (commentToDelete) {
      const formData = new FormData();
      formData.append("commentId", commentToDelete);
      executeDeleteComment(formData);
      setCommentToDelete(null);
    }
  };

  const handleCancelDeleteComment = () => {
    setCommentToDelete(null);
  };

  // Handle key press in comment input
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmitComment();
    }
  };

  return (
    <div className="w-full">
      {/* Back to feed button */}
      <div className="mb-4">
        <Link
          href="/"
          className="inline-flex items-center gap-2 text-gray-600 transition-colors hover:text-gray-900"
        >
          <svg
            className="h-4 w-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 19l-7-7 7-7"
            />
          </svg>
          Back to feed
        </Link>
      </div>

      {/* Original Post */}
      <div className="mb-6 rounded-lg border border-gray-200 bg-white p-6">
        {/* Post header with user info */}
        <div className="mb-4 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-600 text-sm font-semibold text-white">
            {getUserInitials(post.author.name, post.author.email)}
          </div>
          <div className="flex items-center gap-2 text-sm">
            <span className="font-semibold text-gray-900">
              {post.author.name ||
                post.author.email?.split("@")[0] ||
                "Unknown User"}
            </span>
            <span className="text-gray-500">·</span>
            <span className="text-gray-500">
              {formatRelativeTime(post.createdAt)}
            </span>
          </div>
        </div>

        {/* Post title */}
        <h1 className="mb-4 text-xl font-bold text-gray-900">{post.title}</h1>

        {/* Post content */}
        {post.content && (
          <div className="mb-4 leading-relaxed whitespace-pre-wrap text-gray-900">
            {linkifyText(post.content)}
          </div>
        )}

        {/* Post metrics */}
        <div className="flex items-center gap-6 text-sm text-gray-500">
          <span>{comments.length} replies</span>
          <span>{post.queues} queues</span>
        </div>
      </div>

      {/* Comment Input */}
      {currentUser ? (
        <div className="mb-6 rounded-lg border border-gray-200 bg-white p-4">
          <div className="flex gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-600 text-xs font-semibold text-white">
              {getUserInitials(currentUser.name, currentUser.email)}
            </div>
            <div className="flex-1">
              <textarea
                className="min-h-[80px] w-full resize-none rounded-lg border border-gray-200 px-3 py-2 text-sm text-gray-900 placeholder-gray-400 focus:border-transparent focus:ring-2 focus:ring-blue-500"
                placeholder="Write a comment..."
                value={newComment}
                onChange={(e) => setNewComment(e.target.value)}
                onKeyDown={handleKeyPress}
              />
              <div className="mt-2 flex justify-end">
                <button
                  className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
                    newComment.trim() && !isSubmitting
                      ? "bg-blue-600 text-white hover:bg-blue-700"
                      : "cursor-not-allowed bg-gray-200 text-gray-500"
                  }`}
                  disabled={!newComment.trim() || isSubmitting}
                  onClick={handleSubmitComment}
                >
                  {isSubmitting ? "Posting..." : "Comment"}
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="mb-6 rounded-lg border border-gray-200 bg-gray-50 p-4 text-center">
          <p className="text-gray-600">Sign in to join the discussion</p>
        </div>
      )}

      {/* Comments List */}
      <div className="space-y-4">
        {isLoading ? (
          <div className="py-8 text-center">
            <div className="inline-block h-6 w-6 animate-spin rounded-full border-b-2 border-blue-600"></div>
            <p className="mt-2 text-gray-600">Loading comments...</p>
          </div>
        ) : comments.length > 0 ? (
          comments.map((comment) => (
            <div
              key={comment.id}
              className="rounded-lg border border-gray-200 bg-white p-4"
            >
              {/* Comment header */}
              <div className="mb-3 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-600 text-xs font-semibold text-white">
                    {getUserInitials(comment.author.name, comment.author.email)}
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    <span className="font-semibold text-gray-900">
                      {comment.author.name ||
                        comment.author.email?.split("@")[0] ||
                        "Unknown User"}
                    </span>
                    <span className="text-gray-500">·</span>
                    <span className="text-gray-500">
                      {formatRelativeTime(comment.createdAt)}
                    </span>
                  </div>
                </div>

                {/* Delete button for comment author */}
                {currentUser && currentUser.id === comment.author.id && (
                  <button
                    onClick={() => handleDeleteComment(comment.id)}
                    disabled={isDeletingComment}
                    className="p-1 text-gray-400 transition-colors hover:text-red-600"
                  >
                    <svg
                      className="h-4 w-4"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9zM4 5a2 2 0 012-2h8a2 2 0 012 2v10a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 112 0v4a1 1 0 11-2 0V9zm4 0a1 1 0 112 0v4a1 1 0 11-2 0V9z"
                        clipRule="evenodd"
                      />
                    </svg>
                  </button>
                )}
              </div>

              {/* Comment content */}
              <div className="leading-relaxed whitespace-pre-wrap text-gray-900">
                {linkifyText(comment.content)}
              </div>
            </div>
          ))
        ) : (
          <div className="py-8 text-center">
            <p className="text-gray-600">
              No comments yet. Be the first to comment!
            </p>
          </div>
        )}
      </div>

      {/* Delete Comment Modal */}
      <DeleteConfirmationModal
        isOpen={!!commentToDelete}
        onClose={handleCancelDeleteComment}
        onConfirm={handleConfirmDeleteComment}
        title="Delete Comment"
        message="Are you sure you want to delete this comment? This action cannot be undone."
        isDeleting={isDeletingComment}
      />
    </div>
  );
};
