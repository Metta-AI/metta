"use client";

import { FC, useEffect, useState } from "react";
import { useAction } from "next-safe-action/hooks";
import { FeedPostDTO } from "@/posts/data/feed";
import { CommentDTO } from "@/posts/data/comments";
import { PaperCard } from "@/components/PaperCard";
import { DeleteConfirmationModal } from "@/components/DeleteConfirmationModal";
import { createCommentAction } from "@/posts/actions/createCommentAction";
import { loadCommentsAction } from "@/posts/actions/loadCommentsAction";
import { deleteCommentAction } from "@/posts/actions/deleteCommentAction";

interface CommentSidebarProps {
  post: FeedPostDTO | null;
  onClose: () => void;
  onPaperClick?: (paperId: string) => void;
  currentUser: {
    id: string;
    name?: string | null;
    email?: string | null;
  } | null;
}

/**
 * CommentSidebar Component
 *
 * A sidebar that appears on the right side when a user wants to comment on a post.
 * Shows the post details, existing comments, and a comment input box.
 */
export const CommentSidebar: FC<CommentSidebarProps> = ({
  post,
  onClose,
  onPaperClick,
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
        // Clear the input
        setNewComment("");
        // Refresh comments
        if (post) {
          const formData = new FormData();
          formData.append("postId", post.id);
          executeLoadComments(formData);
        }
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
        if (post) {
          const formData = new FormData();
          formData.append("postId", post.id);
          executeLoadComments(formData);
        }
      },
      onError: (error) => {
        console.error("Error deleting comment:", error);
      },
    });

  // Load comments when post changes
  useEffect(() => {
    if (post) {
      setIsLoading(true);
      const formData = new FormData();
      formData.append("postId", post.id);
      executeLoadComments(formData);
    } else {
      setComments([]);
    }
  }, [post, executeLoadComments]);

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
    if (!newComment.trim() || !post || !currentUser) return;

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

  if (!post) return null;

  return (
    <div className="fixed top-0 right-0 z-50 flex h-full w-96 flex-col border-l border-gray-200 bg-white shadow-lg">
      {/* Header */}
      <div className="flex flex-shrink-0 items-center justify-between border-b border-gray-200 p-4">
        <h2 className="text-lg font-semibold text-gray-900">Comments</h2>
        <button
          onClick={onClose}
          className="rounded-full p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-600"
        >
          <svg
            className="h-5 w-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      </div>

      {/* Scrollable Content Area */}
      <div className="flex-1 overflow-y-auto">
        {/* Post Summary */}
        <div className="border-b border-gray-200 bg-gray-50 p-4">
          <div className="mb-2 flex items-center gap-2">
            <div className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-600 text-xs font-semibold text-white">
              {getUserInitials(post.author.name, post.author.email)}
            </div>
            <span className="text-sm font-medium text-gray-900">
              {post.author.name ||
                post.author.email?.split("@")[0] ||
                "Unknown User"}
            </span>
            <span className="text-xs text-gray-500">
              {formatRelativeTime(post.createdAt)}
            </span>
          </div>
          <p className="line-clamp-3 text-sm text-gray-700">
            {post.content || post.title}
          </p>

          {/* Embedded paper card for paper posts */}
          {post.postType === "paper-post" && post.paper && (
            <div className="mt-3">
              <PaperCard paper={post.paper} onPaperClick={onPaperClick} />
            </div>
          )}
        </div>

        {/* Comments List */}
        <div className="p-4">
          {isLoading || isLoadingComments ? (
            <div className="flex items-center justify-center py-8">
              <div className="h-6 w-6 animate-spin rounded-full border-b-2 border-blue-600"></div>
            </div>
          ) : comments.length > 0 ? (
            <div className="space-y-4">
              {comments.map((comment) => (
                <div key={comment.id} className="border-b border-gray-100 pb-4">
                  <div className="mb-2 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="flex h-6 w-6 items-center justify-center rounded-full bg-gray-600 text-xs font-semibold text-white">
                        {getUserInitials(
                          comment.author.name,
                          comment.author.email
                        )}
                      </div>
                      <span className="text-sm font-medium text-gray-900">
                        {comment.author.name ||
                          comment.author.email?.split("@")[0] ||
                          "Unknown User"}
                      </span>
                      <span className="text-xs text-gray-500">
                        {formatRelativeTime(comment.createdAt)}
                      </span>
                    </div>
                    {/* Delete button - only show for comment author */}
                    {currentUser && comment.author.id === currentUser.id && (
                      <button
                        onClick={() => handleDeleteComment(comment.id)}
                        disabled={isDeletingComment}
                        className="rounded-full p-1 text-gray-400 transition-colors hover:bg-gray-100 hover:text-red-500 disabled:opacity-50"
                        title="Delete comment"
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
                            d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                          />
                        </svg>
                      </button>
                    )}
                  </div>
                  <p className="ml-8 text-sm whitespace-pre-wrap text-gray-700">
                    {comment.content}
                  </p>
                </div>
              ))}
            </div>
          ) : (
            <div className="py-8 text-center text-gray-500">
              <div className="mb-2">
                <svg
                  className="mx-auto h-8 w-8 text-gray-300"
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
              <p className="text-sm">No comments yet</p>
              <p className="text-xs">Be the first to comment!</p>
            </div>
          )}
        </div>

        {/* Comment Input */}
        {currentUser ? (
          <div className="border-t border-gray-200 p-4">
            <div className="flex gap-3">
              <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-blue-600 text-sm font-semibold text-white">
                {getUserInitials(currentUser.name, currentUser.email)}
              </div>
              <div className="flex-1">
                <textarea
                  value={newComment}
                  onChange={(e) => setNewComment(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Write a comment..."
                  className="w-full resize-none rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
                  rows={3}
                  disabled={isSubmitting}
                />
                <div className="mt-2 flex items-center justify-between">
                  <span className="text-xs text-gray-500">
                    Press Enter to post, Shift+Enter for new line
                  </span>
                  <button
                    onClick={handleSubmitComment}
                    disabled={!newComment.trim() || isSubmitting}
                    className="rounded-md bg-blue-600 px-3 py-1 text-sm text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {isSubmitting ? "Posting..." : "Post"}
                  </button>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="border-t border-gray-200 p-4 text-center text-gray-500">
            <p className="text-sm">Sign in to comment</p>
          </div>
        )}
      </div>

      {/* Delete Comment Modal */}
      <DeleteConfirmationModal
        isOpen={commentToDelete !== null}
        onClose={handleCancelDeleteComment}
        onConfirm={handleConfirmDeleteComment}
        title="Delete Comment"
        message="Are you sure you want to delete this comment? This action cannot be undone and the comment will be permanently removed from the discussion."
        isDeleting={isDeletingComment}
      />
    </div>
  );
};
