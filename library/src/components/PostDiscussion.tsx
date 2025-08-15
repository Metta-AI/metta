"use client";

import { FC, useState, useEffect } from "react";
import { useAction } from "next-safe-action/hooks";
import Link from "next/link";

import { FeedPostDTO } from "@/posts/data/feed";
import { CommentDTO } from "@/posts/data/comments";
import { createCommentAction } from "@/posts/actions/createCommentAction";
import { createBotResponseAction } from "@/posts/actions/createBotResponseAction";
import { loadCommentsAction } from "@/posts/actions/loadCommentsAction";
import { deleteCommentAction } from "@/posts/actions/deleteCommentAction";
import { DeleteConfirmationModal } from "@/components/DeleteConfirmationModal";
import { ThreadedComment } from "@/posts/components/ThreadedComment";
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
      onSuccess: (result) => {
        // Check if we need to trigger bot response
        if (
          result?.data?.comment &&
          containsBotMention(result.data.comment.content)
        ) {
          const botMessage = result.data.comment.content
            .replace(/@library_bot\s*/i, "")
            .trim();

          // Trigger bot response
          const botFormData = new FormData();
          botFormData.append("postId", post.id);
          botFormData.append("parentCommentId", result.data.comment.id);
          botFormData.append("userMessage", botMessage);
          executeCreateBotResponse(botFormData);
        }

        // For non-bot comments, clear input and refresh
        if (
          !result?.data?.comment ||
          !containsBotMention(result.data.comment.content)
        ) {
          setNewComment("");
        }

        // Refresh comments to show real data (replacing optimistic updates)
        const formData = new FormData();
        formData.append("postId", post.id);
        executeLoadComments(formData);
      },
      onError: (error) => {
        console.error("Error creating comment:", error);
        // Refresh comments to remove optimistic updates on error
        const formData = new FormData();
        formData.append("postId", post.id);
        executeLoadComments(formData);
      },
    });

  // Create bot response action (async)
  const { execute: executeCreateBotResponse, isExecuting: isBotResponding } =
    useAction(createBotResponseAction, {
      onSuccess: () => {
        // Refresh comments to show the actual bot response
        const formData = new FormData();
        formData.append("postId", post.id);
        executeLoadComments(formData);
      },
      onError: (error) => {
        console.error("Error creating bot response:", error);
        // Refresh to remove thinking state
        const formData = new FormData();
        formData.append("postId", post.id);
        executeLoadComments(formData);
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

  // Check if comment mentions @library_bot
  const containsBotMention = (text: string) => {
    return /@library_bot\b/i.test(text);
  };

  // Handle comment submission
  const handleSubmitComment = async () => {
    if (!newComment.trim() || !currentUser) return;

    const commentContent = newComment.trim();

    // Check if comment contains @library_bot mention
    if (containsBotMention(commentContent)) {
      // For bot mentions: show optimistic user comment only

      // Generate optimistic user comment
      const optimisticUserComment: CommentDTO = {
        id: `optimistic-user-${Date.now()}`,
        content: commentContent,
        postId: post.id,
        parentId: null,
        isBot: false,
        author: {
          id: currentUser.id,
          name: currentUser.name,
          email: currentUser.email,
          image: null,
        },
        createdAt: new Date(),
        updatedAt: new Date(),
        depth: 0,
      };

      // Add optimistic user comment immediately
      setComments((prev) => [...prev, optimisticUserComment]);

      // Clear input
      setNewComment("");

      // Create actual user comment and trigger bot response
      const formData = new FormData();
      formData.append("postId", post.id);
      formData.append("content", commentContent);
      executeCreateComment(formData);
    } else {
      // Regular comment - use existing flow
      const formData = new FormData();
      formData.append("postId", post.id);
      formData.append("content", commentContent);
      executeCreateComment(formData);
    }
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
    } else if (
      e.key === "Tab" &&
      newComment.endsWith("@library") &&
      !containsBotMention(newComment)
    ) {
      e.preventDefault();
      setNewComment(newComment + "_bot ");
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
              <div className="relative">
                <textarea
                  className={`min-h-[80px] w-full resize-none rounded-lg border px-3 py-2 text-sm text-gray-900 placeholder-gray-400 focus:border-transparent focus:ring-2 ${
                    containsBotMention(newComment)
                      ? "border-green-300 bg-green-50 focus:ring-green-500"
                      : newComment.endsWith("@library")
                        ? "border-blue-300 bg-blue-50/30 focus:ring-blue-500"
                        : "border-gray-200 bg-white focus:ring-blue-500"
                  }`}
                  placeholder="Write a comment... (Try @library_bot to ask about this paper)"
                  value={newComment}
                  onChange={(e) => setNewComment(e.target.value)}
                  onKeyDown={handleKeyPress}
                />

                {containsBotMention(newComment) && (
                  <div className="mt-2 flex items-center gap-2 text-sm text-green-600">
                    <span className="flex h-5 w-5 items-center justify-center rounded bg-green-100 text-xs">
                      🤖
                    </span>
                    Library Bot will respond to your message
                  </div>
                )}

                {newComment.endsWith("@library") &&
                  !containsBotMention(newComment) && (
                    <div className="mt-1 text-xs font-medium text-blue-600">
                      Press{" "}
                      <kbd className="rounded border border-blue-200 bg-blue-100 px-1 py-0.5 font-mono text-xs">
                        Tab
                      </kbd>{" "}
                      to complete @library_bot
                    </div>
                  )}
              </div>
              <div className="mt-2 flex justify-end">
                <button
                  className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
                    newComment.trim() && !isSubmitting && !isBotResponding
                      ? containsBotMention(newComment)
                        ? "bg-green-600 text-white hover:bg-green-700"
                        : "bg-blue-600 text-white hover:bg-blue-700"
                      : "cursor-not-allowed bg-gray-200 text-gray-500"
                  }`}
                  disabled={
                    !newComment.trim() || isSubmitting || isBotResponding
                  }
                  onClick={handleSubmitComment}
                >
                  {isSubmitting
                    ? "Posting..."
                    : containsBotMention(newComment)
                      ? "Ask Library Bot"
                      : "Comment"}
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
            <ThreadedComment
              key={comment.id}
              comment={comment}
              postId={post.id}
              currentUser={currentUser}
              onCommentUpdated={() => {
                const formData = new FormData();
                formData.append("postId", post.id);
                executeLoadComments(formData);
              }}
              onCommentDeleted={() => {
                // Just refresh comments when a deletion is completed
                const formData = new FormData();
                formData.append("postId", post.id);
                executeLoadComments(formData);
              }}
            />
          ))
        ) : (
          <div className="py-8 text-center">
            <p className="text-gray-600">
              No comments yet. Be the first to comment!
            </p>
          </div>
        )}

        {/* Show bot thinking indicator */}
        {isBotResponding && (
          <div className="border-l-2 border-green-200 bg-green-50/30 py-2 pl-4">
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0">
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-green-600 text-xs font-medium text-white">
                  🤖
                </div>
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-medium text-green-700">
                    Library Bot
                    <span className="ml-1 inline-flex items-center rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-800">
                      Bot
                    </span>
                  </span>
                  <span className="text-xs text-gray-500">now</span>
                </div>
                <div className="mt-1 flex items-center gap-2 text-sm text-gray-600">
                  <div className="inline-block h-4 w-4 animate-spin rounded-full border-b-2 border-green-600"></div>
                  Analyzing the paper and generating response...
                </div>
              </div>
            </div>
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
