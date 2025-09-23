"use client";

import { FC, useState } from "react";
import { useAction } from "next-safe-action/hooks";
import { CommentDTO } from "@/posts/data/comments";
import { createCommentAction } from "@/posts/actions/createCommentAction";
import { deleteCommentAction } from "@/posts/actions/deleteCommentAction";
import { DeleteConfirmationModal } from "@/components/DeleteConfirmationModal";
import { linkifyText } from "@/lib/utils/linkify";
import { RichTextRenderer } from "@/components/RichTextRenderer";
import { MentionInput } from "@/components/MentionInput";
import { parseMentions } from "@/lib/mentions";

interface ThreadedCommentProps {
  comment: CommentDTO;
  postId: string;
  currentUser: {
    id: string;
    name?: string | null;
    email?: string | null;
  } | null;
  onCommentUpdated: () => void;
  onCommentDeleted?: () => void;
  maxDepth?: number;
}

/**
 * ThreadedComment Component
 *
 * Recursively renders a comment and its replies with proper nesting and indentation.
 * Includes reply functionality and delete capabilities.
 */
export const ThreadedComment: FC<ThreadedCommentProps> = ({
  comment,
  postId,
  currentUser,
  onCommentUpdated,
  onCommentDeleted,
  maxDepth = 8,
}) => {
  const [isReplying, setIsReplying] = useState(false);
  const [replyContent, setReplyContent] = useState("");
  const [replyMentions, setReplyMentions] = useState<string[]>([]);
  const [commentToDelete, setCommentToDelete] = useState<string | null>(null);

  const depth = comment.depth || 0;
  const indentLevel = Math.min(depth, maxDepth);

  // Handle content changes that also updates mentions
  const handleReplyContentChange = (newContent: string) => {
    setReplyContent(newContent);

    // Parse mentions from content
    const parsedMentions = parseMentions(newContent);
    const mentionValues = parsedMentions.map((m) => m.raw);
    setReplyMentions(mentionValues);
  };

  // Create reply action
  const { execute: executeCreateReply, isExecuting: isSubmittingReply } =
    useAction(createCommentAction, {
      onSuccess: () => {
        setReplyContent("");
        setReplyMentions([]);
        setIsReplying(false);
        onCommentUpdated();
      },
      onError: (error) => {
        console.error("Error creating reply:", error);
      },
    });

  // Delete comment action
  const { execute: executeDeleteComment, isExecuting: isDeletingComment } =
    useAction(deleteCommentAction, {
      onSuccess: () => {
        onCommentUpdated();
        if (onCommentDeleted) {
          onCommentDeleted();
        }
      },
      onError: (error) => {
        console.error("Error deleting comment:", error);
      },
    });

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

  // Handle reply submission
  const handleSubmitReply = async () => {
    if (!replyContent.trim() || !currentUser) return;

    const formData = new FormData();
    formData.append("postId", postId);
    formData.append("parentId", comment.id);
    formData.append("content", replyContent.trim());

    // Add mentions to form data
    if (replyMentions.length > 0) {
      formData.append("mentions", JSON.stringify(replyMentions));
    }

    executeCreateReply(formData);
  };

  // Handle comment deletion - show confirmation first
  const handleDeleteComment = () => {
    setCommentToDelete(comment.id);
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

  // Handle key press in reply input
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmitReply();
    }
  };

  return (
    <div
      className={`${indentLevel > 0 ? `ml-${Math.min(indentLevel * 4, 12)}` : ""}`}
    >
      {/* Main comment */}
      <div
        className={`border-l-2 py-2 pl-4 ${comment.isBot ? "border-green-200 bg-green-50/30" : "border-gray-100"}`}
      >
        <div className="flex items-start space-x-3">
          {/* Author avatar */}
          <div className="flex-shrink-0">
            {comment.isBot ? (
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-green-600 text-xs font-medium text-white">
                🤖
              </div>
            ) : comment.author.image ? (
              <img
                src={comment.author.image}
                alt={comment.author.name || "User"}
                className="h-8 w-8 rounded-full"
              />
            ) : (
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-500 text-xs font-medium text-white">
                {getUserInitials(comment.author.name, comment.author.email)}
              </div>
            )}
          </div>

          <div className="min-w-0 flex-1">
            {/* Author info and actions */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <span
                  className={`text-sm font-medium ${comment.isBot ? "text-green-700" : "text-gray-900"}`}
                >
                  {comment.author.name || comment.author.email || "Anonymous"}
                  {comment.isBot && (
                    <span className="ml-1 inline-flex items-center rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-800">
                      Bot
                    </span>
                  )}
                </span>
                <span className="text-xs text-gray-500">
                  {formatRelativeTime(comment.createdAt)}
                </span>
              </div>

              <div className="flex items-center space-x-1">
                {/* Reply button - disabled for bot comments */}
                {currentUser && depth < maxDepth && !comment.isBot && (
                  <button
                    onClick={() => setIsReplying(!isReplying)}
                    className="text-xs text-gray-500 transition-colors hover:text-blue-600"
                    disabled={isSubmittingReply}
                  >
                    {isReplying ? "Cancel" : "Reply"}
                  </button>
                )}

                {/* Delete button for comment author - disabled for bot comments */}
                {currentUser &&
                  currentUser.id === comment.author.id &&
                  !comment.isBot && (
                    <button
                      onClick={handleDeleteComment}
                      disabled={isDeletingComment}
                      className="p-1 text-gray-400 transition-colors hover:text-red-600"
                      title="Delete comment"
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
            </div>

            {/* Comment content */}
            <div className="mt-1 text-sm leading-relaxed whitespace-pre-wrap text-gray-900">
              <RichTextRenderer text={comment.content} />
            </div>

            {/* Reply form */}
            {isReplying && currentUser && (
              <div className="mt-3 space-y-2">
                <MentionInput
                  wrapperClassName="w-full"
                  value={replyContent}
                  onChange={handleReplyContentChange}
                  onMentionsChange={setReplyMentions}
                  onKeyDown={handleKeyPress}
                  placeholder="Write a reply..."
                  className="resize-none rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
                  rows={2}
                  disabled={isSubmittingReply}
                />
                <div className="flex justify-end space-x-2">
                  <button
                    onClick={() => setIsReplying(false)}
                    className="px-3 py-1 text-xs text-gray-600 transition-colors hover:text-gray-800"
                    disabled={isSubmittingReply}
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleSubmitReply}
                    disabled={!replyContent.trim() || isSubmittingReply}
                    className="rounded bg-blue-600 px-3 py-1 text-xs text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {isSubmittingReply ? "Posting..." : "Reply"}
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Nested replies */}
      {comment.replies && comment.replies.length > 0 && (
        <div className="mt-2">
          {comment.replies.map((reply) => (
            <ThreadedComment
              key={reply.id}
              comment={reply}
              postId={postId}
              currentUser={currentUser}
              onCommentUpdated={onCommentUpdated}
              onCommentDeleted={onCommentDeleted}
              maxDepth={maxDepth}
            />
          ))}
        </div>
      )}

      {/* Delete Comment Confirmation Modal */}
      <DeleteConfirmationModal
        isOpen={!!commentToDelete}
        onClose={handleCancelDeleteComment}
        onConfirm={handleConfirmDeleteComment}
        title="Delete Comment"
        message="Are you sure you want to delete this comment and all its replies? This action cannot be undone."
        isDeleting={isDeletingComment}
      />
    </div>
  );
};
