"use client";

import React, { useState, useEffect } from "react";
import { MessageSquare } from "lucide-react";
import { useAction } from "next-safe-action/hooks";

import { CommentDTO } from "@/posts/data/comments";
import { createCommentAction } from "@/posts/actions/createCommentAction";
import { loadCommentsAction } from "@/posts/actions/loadCommentsAction";
import { deleteCommentAction } from "@/posts/actions/deleteCommentAction";
import { createBotResponseAction } from "@/posts/actions/createBotResponseAction";
import { DeleteConfirmationModal } from "@/components/DeleteConfirmationModal";

interface ThreadedCommentsProps {
  postId: string;
  currentUser: {
    id: string;
    name?: string | null;
    email?: string | null;
  } | null;
  showBackToFeed?: boolean;
  initialComments?: CommentDTO[];
  onCommentCountChange?: (delta: number) => void;
}

// --- Helpers ----------------------------------------------------------------
function countComments(nodes: CommentDTO[]): number {
  let n = 0;
  for (const c of nodes) {
    n += 1;
    if (c.replies) n += countComments(c.replies);
  }
  return n;
}

function getUserInitials(name: string | null, email: string | null): string {
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
}

function formatRelativeTime(date: Date): string {
  const now = new Date();
  const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);

  if (diffInSeconds < 60) return "now";
  if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m`;
  if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h`;
  if (diffInSeconds < 604800) return `${Math.floor(diffInSeconds / 86400)}d`;
  return `${Math.floor(diffInSeconds / 604800)}w`;
}

// Check if comment contains bot mention
function containsBotMention(text: string): boolean {
  return /@library_bot/i.test(text);
}

// --- Avatar component -------------------------------------------------------
function Avatar({
  className,
  children,
}: {
  className: string;
  children: React.ReactNode;
}) {
  return <div className={className}>{children}</div>;
}

function AvatarFallback({
  children,
  isBot,
}: {
  children: React.ReactNode;
  isBot?: boolean;
}) {
  return (
    <div
      className={`flex h-full w-full items-center justify-center rounded-full text-xs font-semibold ${
        isBot ? "bg-green-100 text-green-700" : "bg-slate-200 text-slate-700"
      }`}
    >
      {children}
    </div>
  );
}

// --- Comment components -----------------------------------------------------
function CommentItem({ c }: { c: CommentDTO }) {
  return (
    <div className="flex items-start gap-3">
      <Avatar className="h-7 w-7">
        <AvatarFallback isBot={c.isBot}>
          {c.isBot ? "🤖" : getUserInitials(c.author.name, c.author.email)}
        </AvatarFallback>
      </Avatar>
      <div className="min-w-0 flex-1">
        <div className="text-[12px] text-neutral-600">
          <span
            className={`font-medium ${c.isBot ? "text-green-700" : "text-neutral-900"}`}
          >
            {c.isBot
              ? "@library_bot"
              : c.author.name || c.author.email?.split("@")[0] || "Unknown"}
          </span>{" "}
          • {formatRelativeTime(c.createdAt)}
        </div>
        <div className="mt-0.5 text-[14px] leading-[1.55] whitespace-pre-wrap text-neutral-900">
          {c.content}
        </div>
      </div>
    </div>
  );
}

function CommentComposer({
  placeholder,
  onSubmit,
  onCancel,
  value,
  onChange,
  showBotHint = false,
}: {
  placeholder: string;
  onSubmit: (text: string) => void;
  onCancel?: () => void;
  value: string;
  onChange: (value: string) => void;
  showBotHint?: boolean;
}) {
  const handleSubmit = () => {
    if (value.trim()) {
      onSubmit(value.trim());
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    } else if (
      e.key === "Tab" &&
      value.endsWith("@library") &&
      !containsBotMention(value)
    ) {
      e.preventDefault();
      onChange(value + "_bot ");
    }
  };

  return (
    <div
      className="mt-2 rounded-xl border bg-white p-2"
      onClick={(e) => e.stopPropagation()}
    >
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        className={`min-h-[64px] w-full resize-none border-0 p-2 text-[14px] leading-[1.5] placeholder-gray-400 focus:ring-0 focus:outline-none ${
          containsBotMention(value)
            ? "bg-green-50"
            : value.endsWith("@library")
              ? "bg-blue-50/30"
              : ""
        }`}
        rows={3}
      />

      {containsBotMention(value) && (
        <div className="mt-2 flex items-center gap-2 text-sm text-green-600">
          <span className="flex h-5 w-5 items-center justify-center rounded bg-green-100 text-xs">
            🤖
          </span>
          Library Bot will respond to your message
        </div>
      )}

      <div className="mt-2 flex items-center gap-2">
        <button
          onClick={handleSubmit}
          disabled={!value.trim()}
          className="rounded px-3 py-1 text-sm text-white transition-colors hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
          style={{ backgroundColor: "#0F172A" }}
        >
          Reply
        </button>
        {onCancel && (
          <button
            onClick={onCancel}
            className="px-3 py-1 text-sm text-gray-600 transition-colors hover:text-gray-800"
          >
            Cancel
          </button>
        )}
      </div>
    </div>
  );
}

function CommentNode({
  c,
  depth = 0,
  onReply,
  activeComposer,
  setActiveComposer,
  currentUser,
  onDelete,
}: {
  c: CommentDTO;
  depth?: number;
  onReply: (parentId: string, text: string) => void;
  activeComposer: string | null;
  setActiveComposer: (id: string | null) => void;
  currentUser: any;
  onDelete: (commentId: string) => void;
}) {
  const isOpen = activeComposer === c.id;
  const [replyText, setReplyText] = useState("");

  return (
    <div className="space-y-2">
      <div className="group relative">
        <CommentItem c={c} />

        {/* Actions */}
        <div className="-mt-1 flex items-center gap-3 pl-10 text-[12px] text-neutral-600">
          {currentUser && (
            <button
              className="hover:underline"
              onClick={(e) => {
                e.stopPropagation();
                setActiveComposer(isOpen ? null : c.id);
                if (!isOpen) setReplyText("");
              }}
            >
              Reply
            </button>
          )}

          {currentUser && currentUser.id === c.author.id && !c.isBot && (
            <button
              className="hover:text-red-600 hover:underline"
              onClick={(e) => {
                e.stopPropagation();
                onDelete(c.id);
              }}
            >
              Delete
            </button>
          )}
        </div>
      </div>

      {isOpen && (
        <div className="ml-6">
          <CommentComposer
            placeholder={`Reply to ${c.author.name || c.author.email?.split("@")[0] || "user"}…`}
            value={replyText}
            onChange={setReplyText}
            onSubmit={(text) => {
              onReply(c.id, text);
              setActiveComposer(null);
              setReplyText("");
            }}
            onCancel={() => {
              setActiveComposer(null);
              setReplyText("");
            }}
            showBotHint
          />
        </div>
      )}

      {c.replies && c.replies.length > 0 && (
        <div className="ml-6 space-y-3 border-l border-neutral-200 pl-3">
          {c.replies.map((r: CommentDTO) => (
            <CommentNode
              key={r.id}
              c={r}
              depth={depth + 1}
              onReply={onReply}
              activeComposer={activeComposer}
              setActiveComposer={setActiveComposer}
              currentUser={currentUser}
              onDelete={onDelete}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// --- Main component ---------------------------------------------------------
export const ThreadedComments: React.FC<ThreadedCommentsProps> = ({
  postId,
  currentUser,
  showBackToFeed = false,
  initialComments,
  onCommentCountChange,
}) => {
  const [comments, setComments] = useState<CommentDTO[]>(initialComments || []);
  const [isLoading, setIsLoading] = useState(false);
  const [activeComposer, setActiveComposer] = useState<string | null>(null);
  const [rootCommentText, setRootCommentText] = useState("");
  const [isBotResponding, setIsBotResponding] = useState(false);
  const [commentToDelete, setCommentToDelete] = useState<string | null>(null);

  // Load comments action
  const { execute: executeLoadComments } = useAction(loadCommentsAction, {
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
  const { execute: executeCreateComment } = useAction(createCommentAction, {
    onSuccess: (result) => {
      // Check if we need to trigger bot response
      if (
        result?.data?.comment &&
        containsBotMention(result.data.comment.content)
      ) {
        setIsBotResponding(true);
        const botMessage = result.data.comment.content
          .replace(/@library_bot\s*/i, "")
          .trim();

        // Trigger bot response
        const botFormData = new FormData();
        botFormData.append("postId", postId);
        botFormData.append("parentCommentId", result.data.comment.id);
        botFormData.append("userMessage", botMessage);
        executeCreateBotResponse(botFormData);
      }

      // For non-bot comments, clear input and refresh
      if (
        !result?.data?.comment ||
        !containsBotMention(result.data.comment.content)
      ) {
        setRootCommentText("");
      }

      // Notify parent of comment count change
      onCommentCountChange?.(1);

      // Refresh comments
      const formData = new FormData();
      formData.append("postId", postId);
      executeLoadComments(formData);
    },
    onError: (error) => {
      console.error("Error creating comment:", error);
      // Refresh comments to remove optimistic updates on error
      const formData = new FormData();
      formData.append("postId", postId);
      executeLoadComments(formData);
    },
  });

  // Create bot response action
  const { execute: executeCreateBotResponse } = useAction(
    createBotResponseAction,
    {
      onSuccess: () => {
        setIsBotResponding(false);
        setRootCommentText("");

        // Notify parent of comment count change (bot response)
        onCommentCountChange?.(1);

        // Refresh comments to show bot response
        const formData = new FormData();
        formData.append("postId", postId);
        executeLoadComments(formData);
      },
      onError: (error) => {
        console.error("Error creating bot response:", error);
        setIsBotResponding(false);
        // Still refresh to show user comment
        const formData = new FormData();
        formData.append("postId", postId);
        executeLoadComments(formData);
      },
    }
  );

  // Delete comment action
  const { execute: executeDeleteComment } = useAction(deleteCommentAction, {
    onSuccess: () => {
      setCommentToDelete(null);
      // Refresh comments
      const formData = new FormData();
      formData.append("postId", postId);
      executeLoadComments(formData);
    },
    onError: (error) => {
      console.error("Error deleting comment:", error);
      setCommentToDelete(null);
    },
  });

  // Load comments when component mounts
  useEffect(() => {
    if (!initialComments && comments.length === 0) {
      setIsLoading(true);
      const formData = new FormData();
      formData.append("postId", postId);
      executeLoadComments(formData);
    }
  }, [postId, executeLoadComments, comments.length, initialComments]);

  function addReply(parentId: string | null, text: string) {
    if (!currentUser) return;

    const formData = new FormData();
    formData.append("postId", postId);
    formData.append("content", text);
    if (parentId) {
      formData.append("parentId", parentId);
    }
    executeCreateComment(formData);
  }

  const isRootComposerOpen = activeComposer === "root";

  return (
    <div className="w-full">
      {/* Comments thread */}
      {isLoading ? (
        <div className="py-4 text-center">
          <div className="inline-block h-4 w-4 animate-spin rounded-full border-b-2 border-neutral-900"></div>
        </div>
      ) : (
        <div className="space-y-3">
          {comments.map((c: CommentDTO) => (
            <CommentNode
              key={c.id}
              c={c}
              onReply={(parentId: string, text: string) =>
                addReply(parentId, text)
              }
              activeComposer={activeComposer}
              setActiveComposer={setActiveComposer}
              currentUser={currentUser}
              onDelete={(commentId) => setCommentToDelete(commentId)}
            />
          ))}

          {/* Show bot thinking indicator */}
          {isBotResponding && (
            <div className="flex items-start gap-3 opacity-70">
              <Avatar className="h-7 w-7">
                <AvatarFallback isBot>🤖</AvatarFallback>
              </Avatar>
              <div className="min-w-0 flex-1">
                <div className="text-[12px] font-medium text-green-700">
                  @library_bot • thinking...
                </div>
                <div className="mt-1 flex gap-1">
                  <span className="h-2 w-2 animate-bounce rounded-full bg-gray-400 [animation-delay:-0.3s]"></span>
                  <span className="h-2 w-2 animate-bounce rounded-full bg-gray-400 [animation-delay:-0.15s]"></span>
                  <span className="h-2 w-2 animate-bounce rounded-full bg-gray-400"></span>
                </div>
              </div>
            </div>
          )}

          {/* Root-level composer */}
          {currentUser && (
            <div className="pt-1">
              {isRootComposerOpen ? (
                <CommentComposer
                  placeholder={
                    showBackToFeed
                      ? "Write a comment... (Try @library_bot to ask about this paper)"
                      : "Write a comment…"
                  }
                  value={rootCommentText}
                  onChange={setRootCommentText}
                  onSubmit={(text) => {
                    addReply(null, text);
                    setActiveComposer(null);
                  }}
                  onCancel={() => {
                    setActiveComposer(null);
                    setRootCommentText("");
                  }}
                  showBotHint={showBackToFeed}
                />
              ) : (
                <button
                  onClick={() => setActiveComposer("root")}
                  className="w-full rounded-xl border bg-white p-4 text-left text-[14px] text-gray-400 hover:bg-gray-50"
                >
                  {showBackToFeed
                    ? "Write a comment... (Try @library_bot to ask about this paper)"
                    : "Write a comment…"}
                </button>
              )}
            </div>
          )}

          {!currentUser && (
            <div className="py-4 text-center text-[14px] text-neutral-600">
              Sign in to join the conversation
            </div>
          )}
        </div>
      )}

      {/* Delete Comment Confirmation Modal */}
      <DeleteConfirmationModal
        isOpen={!!commentToDelete}
        onClose={() => setCommentToDelete(null)}
        onConfirm={() => {
          if (commentToDelete) {
            // Find the comment and count how many will be deleted (including children)
            const findCommentAndChildren = (
              commentsList: CommentDTO[],
              targetId: string
            ): CommentDTO | null => {
              for (const comment of commentsList) {
                if (comment.id === targetId) return comment;
                if (comment.replies) {
                  const found = findCommentAndChildren(
                    comment.replies,
                    targetId
                  );
                  if (found) return found;
                }
              }
              return null;
            };

            const commentToDeleteObj = findCommentAndChildren(
              comments,
              commentToDelete
            );
            if (commentToDeleteObj) {
              // Count this comment + all its children
              const deleteCount = countComments([commentToDeleteObj]);

              // Notify parent of negative count change
              onCommentCountChange?.(-deleteCount);
            }

            const formData = new FormData();
            formData.append("commentId", commentToDelete);
            executeDeleteComment(formData);
          }
        }}
        title="Delete Comment"
        message="Are you sure you want to delete this comment? This action cannot be undone."
        isDeleting={false}
      />
    </div>
  );
};
