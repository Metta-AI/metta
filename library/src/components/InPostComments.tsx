"use client";

import React, { useState, useEffect } from "react";
import { MessageSquare } from "lucide-react";
import { useAction } from "next-safe-action/hooks";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/Button";
import { FeedPostDTO } from "@/posts/data/feed";
import { CommentDTO } from "@/posts/data/comments";
import { createCommentAction } from "@/posts/actions/createCommentAction";
import { loadCommentsAction } from "@/posts/actions/loadCommentsAction";

interface InPostCommentsProps {
  post: FeedPostDTO;
  isExpanded: boolean;
  onToggle: () => void;
  currentUser: {
    id: string;
    name?: string | null;
    email?: string | null;
  } | null;
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

function insertReply(
  nodes: CommentDTO[],
  parentId: string | null,
  reply: CommentDTO
): CommentDTO[] {
  if (parentId == null) {
    return [...nodes, reply];
  }
  return nodes.map((c) => {
    if (c.id === parentId) {
      const replies = c.replies ? [...c.replies, reply] : [reply];
      return { ...c, replies };
    }
    if (c.replies)
      return { ...c, replies: insertReply(c.replies, parentId, reply) };
    return c;
  });
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

function AvatarFallback({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-full w-full items-center justify-center rounded-full bg-slate-200 text-xs font-semibold text-slate-700">
      {children}
    </div>
  );
}

// --- Comment components -----------------------------------------------------
function CommentItem({ c }: { c: CommentDTO }) {
  return (
    <div className="flex items-start gap-3">
      <Avatar className="h-7 w-7">
        <AvatarFallback>
          {getUserInitials(c.author.name, c.author.email)}
        </AvatarFallback>
      </Avatar>
      <div className="min-w-0 flex-1">
        <div className="text-[12px] text-neutral-600">
          <span className="font-medium text-neutral-900">
            {c.author.name || c.author.email?.split("@")[0] || "Unknown"}
          </span>{" "}
          • {formatRelativeTime(c.createdAt)}
        </div>
        <div className="mt-0.5 text-[14px] leading-[1.55] text-neutral-900">
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
}: {
  placeholder: string;
  onSubmit: (text: string) => void;
  onCancel?: () => void;
}) {
  const [value, setValue] = useState("");

  const handleSubmit = () => {
    if (value.trim()) {
      onSubmit(value.trim());
      setValue("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div
      className="mt-2 rounded-xl border bg-white p-2"
      onClick={(e) => e.stopPropagation()}
    >
      <textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        className="min-h-[64px] w-full resize-none border-0 p-2 text-[14px] leading-[1.5] placeholder-gray-400 focus:ring-0 focus:outline-none"
        rows={3}
      />
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
}: {
  c: CommentDTO;
  depth?: number;
  onReply: (parentId: string, text: string) => void;
  activeComposer: string | null;
  setActiveComposer: (id: string | null) => void;
}) {
  const isOpen = activeComposer === c.id;

  return (
    <div className="space-y-2">
      <CommentItem c={c} />
      <div className="-mt-1 pl-10 text-[12px] text-neutral-600">
        <button
          className="hover:underline"
          onClick={(e) => {
            e.stopPropagation();
            setActiveComposer(isOpen ? null : c.id);
          }}
        >
          Reply
        </button>
      </div>
      {isOpen && (
        <div className="ml-6">
          <CommentComposer
            placeholder={`Reply to ${c.author.name || c.author.email?.split("@")[0] || "user"}…`}
            onSubmit={(text) => {
              onReply(c.id, text);
              setActiveComposer(null);
            }}
            onCancel={() => setActiveComposer(null)}
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
            />
          ))}
        </div>
      )}
    </div>
  );
}

function CommentsThread({
  comments,
  onReply,
  onReplyToPost,
  activeComposer,
  setActiveComposer,
}: {
  comments: CommentDTO[];
  onReply: (parentId: string, text: string) => void;
  onReplyToPost: (text: string) => void;
  activeComposer: string | null;
  setActiveComposer: (id: string | null) => void;
}) {
  const isRootComposerOpen = activeComposer === "root";

  return (
    <div className="mt-3 space-y-3" onClick={(e) => e.stopPropagation()}>
      {comments.map((c: CommentDTO) => (
        <CommentNode
          key={c.id}
          c={c}
          onReply={onReply}
          activeComposer={activeComposer}
          setActiveComposer={setActiveComposer}
        />
      ))}
      {/* Root-level composer */}
      <div className="pt-1">
        {isRootComposerOpen ? (
          <CommentComposer
            placeholder="Write a comment…"
            onSubmit={(text) => {
              onReplyToPost(text);
              setActiveComposer(null);
            }}
            onCancel={() => setActiveComposer(null)}
          />
        ) : (
          <button
            onClick={() => setActiveComposer("root")}
            className="w-full rounded-xl border bg-white p-4 text-left text-[14px] text-gray-400 hover:bg-gray-50"
          >
            Write a comment…
          </button>
        )}
      </div>
    </div>
  );
}

// --- Main component ---------------------------------------------------------
export const InPostComments: React.FC<InPostCommentsProps> = ({
  post,
  isExpanded,
  onToggle,
  currentUser,
}) => {
  const [comments, setComments] = useState<CommentDTO[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeComposer, setActiveComposer] = useState<string | null>(null);

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
    onSuccess: () => {
      // Refresh comments
      const formData = new FormData();
      formData.append("postId", post.id);
      executeLoadComments(formData);
    },
    onError: (error) => {
      console.error("Error creating comment:", error);
    },
  });

  // Load comments when expanded
  useEffect(() => {
    if (isExpanded && comments.length === 0) {
      setIsLoading(true);
      const formData = new FormData();
      formData.append("postId", post.id);
      executeLoadComments(formData);
    }
  }, [isExpanded, executeLoadComments, comments.length, post.id]);

  // Close active composer when comments are collapsed
  useEffect(() => {
    if (!isExpanded) {
      setActiveComposer(null);
    }
  }, [isExpanded]);

  const totalComments = countComments(comments);

  function addReply(parentId: string | null, text: string) {
    if (!currentUser) return;

    const formData = new FormData();
    formData.append("postId", post.id);
    formData.append("content", text);
    if (parentId) {
      formData.append("parentId", parentId);
    }
    executeCreateComment(formData);
  }

  return (
    <>
      {/* Comments thread when expanded */}
      {isExpanded && (
        <div className="px-4 pb-4">
          {isLoading ? (
            <div className="py-4 text-center">
              <div className="inline-block h-4 w-4 animate-spin rounded-full border-b-2 border-neutral-900"></div>
            </div>
          ) : currentUser ? (
            <CommentsThread
              comments={comments}
              onReply={(parentId: string, text: string) =>
                addReply(parentId, text)
              }
              onReplyToPost={(text: string) => addReply(null, text)}
              activeComposer={activeComposer}
              setActiveComposer={setActiveComposer}
            />
          ) : (
            <div className="py-4 text-center text-[14px] text-neutral-600">
              Sign in to join the conversation
            </div>
          )}
        </div>
      )}
    </>
  );
};
