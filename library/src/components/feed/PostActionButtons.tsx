"use client";

import Link from "next/link";
import { MessageSquare, ExternalLink, Quote } from "lucide-react";

interface PostActionButtonsProps {
  postId: string;
  commentCount: number;
  canDelete: boolean | null;
  isPurePaper?: boolean;
  paperId?: string;
  optimisticQueues: number;
  optimisticQueued: boolean;
  isQueuePending: boolean;
  isDeletePending: boolean;
  onQueue?: () => void;
  onDelete?: () => void;
  onQuote?: (e: React.MouseEvent) => void;
  onCommentClick?: (e: React.MouseEvent) => void;
}

export function PostActionButtons({
  postId,
  commentCount,
  canDelete,
  isPurePaper = false,
  paperId,
  optimisticQueues,
  optimisticQueued,
  isQueuePending,
  isDeletePending,
  onQueue,
  onDelete,
  onQuote,
  onCommentClick,
}: PostActionButtonsProps) {
  return (
    <div className="flex items-center gap-2">
      {/* Permalink button - only for non-pure-paper posts */}
      {!isPurePaper && (
        <Link
          href={`/posts/${postId}`}
          onClick={(e) => e.stopPropagation()}
          className="rounded-full p-1 text-neutral-400 opacity-0 transition-all group-hover:opacity-100 hover:bg-neutral-100 hover:text-blue-500"
          title="Open post"
        >
          <ExternalLink className="h-4 w-4" />
        </Link>
      )}

      {/* Quote post button - only for non-pure-paper posts */}
      {!isPurePaper && onQuote && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onQuote(e);
          }}
          className="rounded-full p-1 text-neutral-400 opacity-0 transition-all group-hover:opacity-100 hover:bg-neutral-100 hover:text-green-500"
          title="Quote post (up to 2 posts can be quoted)"
        >
          <Quote className="h-4 w-4" />
        </button>
      )}

      {/* Queue button - only show for posts with papers */}
      {paperId && onQueue && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onQueue();
          }}
          disabled={isQueuePending}
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
      {canDelete && onDelete && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
          disabled={isDeletePending}
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

      {/* Comment button */}
      <button
        onClick={(e) => {
          e.stopPropagation();
          onCommentClick?.(e);
        }}
        className="rounded-full p-1 text-neutral-400 transition-colors hover:bg-neutral-100 hover:text-blue-500"
        title={`${commentCount} comments`}
      >
        <div className="flex items-center gap-1">
          <MessageSquare className="h-4 w-4" />
          <span className="text-[11px] tabular-nums">{commentCount}</span>
        </div>
      </button>
    </div>
  );
}
