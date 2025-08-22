"use client";

import { FC } from "react";
import Link from "next/link";

import { FeedPostDTO } from "@/posts/data/feed";
import { linkifyText } from "@/lib/utils/linkify";
import { ThreadedComments } from "@/components/ThreadedComments";

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
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-slate-200 text-sm font-semibold text-slate-700">
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
          <span>{post.replies} replies</span>
          <span>{post.queues} queues</span>
        </div>
      </div>

      {/* Comments section using unified ThreadedComments component */}
      <div className="rounded-lg border border-gray-200 bg-white p-4">
        <ThreadedComments
          postId={post.id}
          currentUser={currentUser}
          showBackToFeed={true}
        />
      </div>
    </div>
  );
};
