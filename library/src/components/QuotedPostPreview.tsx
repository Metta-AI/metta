"use client";

import { FC } from "react";
import Link from "next/link";
import { X } from "lucide-react";
import { useQueries } from "@tanstack/react-query";

import { getUserDisplayName, getUserInitials } from "@/lib/utils/user";
import * as postsApi from "@/lib/api/resources/posts";
import type { PostDetail } from "@/lib/api/resources/posts";

interface QuotedPostPreviewProps {
  quotedPostIds: string[];
  onRemove?: (postId: string) => void;
}

export const QuotedPostPreview: FC<QuotedPostPreviewProps> = ({
  quotedPostIds,
  onRemove,
}) => {
  // Use React Query's useQueries to fetch multiple posts efficiently
  const postQueries = useQueries({
    queries: quotedPostIds.map((id) => ({
      queryKey: ["posts", id],
      queryFn: () => postsApi.getPost(id),
      enabled: !!id,
    })),
  });

  const loading = postQueries.some((query) => query.isLoading);
  const quotedPosts = postQueries
    .map((query) => query.data)
    .filter(Boolean) as PostDetail[];

  if (quotedPostIds.length === 0) {
    return null;
  }

  return (
    <div className="space-y-3">
      {quotedPosts.length === 1 && (
        <div className="mb-2 text-right text-xs text-gray-500">
          You can quote 1 more post by pasting its URL
        </div>
      )}
      {loading ? (
        <div className="text-sm text-gray-500">Loading quoted posts...</div>
      ) : (
        <div className="space-y-2">
          {quotedPosts.map((post) => (
            <div
              key={post.id}
              className="flex items-start gap-3 rounded-lg border border-gray-200 bg-gray-50 p-3"
            >
              {/* Author avatar */}
              <div className="flex-shrink-0">
                {post.author.image ? (
                  <img
                    src={post.author.image}
                    alt={getUserDisplayName(
                      post.author.name,
                      post.author.email
                    )}
                    className="h-8 w-8 rounded-full"
                  />
                ) : (
                  <div className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-500 text-xs font-medium text-white">
                    {getUserInitials(post.author.name, post.author.email)}
                  </div>
                )}
              </div>

              {/* Post content */}
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-gray-900">
                    {getUserDisplayName(post.author.name, post.author.email)}
                  </span>
                  <span className="text-xs text-gray-500">
                    {new Date(post.createdAt).toLocaleDateString()}
                  </span>
                </div>

                {post.content && (
                  <p className="mt-1 line-clamp-2 text-sm text-gray-700">
                    {post.content.length > 100
                      ? post.content.slice(0, 100) + "..."
                      : post.content}
                  </p>
                )}

                <Link
                  href={`/posts/${post.id}`}
                  className="mt-2 inline-block text-xs text-blue-600 hover:text-blue-800"
                >
                  View full post →
                </Link>
              </div>

              {/* Remove button */}
              {onRemove && (
                <button
                  onClick={() => onRemove(post.id)}
                  className="flex-shrink-0 rounded-full p-1 text-gray-400 hover:bg-gray-200 hover:text-gray-600"
                  title="Remove from quote"
                >
                  <X className="h-4 w-4" />
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
