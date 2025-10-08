"use client";

import { FC, useEffect, useState } from "react";
import Link from "next/link";
import { X } from "lucide-react";

import { getUserDisplayName, getUserInitials } from "@/lib/utils/user";

interface QuotedPost {
  id: string;
  title: string;
  content: string | null;
  author: {
    id: string;
    name: string | null;
    email: string | null;
    image: string | null;
  };
  createdAt: Date;
}

interface QuotedPostPreviewProps {
  quotedPostIds: string[];
  onRemove?: (postId: string) => void;
}

export const QuotedPostPreview: FC<QuotedPostPreviewProps> = ({
  quotedPostIds,
  onRemove,
}) => {
  const [quotedPosts, setQuotedPosts] = useState<QuotedPost[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchQuotedPosts = async () => {
      if (quotedPostIds.length === 0) {
        setQuotedPosts([]);
        return;
      }

      setLoading(true);
      try {
        // Fetch posts data from the API
        const promises = quotedPostIds.map(async (id) => {
          const response = await fetch(`/api/posts/${id}`);
          if (response.ok) {
            return await response.json();
          }
          return null;
        });

        const results = await Promise.all(promises);
        const validPosts = results.filter(Boolean);
        setQuotedPosts(validPosts);
      } catch (error) {
        console.error("Error fetching quoted posts:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchQuotedPosts();
  }, [quotedPostIds]);

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
