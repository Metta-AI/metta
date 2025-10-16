"use client";

import Link from "next/link";
import { getUserDisplayName } from "@/lib/utils/user";
import { formatRelativeTimeCompact } from "@/lib/utils/date";

interface QuotedPostCardProps {
  quotedPost: {
    id: string;
    content: string | null;
    createdAt: Date | string;
    author: {
      name: string | null;
      email: string | null;
      image?: string | null;
    };
  };
}

export function QuotedPostCard({ quotedPost }: QuotedPostCardProps) {
  return (
    <div
      className="rounded-lg border border-gray-300 bg-gray-50 p-3"
      onClick={(e) => e.stopPropagation()}
    >
      <div className="flex items-start gap-3">
        {/* Author avatar */}
        <div className="flex-shrink-0">
          {quotedPost.author.image ? (
            <img
              src={quotedPost.author.image}
              alt={quotedPost.author.name || "User"}
              className="h-6 w-6 rounded-full"
            />
          ) : (
            <div className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-500 text-xs font-medium text-white">
              {(
                quotedPost.author.name ||
                quotedPost.author.email?.split("@")[0] ||
                "U"
              )
                .split(" ")
                .map((n) => n[0])
                .join("")
                .toUpperCase()
                .slice(0, 2)}
            </div>
          )}
        </div>

        {/* Quoted post content */}
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-gray-900">
              {getUserDisplayName(
                quotedPost.author.name,
                quotedPost.author.email
              )}
            </span>
            <span className="text-xs text-gray-500">
              {formatRelativeTimeCompact(quotedPost.createdAt)}
            </span>
          </div>

          {quotedPost.content && (
            <p className="mt-1 line-clamp-3 text-sm text-gray-700">
              {quotedPost.content}
            </p>
          )}

          <Link
            href={`/posts/${quotedPost.id}`}
            className="mt-2 inline-block text-xs text-blue-600 hover:text-blue-800"
            onClick={(e) => e.stopPropagation()}
          >
            View original post →
          </Link>
        </div>
      </div>
    </div>
  );
}
