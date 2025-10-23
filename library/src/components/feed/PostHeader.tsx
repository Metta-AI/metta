"use client";

import { getUserInitials, getUserDisplayName } from "@/lib/utils/user";
import { formatRelativeTimeCompact } from "@/lib/utils/date";

interface PostHeaderProps {
  author: {
    id: string;
    name: string | null;
    email: string | null;
    image?: string | null;
  };
  createdAt: Date | string;
  onUserClick?: (userId: string) => void;
}

export function PostHeader({
  author,
  createdAt,
  onUserClick,
}: PostHeaderProps) {
  return (
    <div className="flex items-center gap-3">
      <button
        onClick={(e) => {
          e.stopPropagation();
          onUserClick?.(author.id);
        }}
        className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-full text-sm font-medium transition-colors hover:bg-neutral-100"
        style={{ backgroundColor: "#EFF3F9", color: "#131720" }}
      >
        {getUserInitials(author.name, author.email)}
      </button>
      <div className="flex items-center gap-2 text-[12.5px] leading-5 text-neutral-600">
        <button
          onClick={(e) => {
            e.stopPropagation();
            onUserClick?.(author.id);
          }}
          className="cursor-pointer font-medium text-neutral-900 transition-colors hover:text-blue-600"
        >
          {getUserDisplayName(author.name, author.email)}
        </button>
        <span>•</span>
        <span>{formatRelativeTimeCompact(createdAt)}</span>
      </div>
    </div>
  );
}
