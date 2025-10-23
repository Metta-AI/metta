"use client";

import React from "react";
import { FeedPostDTO } from "@/posts/data/feed";
import { ThreadedComments } from "@/components/ThreadedComments";

interface InPostCommentsProps {
  post: FeedPostDTO;
  isExpanded: boolean;
  onToggle: () => void;
  currentUser: {
    id: string;
    name?: string | null;
    email?: string | null;
  } | null;
  onCommentCountChange?: (delta: number) => void;
  highlightedCommentId?: string | null;
}

// --- Main component ---------------------------------------------------------
export const InPostComments: React.FC<InPostCommentsProps> = ({
  post,
  isExpanded,
  onToggle,
  currentUser,
  onCommentCountChange,
  highlightedCommentId,
}) => {
  return (
    <>
      {/* Comments thread when expanded */}
      {isExpanded && (
        <div className="px-4 pb-4">
          <ThreadedComments
            postId={post.id}
            currentUser={currentUser}
            showBackToFeed={false}
            onCommentCountChange={onCommentCountChange}
            highlightedCommentId={highlightedCommentId}
          />
        </div>
      )}
    </>
  );
};
