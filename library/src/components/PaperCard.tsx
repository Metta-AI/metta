"use client";

import { FC } from "react";

import { FeedPostDTO } from "@/posts/data/feed";
import { StarWidgetQuery } from "./StarWidgetQuery";

interface PaperCardProps {
  paper: FeedPostDTO["paper"];
  onPaperClick?: (paperId: string) => void;
}

/**
 * PaperCard Component
 *
 * Displays a paper in a card format with:
 * - Clickable title that opens paper overlay
 * - Expandable abstract
 * - Paper metadata (authors, institutions, source)
 * - Star count and interaction buttons
 */
export const PaperCard: FC<PaperCardProps> = ({ paper, onPaperClick }) => {
  if (!paper) return null;

  const handleTitleClick = () => {
    if (onPaperClick) {
      onPaperClick(paper.id);
    }
  };

  return (
    <div className="rounded-xl border border-neutral-200 bg-white p-2.5">
      {/* Header with star widget and title */}
      <div className="flex items-start gap-2">
        <div className="mt-0.5 shrink-0">
          <StarWidgetQuery
            paperId={paper.id}
            initialTotalStars={paper.stars}
            initialIsStarredByCurrentUser={paper.starred}
            size="sm"
            readonly={true}
          />
        </div>
        <div className="min-w-0">
          {/* Paper title - clickable */}
          <h3
            className="cursor-pointer text-[15.5px] leading-[1.3] font-semibold tracking-tight text-neutral-900 transition-colors hover:text-blue-600"
            onClick={handleTitleClick}
          >
            {paper.title}
          </h3>

          {/* Tags */}
          {paper.tags && paper.tags.length > 0 && (
            <div className="mt-1 flex flex-wrap gap-1">
              {paper.tags.map((tag, index) => (
                <button
                  key={index}
                  onClick={(e) => {
                    e.stopPropagation(); // Prevent triggering parent click handlers
                    const params = new URLSearchParams();
                    params.set("search", tag);
                    window.open(`/papers?${params.toString()}`, "_blank");
                  }}
                  className="cursor-pointer rounded-md px-2 py-0.5 text-[12px] font-bold transition-colors hover:bg-neutral-200"
                  style={{ backgroundColor: "#EFF3F9", color: "#131720" }}
                  title={`Click to view all papers tagged with "${tag}"`}
                >
                  {tag}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
