"use client";

import { FC, useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";

import { FeedPostDTO } from "@/posts/data/feed";
import { StarWidgetQuery } from "./StarWidgetQuery";
import { LLMAbstract } from "@/lib/llm-abstract-generator-clean";

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

  const [isExpanded, setIsExpanded] = useState(false);

  const handleTitleClick = () => {
    if (onPaperClick) {
      onPaperClick(paper.id);
    }
  };

  const handleExpandClick = (e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent triggering parent click handlers
    setIsExpanded(!isExpanded);
  };

  // Extract AI summary data
  const llmAbstract = paper.llmAbstract as LLMAbstract | null;

  return (
    <div className="rounded-xl border border-neutral-200 bg-white p-3 md:p-2.5">
      {/* Header with star widget and title */}
      <div className="flex items-start gap-2 md:gap-3">
        <div className="mt-0.5 shrink-0">
          <StarWidgetQuery
            paperId={paper.id}
            initialTotalStars={paper.stars}
            initialIsStarredByCurrentUser={paper.starred}
            size="sm"
            readonly={true}
          />
        </div>
        <div className="min-w-0 flex-1">
          {/* Paper title - clickable */}
          <h3
            className="cursor-pointer text-[15.5px] leading-[1.3] font-semibold tracking-tight text-neutral-900 transition-colors hover:text-blue-600"
            onClick={handleTitleClick}
          >
            {paper.title}
          </h3>

          {/* Tags */}
          {paper.tags && paper.tags.length > 0 && (
            <div className="mt-1.5 flex flex-wrap gap-1">
              {paper.tags.map((tag, index) => (
                <button
                  key={index}
                  onClick={(e) => {
                    e.stopPropagation(); // Prevent triggering parent click handlers
                    const params = new URLSearchParams();
                    params.set("search", tag);
                    window.open(`/papers?${params.toString()}`, "_blank");
                  }}
                  className="cursor-pointer rounded-md px-2 py-1 text-xs font-bold transition-colors hover:bg-neutral-200 md:py-0.5 md:text-[12px]"
                  style={{ backgroundColor: "#EFF3F9", color: "#131720" }}
                  title={`Click to view all papers tagged with "${tag}"`}
                >
                  {tag}
                </button>
              ))}
            </div>
          )}

          {/* AI Summary Section */}
          {llmAbstract?.shortExplanation && (
            <div className="mt-2">
              <div className="text-[13px] leading-[1.4] text-neutral-700">
                {isExpanded
                  ? llmAbstract.summary || llmAbstract.shortExplanation
                  : `${llmAbstract.shortExplanation.slice(0, 120)}${llmAbstract.shortExplanation.length > 120 ? "..." : ""}`}
              </div>

              {/* Expand/Collapse Button */}
              {(llmAbstract.summary ||
                llmAbstract.shortExplanation.length > 120) && (
                <button
                  onClick={handleExpandClick}
                  className="mt-1 flex items-center gap-1 text-[11px] font-medium text-neutral-500 transition-colors hover:text-neutral-700"
                >
                  {isExpanded ? (
                    <>
                      <ChevronUp className="h-3 w-3" />
                      Show less
                    </>
                  ) : (
                    <>
                      <ChevronDown className="h-3 w-3" />
                      Show more
                    </>
                  )}
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
