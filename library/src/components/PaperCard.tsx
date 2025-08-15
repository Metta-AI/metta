"use client";

import { FC, useState } from "react";

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
  const [showAbstract, setShowAbstract] = useState(false);

  if (!paper) return null;

  const handleTitleClick = () => {
    if (onPaperClick) {
      onPaperClick(paper.id);
    }
  };

  const toggleAbstract = () => {
    setShowAbstract(!showAbstract);
  };

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      {/* Header with star widget, title and abstract toggle */}
      <div className="mb-3 flex items-start justify-between">
        <div className="flex min-w-0 flex-1 items-start gap-2">
          <div className="mt-1 flex-shrink-0">
            <StarWidgetQuery
              paperId={paper.id}
              initialTotalStars={paper.stars}
              initialIsStarredByCurrentUser={paper.starred}
              size="md"
              readonly={true}
            />
          </div>
          {/* Paper title - clickable */}
          <h3
            className="cursor-pointer text-lg font-semibold text-gray-900 transition-colors hover:text-blue-600"
            onClick={handleTitleClick}
          >
            {paper.title}
          </h3>
        </div>

        {/* Abstract toggle button in upper right */}
        {paper.abstract && (
          <button
            onClick={toggleAbstract}
            className="ml-4 flex-shrink-0 cursor-pointer text-xs font-medium text-blue-600 hover:text-blue-700"
            aria-label={showAbstract ? "Hide abstract" : "Show abstract"}
          >
            {showAbstract ? "Hide Abstract" : "Show Abstract"}
          </button>
        )}
      </div>

      {/* Paper metadata */}
      <div className="mb-3 flex items-center gap-4 text-xs text-gray-500">
        {paper.source && (
          <>
            {paper.link ? (
              <a
                href={paper.link}
                target="_blank"
                rel="noopener noreferrer"
                className="transition-colors hover:text-blue-600"
              >
                {paper.source}
              </a>
            ) : (
              <span>{paper.source}</span>
            )}
          </>
        )}
        {paper.institutions && paper.institutions.length > 0 && (
          <>
            {paper.source && <span>•</span>}
            <span>{paper.institutions.join(", ")}</span>
          </>
        )}
      </div>

      {/* Authors */}
      {paper.authors && paper.authors.length > 0 && (
        <div className="mb-3 text-sm text-gray-700">
          <span className="font-medium">Authors:</span>{" "}
          {paper.authors.map((author) => author.name).join(", ")}
        </div>
      )}

      {/* Abstract content */}
      {showAbstract && paper.abstract && (
        <div className="mb-3 border-t border-gray-100 pt-3">
          <p className="text-sm leading-relaxed whitespace-pre-wrap text-gray-700">
            {paper.abstract}
          </p>
        </div>
      )}

      {/* Tags */}
      {paper.tags && paper.tags.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {paper.tags.map((tag, index) => (
            <button
              key={index}
              onClick={(e) => {
                e.stopPropagation(); // Prevent triggering parent click handlers
                const params = new URLSearchParams();
                params.set("search", tag);
                window.open(`/papers?${params.toString()}`, "_blank");
              }}
              className="cursor-pointer rounded-full bg-gray-100 px-2 py-1 text-xs text-gray-700 transition-colors hover:bg-gray-200"
              title={`Click to view all papers tagged with "${tag}"`}
            >
              {tag}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};
