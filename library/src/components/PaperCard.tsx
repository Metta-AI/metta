"use client";

import { FC, useState } from "react";

import { FeedPostDTO } from "@/posts/data/feed";

interface PaperCardProps {
  paper: FeedPostDTO['paper'];
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
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      {/* Header with title and abstract toggle */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1 min-w-0">
          {/* Paper title - clickable */}
          <h3 
            className="text-lg font-semibold text-gray-900 mb-2 cursor-pointer hover:text-blue-600 transition-colors"
            onClick={handleTitleClick}
          >
            {paper.title}
          </h3>
        </div>
        
        {/* Abstract toggle button in upper right */}
        {paper.abstract && (
          <button
            onClick={toggleAbstract}
            className="text-xs text-blue-600 hover:text-blue-700 font-medium ml-4 flex-shrink-0"
          >
            {showAbstract ? 'Hide Abstract' : 'Show Abstract'}
          </button>
        )}
      </div>

      {/* Paper metadata */}
      <div className="flex items-center gap-4 text-xs text-gray-500 mb-3">
        {paper.source && (
          <>
            {paper.link ? (
              <a 
                href={paper.link} 
                target="_blank" 
                rel="noopener noreferrer"
                className="hover:text-blue-600 transition-colors"
              >
                {paper.source}
              </a>
            ) : (
              <span>{paper.source}</span>
            )}
            <span>•</span>
          </>
        )}
        <span>{paper.stars} stars</span>
        {paper.institutions && paper.institutions.length > 0 && (
          <>
            <span>•</span>
            <span>{paper.institutions.join(', ')}</span>
          </>
        )}
      </div>

      {/* Authors */}
      {paper.authors && paper.authors.length > 0 && (
        <div className="text-sm text-gray-700 mb-3">
          <span className="font-medium">Authors:</span> {paper.authors.join(', ')}
        </div>
      )}

      {/* Abstract content */}
      {showAbstract && paper.abstract && (
        <div className="border-t border-gray-100 pt-3 mb-3">
          <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">
            {paper.abstract}
          </p>
        </div>
      )}

      {/* Tags */}
      {paper.tags && paper.tags.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {paper.tags.map((tag, index) => (
            <span
              key={index}
              className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-full"
            >
              {tag}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}; 