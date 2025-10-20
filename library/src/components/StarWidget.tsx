"use client";

import { FC } from "react";

interface StarWidgetProps {
  totalStars: number;
  isStarredByCurrentUser: boolean;
  onClick?: () => void;
  size?: "sm" | "md" | "lg" | "xl";
}

/**
 * StarWidget Component
 *
 * A reusable star widget with three states:
 * 1. Empty gray outline (no one starred it)
 * 2. Gray outline with numeral inside (others starred, not you)
 * 3. Yellow filled star with numeral inside (you and 0+ others starred)
 */
export const StarWidget: FC<StarWidgetProps> = ({
  totalStars,
  isStarredByCurrentUser,
  onClick,
  size = "md",
}) => {
  const sizeClasses = {
    sm: "h-4 w-4",
    md: "h-5 w-5",
    lg: "h-6 w-6",
    xl: "h-7 w-7",
  };

  const textSizeClasses = {
    sm: "text-[7px]",
    md: "text-[9px]",
    lg: "text-[10px]",
    xl: "text-xs",
  };

  const starSize = sizeClasses[size];
  const textSize = textSizeClasses[size];

  const renderStarWithCount = () => {
    if (isStarredByCurrentUser) {
      // Starred by current user - warm gold with elegant styling
      const displayCount = Math.max(totalStars, 1);
      return (
        <div className="relative transition-all duration-200 hover:scale-105">
          <svg
            className={`${starSize} text-yellow-500 drop-shadow-sm hover:text-yellow-600`}
            fill="currentColor"
            viewBox="0 0 20 20"
          >
            <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.286 3.967a1 1 0 00.95.69h4.175c.969 0 1.371 1.24.588 1.81l-3.38 2.455a1 1 0 00-.364 1.118l1.287 3.966c.3.922-.755 1.688-1.54 1.118l-3.38-2.454a1 1 0 00-1.175 0l-3.38 2.454c-.784.57-1.838-.196-1.54-1.118l1.287-3.966a1 1 0 00-.364-1.118L2.05 9.394c-.783-.57-.38-1.81.588-1.81h4.175a1 1 0 00.95-.69l1.286-3.967z" />
          </svg>
          {displayCount > 0 && (
            <div
              className={`absolute inset-0 flex items-center justify-center ${textSize} leading-none font-bold text-yellow-900 select-none`}
              style={{ paddingTop: "1px" }}
            >
              {displayCount}
            </div>
          )}
        </div>
      );
    } else if (totalStars > 0) {
      // Starred by others - sophisticated slate with refined outline
      return (
        <div className="relative transition-all duration-200 hover:scale-105">
          <svg
            className={`${starSize} text-slate-400 hover:text-indigo-400`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 20 20"
            strokeWidth="1.5"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M10 15l-5.878 3.09 1.122-6.545L.488 6.91l6.564-.955L10 0l2.948 5.955 6.564.955-4.756 4.635 1.122 6.545z"
            />
          </svg>
          <div
            className={`absolute inset-0 flex items-center justify-center ${textSize} leading-none font-bold text-slate-700 select-none`}
            style={{ paddingTop: "1px" }}
          >
            {totalStars}
          </div>
        </div>
      );
    } else {
      // No stars - gentle and inviting
      return (
        <div className="relative transition-all duration-300 hover:scale-105">
          <svg
            className={`${starSize} text-slate-300 hover:text-indigo-400`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 20 20"
            strokeWidth="1.5"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M10 15l-5.878 3.09 1.122-6.545L.488 6.91l6.564-.955L10 0l2.948 5.955 6.564.955-4.756 4.635 1.122 6.545z"
            />
          </svg>
        </div>
      );
    }
  };

  if (onClick) {
    return (
      <button
        onClick={onClick}
        className="inline-flex items-center justify-center transition-all duration-200 focus:outline-none"
        title={
          isStarredByCurrentUser ? "Remove from favorites" : "Add to favorites"
        }
      >
        {renderStarWithCount()}
      </button>
    );
  }

  return renderStarWithCount();
};
